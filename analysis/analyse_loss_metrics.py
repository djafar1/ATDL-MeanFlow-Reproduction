#!/usr/bin/env python3
"""Read recorded runs only; generate report assets without training or FID work."""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
# Mean Flows.pdf, printed page 8, Table 1(e), not the repository README table.
PAPER = {0.0: 79.75, 0.5: 63.98, 1.0: 61.06, 1.5: 66.57, 2.0: 69.19}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
MARKERS = ["o", "s", "^", "D", "v"]


class Inputs:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def read(self, relative):
        data = (self.root / relative).read_bytes()
        self.hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, relative):
        return json.loads(self.read(relative))

    def csv(self, relative):
        return list(csv.DictReader(io.StringIO(self.read(relative).decode())))

    def text(self, relative):
        return self.read(relative).decode().strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def controlled_config(config):
    result = copy.deepcopy(config)
    del result["method"]["norm_p"]
    del result["training"]["seed"]
    return result


def config_for(inputs, directory, p, seed):
    c = inputs.json(directory / "effective-config.json")
    expected = {
        "method": {"norm_p": p, "norm_eps": 0.01, "guidance_eq": "none",
                   "noise_dist": "logit_normal", "P_mean": -0.4, "P_std": 1.0,
                   "data_proportion": 0.75, "class_dropout_prob": 0.1},
        "model": {"cls": "DiT_B_4"},
        "dataset": {"name": "imgnet_latent", "image_size": 32,
                    "image_channels": 4, "num_classes": 1000, "vae": "mse"},
        "training": {"seed": seed, "num_epochs": 80, "batch_size": 256,
                     "learning_rate": 1e-4, "adam_b2": 0.95, "ema_type": "const",
                     "ema_val": 0.9999, "half_precision": False},
        "sampling": {"seed": 0, "num_steps": 1},
        "fid": {"num_samples": 50000},
    }
    require(c["load_from"] is None and c["eval_only"] is False,
            f"{directory}: expected training from scratch")
    for group, values in expected.items():
        for key, value in values.items():
            require(c[group][key] == value, f"{directory}: unexpected {group}.{key}")
    return c


def completed_run(inputs, directory, p, seed, summary):
    require(summary["status"] == "completed" and summary["exit_code"] == 0,
            f"{directory}: no successful completion")
    require(summary["epochs_completed"] == 80 and summary["final_train_step"] == 400320,
            f"{directory}: incomplete training endpoint")
    require(summary["p"] == p, f"{directory}: summary p mismatch")
    require(inputs.text(directory / "exit-status.txt") == "0" and summary.get("error") is None,
            f"{directory}: unsuccessful process exit")
    started = inputs.text(directory / "start.txt")
    ended = inputs.text(directory / "end.txt")
    require(datetime.fromisoformat(ended) > datetime.fromisoformat(started),
            f"{directory}: invalid process timestamps")
    fids = inputs.csv(directory / "fid.csv")
    epochs = [int(row["epoch"]) for row in fids]
    require(len(epochs) == len(set(epochs)) and epochs == sorted(epochs),
            f"{directory}: duplicate/unordered FID epochs")
    for row in fids:
        epoch = int(row["epoch"])
        require(0 < epoch <= 80 and int(row["train_step"]) == epoch * 5004,
                f"{directory}: FID step mismatch")
        require(int(row["num_samples"]) == 50000 and int(row["sampling_seed"]) == 0
                and int(row["NFE"]) == 1 and row["EMA"].lower() == "true",
                f"{directory}: evaluation protocol mismatch")
        require(math.isfinite(float(row["FID_ema"])), f"{directory}: nonfinite FID")
    final = [row for row in fids if int(row["epoch"]) == 80]
    require(len(final) == 1, f"{directory}: missing epoch-80 FID (transfer incomplete?)")
    fid = float(final[0]["FID_ema"])
    require(math.isclose(fid, summary["final_FID_epoch80"], rel_tol=0, abs_tol=1e-9),
            f"{directory}: summary/CSV endpoint disagreement")
    summary_evaluations = summary["FID_evaluations"]
    require([row["epoch"] for row in summary_evaluations] == epochs,
            f"{directory}: summary/CSV evaluation epochs disagree")
    for recorded, row in zip(summary_evaluations, fids):
        require(recorded["EMA"] is True and recorded["NFE"] == 1
                and recorded["num_samples"] == 50000 and recorded["sampling_seed"] == 0
                and recorded["train_step"] == int(row["train_step"])
                and math.isclose(recorded["FID_ema"], float(row["FID_ema"]),
                                 rel_tol=0, abs_tol=1e-9),
                f"{directory}: summary/CSV evaluation disagreement")
    metrics = inputs.csv(directory / "metrics.csv")
    logged_final = [float(row["FID_ema"]) for row in metrics
                    if row["kind"] == "fid" and int(row["epoch"]) == 80]
    require(logged_final == [fid], f"{directory}: metrics/CSV endpoint disagreement")
    throughput = [float(row["steps_per_second"]) for row in metrics
                  if row["kind"] == "train" and int(row["train_step"]) >= 500
                  and float(row["steps_per_second"]) > 0]
    require(bool(throughput), f"{directory}: no measured throughput")
    median_throughput = statistics.median(throughput)
    require(math.isclose(median_throughput,
                        summary["median_logged_steps_per_second_after_step500"],
                        rel_tol=0, abs_tol=1e-9), f"{directory}: throughput disagreement")
    runtime = inputs.json(directory / "runtime.json")
    require(len(runtime["devices"]) == 1
            and runtime["devices"][0]["kind"] == "NVIDIA B200",
            f"{directory}: unexpected device configuration")
    command = inputs.json(directory / "command.json")
    require(command["environment"]["MEANFLOW_SINGLE_PROCESS"] == "1",
            f"{directory}: unexpected process mode")
    require(hashlib.sha256(inputs.read(directory / "config.yml")).hexdigest()
            == command["config_sha256"], f"{directory}: launch config hash mismatch")
    gpu_uuid = command["environment"]["CUDA_VISIBLE_DEVICES"]
    require(gpu_uuid.startswith("GPU-") and "," not in gpu_uuid,
            f"{directory}: expected one selected physical GPU")
    hours = float(summary["end_to_end_wall_seconds"]) / 3600
    require(hours > 0 and math.isclose(hours, summary["gpu_hours_for_this_one_gpu_run"],
                                      rel_tol=0, abs_tol=1e-9),
            f"{directory}: runtime disagreement")
    evaluations = sum(float(row["evaluation_seconds"]) for row in fids)
    require(math.isclose(evaluations, summary["total_FID_evaluation_seconds"],
                        rel_tol=0, abs_tol=1e-6),
            f"{directory}: evaluation timing disagreement")
    return {"p": p, "training_seed": seed, "epoch": 80, "paper_fid": PAPER[p],
            "fid": fid, "difference": fid - PAPER[p], "run_hours": hours,
            "fid_evaluation_seconds": evaluations,
            "median_logged_steps_per_second_from_step500": median_throughput,
            "directory": str(directory), "status": "verified_complete", "exit_code": 0,
            "epochs_completed": 80, "final_train_step": 400320,
            "started_utc": started, "ended_utc": ended,
            "command_started_utc": command["started_utc"], "physical_gpu_uuid": gpu_uuid,
            "packages": runtime["packages"], "runtime_settings": runtime, "curve": fids}


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path, rows, two_seeds=False):
    header = (r"$p$ & Paper & Seed 42 & Seed 43 & Mean $\pm$ SD & $\Delta$ \\") if two_seeds else (
        r"$p$ & Paper & Seed 42 & $\Delta$ (ours--paper) \\")
    lines = ["% Generated by analysis/analyse_loss_metrics.py; epoch 80 only."]
    if two_seeds:
        lines += ["% Delta = paper FID minus our two-seed mean; bold marks minimum FID.",
                  r"\begingroup", r"\setlength{\tabcolsep}{4pt}",
                  r"\begin{tabular}{@{}crrrrr@{}}"]
        minima = {key: min(row[key] for row in rows)
                  for key in ["paper_fid", "seed42", "seed43", "mean"]}
    else:
        lines.append(r"\begin{tabular}{crrr}")
    lines += [r"\hline", header, r"\hline"]
    for row in rows:
        p = row["p"]
        if two_seeds:
            cells = []
            for key in ["paper_fid", "seed42", "seed43"]:
                value = f"{row[key]:.2f}"
                cells.append(r"\textbf{" + value + "}" if row[key] == minima[key] else value)
            mean = f"{row['mean']:.2f}"
            if row["mean"] == minima["mean"]:
                mean = r"\mathbf{" + mean + "}"
            values = (f"{p:.1f} & " + " & ".join(cells)
                      + f" & ${mean} \\pm {row['sample_sd']:.2f}$"
                      + f" & ${row['paper_minus_mean']:+.2f}$")
        else:
            values = f"{p:.1f} & {PAPER[p]:.2f} & {row['fid']:.2f} & ${row['difference']:+.2f}$"
        lines.append(values + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    if two_seeds:
        lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n")


def save_plot(fig, path):
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plots(runs, second, directory):
    # Physical size matches a single ISBI column; labels are readable at 100%.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    if len(second) == len(PAPER):
        from paper_seed_figure import draw, styling
        styling()
        fig = draw(runs, second, "recorded")
    else:
        # Preserve a measured seed-42 figure when a refreshed transfer is incomplete.
        fig, ax = plt.subplots(figsize=(3.4, 2.55), layout="constrained")
        for run, color, marker in zip(runs, COLORS, MARKERS):
            rows = run["curve"]
            ax.plot([int(x["epoch"]) for x in rows], [float(x["FID_ema"]) for x in rows],
                    color=color, marker=marker, markersize=4, linewidth=1.5,
                    label=f"$p={run['p']:g}$")
        ax.set(xlabel="Training epoch", ylabel="EMA FID-50K",
               xticks=list(range(10, 81, 10)))
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, frameon=False, fontsize=10, handlelength=1.3,
                  columnspacing=0.7, labelspacing=0.15, loc="upper right")
    save_plot(fig, directory / "loss_metrics_fid_epochs")
    fig, ax = plt.subplots(figsize=(3.4, 2.55), layout="constrained")
    ps = list(PAPER)
    ax.plot(ps, list(PAPER.values()), "o--", color="#555555", label="Paper, Table 1(e)")
    ax.plot(ps, [x["fid"] for x in runs], "s-", color=COLORS[0], label="Seed 42")
    if second:
        by_p = {x["p"]: x["fid"] for x in second}
        ax.plot(list(by_p), list(by_p.values()), "D--", color=COLORS[4],
                markerfacecolor="white", label="Seed 43")
    ax.set(xlabel="Adaptive weighting exponent $p$", ylabel="Epoch-80 EMA FID-50K",
           xticks=ps)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=10)
    save_plot(fig, directory / "loss_metrics_fid_vs_p")


def cost_summary(runs):
    """Each run occupies one GPU; this is run duration, not an allocation invoice."""
    return {"verified_runs": len(runs), "summed_run_gpu_hours": sum(x["run_hours"] for x in runs),
            "run_hours_range": [min(x["run_hours"] for x in runs), max(x["run_hours"] for x in runs)],
            "fid_evaluation_minutes_range": [min(x["fid_evaluation_seconds"] for x in runs) / 60,
                                             max(x["fid_evaluation_seconds"] for x in runs) / 60],
            "median_logged_updates_per_second_range": [
                min(x["median_logged_steps_per_second_from_step500"] for x in runs),
                max(x["median_logged_steps_per_second_from_step500"] for x in runs)]}


def parallel_timing(inputs, seed, runs, evidence):
    if seed == 42:
        controller = evidence["parallel-run-timing"]
        seconds = float(controller["controller_wall_seconds"])
        scope = "Recorded monotonic controller duration of the seed-42 sweep."
    else:
        controller = inputs.json(Path("provenance/seed43-all-controller-summary.json"))
        require(controller["success"] is True and controller["completed_seed43_runs"] == 5,
                "Seed-43 controller does not confirm completion of all five arms")
        # There is no recorded controller duration/start for this sweep. Name this UTC
        # interval explicitly; do not equate plan creation or preprocessing with training.
        start = min(datetime.fromisoformat(x["command_started_utc"]) for x in runs)
        finish = datetime.fromisoformat(controller["finished_utc"])
        seconds = (finish - start).total_seconds()
        require(finish >= max(datetime.fromisoformat(x["ended_utc"]) for x in runs),
                "Seed-43 controller timestamp predates a run's exit")
        scope = "UTC interval from first recorded seed-43 launch to controller completion."
    require(controller["launcher_exit_codes"] == {f"{p:.1f}": 0 for p in PAPER},
            f"Seed-{seed} controller has missing/nonzero launcher exits")
    require(len({x["physical_gpu_uuid"] for x in runs}) == 5 and seconds > 0,
            f"Seed-{seed}: invalid parallel GPU/time records")
    return {"elapsed_parallel_hours": seconds / 3600, "timing_scope": scope,
            "five_gpu_occupied_block_hours_estimate": 5 * seconds / 3600,
            "allocated_or_billed_gpu_hours": None,
            "billing_scope": "Allocation duration and invoice unavailable; idle GPUs/time may be billed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=REPORT.parent / "meanflow-table1e-light")
    parser.add_argument("--output-root", type=Path, default=REPORT)
    args = parser.parse_args()
    inputs = Inputs(args.data_root.resolve())
    manifest = inputs.json(Path("provenance/reproducibility-manifest.json"))
    patch = inputs.read(Path("provenance/full-run-gpu-compat.patch"))
    require(hashlib.sha256(patch).hexdigest() == manifest["source_patch_sha256"],
            "Source patch does not match recorded manifest")
    archive = inputs.read(Path("provenance/full-run-source.tar.gz"))
    require(hashlib.sha256(archive).hexdigest() == manifest["source_archive_sha256"],
            "Source archive does not match recorded manifest")
    source_files = manifest["tracked_source_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for name in source_files:
            data = inputs.read(Path("meanflow") / name)
            require(data == tar.extractfile("meanflow/" + name).read()
                    and hashlib.sha256(data).hexdigest() == manifest["tracked_source_sha256"][name],
                    f"{name}: current source differs from archived execution")
    evidence = {name: inputs.json(Path("provenance") / (name + ".json"))
                for name in ["fid-reference", "vae-snapshot", "latent-cache-verification",
                             "parallel-run-timing"]}
    baseline = None
    primary, second, status = [], [], []
    for p in PAPER:
        directory = Path("runs") / f"p_{p:.1f}"
        c = config_for(inputs, directory, p, 42)
        if baseline is None:
            baseline = controlled_config(c)
        require(controlled_config(c) == baseline, "Seed-42 configs vary beyond norm_p")
        primary.append(completed_run(inputs, directory, p, 42,
                                     inputs.json(directory / "run-summary.json")))
        directory = Path("runs") / f"diagnostic_p_{p:.1f}_seed43"
        # Partial transfer snapshots cannot become final results. Read no partial FID series.
        try:
            c = config_for(inputs, directory, p, 43)
            require(controlled_config(c) == baseline, "Seed-43 config changes beyond seed")
            seed_check = inputs.json(directory / "seed-check-provenance.json")
            require(seed_check["git_sha"] == manifest["git_sha"]
                    and seed_check["source_patch_sha256"] == manifest["source_patch_sha256"]
                    and seed_check["primary_manifest_sha256"]
                    == inputs.hashes["provenance/reproducibility-manifest.json"]
                    and seed_check["training_seed"] == 43 and seed_check["sampling_seed"] == 0
                    and seed_check["from_scratch"] and not seed_check["resume_from_primary"],
                    "Seed-43 source/seed provenance mismatch")
            require(c == seed_check["expected_effective_config"],
                    "Seed-43 effective config disagrees with launch provenance")
            summary = inputs.json(directory / "run-summary.json")
            if summary.get("status") != "completed":
                status.append({"p": p, "directory": str(directory),
                               "status": "ongoing_or_incomplete",
                               "reason": f"Recorded status: {summary.get('status')!r}"})
                continue
            run = completed_run(inputs, directory, p, 43, summary)
            require(run["runtime_settings"] == primary[-1]["runtime_settings"],
                    "Seed-43 recorded runtime settings differ from seed 42")
            require(run["physical_gpu_uuid"] == primary[-1]["physical_gpu_uuid"]
                    == seed_check["same_physical_gpu_uuid"],
                    "Paired runs did not use the recorded same physical GPU")
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as error:
            status.append({"p": p, "directory": str(directory),
                           "status": "ongoing_or_unverified_snapshot",
                           "reason": str(error)})
        else:
            second.append(run)
            status.append({"p": p, "directory": str(directory), "status": "verified_complete"})
    output = args.output_root.resolve()
    require(not output.is_relative_to(inputs.root), "Outputs must be outside experiment records")
    for folder in ["analysis/generated", "figures", "tables"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    generated = output / "analysis/generated"
    write_csv(generated / "table1e_seed42.csv", primary,
              ["p", "paper_fid", "fid", "difference", "training_seed", "epoch"])
    write_table(output / "tables/loss_metrics_seed42.tex", primary)
    write_csv(generated / "individual_completed_runs.csv", primary + second,
              ["p", "training_seed", "epoch", "paper_fid", "fid", "difference",
               "run_hours", "fid_evaluation_seconds",
               "median_logged_steps_per_second_from_step500"])
    write_csv(generated / "run_costs.csv", primary + second,
              ["p", "training_seed", "directory", "run_hours", "fid_evaluation_seconds",
               "median_logged_steps_per_second_from_step500", "started_utc", "ended_utc"])
    write_csv(generated / "fid_epochs_seed42.csv",
              [{"p": x["p"], **row} for x in primary for row in x["curve"]],
              ["p", "epoch", "train_step", "FID_ema", "evaluation_seconds",
               "generation_seconds", "inception_and_stats_seconds", "fid_math_seconds"])
    write_csv(generated / "fid_epochs_all_verified.csv",
              [{"p": x["p"], "training_seed": x["training_seed"], **row}
               for x in primary + second for row in x["curve"]],
              ["p", "training_seed", "epoch", "train_step", "FID_ema", "num_samples",
               "sampling_seed", "NFE", "EMA", "evaluation_seconds", "generation_seconds",
               "inception_and_stats_seconds", "fid_math_seconds"])
    two_seed_paths = [generated / "table1e_two_seeds.csv", output / "tables/loss_metrics_two_seeds.tex"]
    two_ready = len(second) == 5
    costs = {"seed42": {**cost_summary(primary),
                         **parallel_timing(inputs, 42, primary, evidence)},
             "all_verified": cost_summary(primary + second)}
    if two_ready:
        costs["seed43"] = {**cost_summary(second), **parallel_timing(inputs, 43, second, evidence)}
        rows = []
        for a, b in zip(primary, second):
            values = np.array([a["fid"], b["fid"]])
            rows.append({"p": a["p"], "paper_fid": a["paper_fid"], "seed42": a["fid"],
                         "seed43": b["fid"], "n": 2, "mean": float(values.mean()),
                         "sample_sd": float(values.std(ddof=1)),
                         "seed42_difference": a["difference"],
                         "seed43_difference": b["difference"],
                         "paper_minus_mean": a["paper_fid"] - float(values.mean()),
                         "mean_difference": float(values.mean()) - a["paper_fid"]})
        write_csv(two_seed_paths[0], rows,
                  ["p", "paper_fid", "seed42", "seed43", "seed42_difference",
                   "seed43_difference", "n", "mean", "sample_sd", "mean_difference",
                   "paper_minus_mean"])
        write_table(two_seed_paths[1], rows, two_seeds=True)
    else:
        # Remove only stale generated aggregates if rerun against an incomplete snapshot.
        for path in two_seed_paths:
            path.unlink(missing_ok=True)
    plots(primary, second if two_ready else [], output / "figures")
    report = {"data_copy": str(inputs.root), "endpoint": "epoch 80, not best epoch/seed",
              "aggregation": "Arithmetic mean of two independently measured FIDs; no pooling of statistics.",
              "sd_ddof": 1, "n_training_seeds_per_p": 2 if two_ready else None,
              "table_delta_definition": "Paper FID minus our arithmetic two-seed mean FID.",
              "seed43_selection": "Second sweep added after inspecting the seed-42 results.",
              "training_seeds": [42, 43], "sampling_seed": 0,
              "two_seed_table_ready": two_ready, "seed43_status": status,
              "seed42_gpu_hours_sum": sum(x["run_hours"] for x in primary),
              "seed42_run_hours_range": [min(x["run_hours"] for x in primary),
                                         max(x["run_hours"] for x in primary)],
              "seed42": primary, "verified_seed43": second,
              "measured_costs": costs,
              "effective_controls": baseline, "upstream_commit": manifest["git_sha"],
              "provenance": evidence, "input_sha256": inputs.hashes,
              "analysis_versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "figure_script_sha256": hashlib.sha256(
                  Path(__file__).with_name("paper_seed_figure.py").read_bytes()).hexdigest(),
              "limitations": ["Two training seeds cannot establish statistical robustness.",
                              "Recorded run GPU-hours exclude preprocessing and idle allocation time."]}
    (generated / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified five seed-42 endpoints; {len(second)}/5 seed-43 endpoints verified.")
    print(f"Seed-42 run GPU-hours: {report['seed42_gpu_hours_sum']:.4f}")
    print(f"All verified run GPU-hours: {costs['all_verified']['summed_run_gpu_hours']:.4f}")
    for item in status:
        if item["status"] != "verified_complete":
            print(f"PENDING {item['directory']}: {item['reason']}")
    print(f"Two-seed table ready: {two_ready}; outputs: {output}")


if __name__ == "__main__":
    main()
