#!/usr/bin/env python3
"""Preview seed layouts with explicit synthetic labels, or verified recorded runs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
PAPER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
MARKERS = ["o", "s", "^", "D", "v"]


def curves(summary, mode):
    first = summary["seed42"]
    if len(first) != 5:
        raise ValueError("Expected five verified seed-42 runs")
    if mode == "recorded":
        second = summary["verified_seed43"]
        if not summary["two_seed_table_ready"] or len(second) != 5:
            raise ValueError("Recorded mode requires all five verified seed-43 endpoints. "
                             "Refresh the experiment copy and rerun analyse_loss_metrics.py first.")
    else:
        # Arbitrary, deterministic offsets for DESIGN ONLY. No seed-43 records are read.
        # They neither estimate training-seed variance nor predict its final FID.
        offsets = [-3.2, 1.9, -2.4, 4.7, -5.6]
        second = []
        for i, run in enumerate(first):
            original = run["curve"]
            n = len(original)
            phase = np.linspace(0, 1, n)
            perturbation = (offsets[i] * (0.5 + 0.5 * phase)
                            + 2.4 * np.sin(phase * 3 * np.pi + i) * (1 - phase))
            fake = [{"epoch": row["epoch"], "FID_ema": float(row["FID_ema"]) + offset}
                    for row, offset in zip(original, perturbation)]
            second.append({"p": run["p"], "training_seed": 43,
                           "synthetic": True, "curve": fake})
    return first, second


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def paper_comparison(first, second, folder, mode):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True, sharey=True,
                             layout="constrained")
    for seed, runs, ax in zip([42, 43], [first, second], axes):
        for run, color, marker in zip(runs, PAPER_COLORS, MARKERS):
            rows = run["curve"]
            ax.plot([int(x["epoch"]) for x in rows], [float(x["FID_ema"]) for x in rows],
                    color=color, marker=marker, markersize=3.5, linewidth=1.3,
                    linestyle="--" if seed == 43 else "-", label=f"$p={run['p']:g}$")
        synthetic = mode == "synthetic" and seed == 43
        ax.set_title(f"Seed {seed} — {'SYNTHETIC DEMO' if synthetic else 'recorded'}",
                     fontsize=10, loc="left", color="#b91c1c" if synthetic else "#111827")
        ax.set(ylim=(55, 160), yticks=[60, 100, 140], ylabel="EMA FID-50K")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, frameon=False, fontsize=10, handlelength=1.2,
                   columnspacing=0.6, labelspacing=0.1, loc="upper right")
    axes[1].set(xlabel="Training epoch", xticks=[10, 30, 50, 80])
    if mode == "synthetic":
        fig.suptitle("DESIGN PREVIEW · NOT RESULTS", fontsize=10, color="#b91c1c")
    save(fig, folder / "combined_2d_seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--summary", type=Path, default=REPORT / "analysis/generated/analysis_summary.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    first, second = curves(summary, args.mode)
    folder = REPORT / "analysis/visualisation-previews" / (
        "SYNTHETIC_SEED43" if args.mode == "synthetic" else "RECORDED_SEEDS")
    folder.mkdir(parents=True, exist_ok=True)
    paper_comparison(first, second, folder, args.mode)
    payload = {"mode": args.mode, "synthetic_seed43": args.mode == "synthetic",
               "warning": "SYNTHETIC SEED 43 IS FOR DESIGN ONLY" if args.mode == "synthetic" else None,
               "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
               "seed42": [{"p": run["p"], "curve": run["curve"]} for run in first],
               "seed43": [{"p": run["p"], "curve": run["curve"]} for run in second]}
    (folder / "preview_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{args.mode.upper()} visualisations written to {folder}")


if __name__ == "__main__":
    main()
