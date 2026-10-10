#!/usr/bin/env python3
"""Preview the four-job array; only --submit creates a snapshot and submits jobs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

from table_b_common import SETTINGS, checkpoint_steps, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", help="Full prepared ImageNet latent root, not its train folder")
    parser.add_argument("--run-root", help="New output root; defaults to a unique results/runs path")
    parser.add_argument("--resume", help="Existing run root; reuses its source snapshot and dataset")
    parser.add_argument("--venv", default=str(ROOT / ".venv-hendrix"))
    parser.add_argument("--gpu-type", required=True, help="Choose an exact GPU resource name from sinfo")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--max-steps", type=int, help="Optional common optimizer-step target; default: 80 epochs")
    parser.add_argument("--submit", action="store_true", help="Actually create snapshot and call sbatch")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.gpu_type):
        parser.error("gpu-type must be a single Slurm model resource name")
    if args.resume:
        if args.run_root or args.dataset_root:
            parser.error("--resume reuses its saved run-root and dataset-root")
        run_root = Path(args.resume).expanduser().resolve()
        manifest = json.loads((run_root / "experiment.json").read_text())
        if manifest.get("table") != "1(b)":
            parser.error("Only Table 1(b) experiments can be resumed by this launcher")
        dataset = Path(manifest["dataset_root"])
        source = run_root / "source"
        venv = Path(manifest["venv"])
        indices = []
        for index, (name, _, _, _) in enumerate(SETTINGS):
            if not checkpoint_steps(run_root / name / "checkpoints"):
                parser.error(f"No resumable checkpoint for {name}; resume it individually")
            config = json.loads((run_root / name / "config.json").read_text())
            data = json.loads((run_root / name / "dataset.json").read_text())
            target = config["training"]["num_epochs"] * data["steps_per_epoch"]
            if args.max_steps is not None:
                target = min(target, args.max_steps)
            if max(checkpoint_steps(run_root / name / "checkpoints")) < target:
                indices.append(str(index))
        if not indices:
            parser.error("All four runs have already reached the requested target")
        array = ",".join(indices) + f"%{args.concurrency}"
    else:
        if not args.dataset_root:
            parser.error("--dataset-root is required for a new experiment")
        dataset = Path(args.dataset_root).expanduser().resolve()
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_root = (Path(args.run_root).expanduser().resolve() if args.run_root
                    else ROOT / "results" / "runs" / ("table-b-" + token))
        source = run_root / "source"
        venv = Path(args.venv).expanduser().resolve()
        if run_root.exists():
            parser.error("New run-root must not already exist")
        array = f"0-3%{args.concurrency}"
    if not (dataset / "train").is_dir():
        parser.error(f"No prepared latent train directory at {dataset / 'train'}")
    command = ["sbatch", "--parsable", f"--array={array}",
               f"--gres=gpu:{args.gpu_type}:1", "--time=48:00:00",
               f"--output={run_root}/slurm-%A_%a.out", f"--error={run_root}/slurm-%A_%a.err",
               str(source / "scripts" / "hendrix" / "table_b.sbatch"),
               str(source), str(run_root), str(dataset), str(venv),
               "true" if args.resume else "false", str(args.max_steps or 0)]
    print(shlex.join(command), flush=True)
    if not args.submit:
        print("PREVIEW ONLY: no files created and no jobs submitted. Add --submit on Hendrix when ready.")
        return
    if not (venv / "bin" / "python").is_file():
        parser.error(f"No cluster Python at {venv / 'bin' / 'python'}")
    resources = subprocess.check_output(["sinfo", "-h", "-p", "gpu", "-o", "%G"], text=True)
    if not re.search(r"(?:^|[\s,])gpu:" + re.escape(args.gpu_type) + r":", resources):
        parser.error(f"GPU resource {args.gpu_type!r} not found in sinfo; specify --gpu-type explicitly")
    # Check the full cache on the gateway before allocating four GPUs.
    checker = (source if args.resume else ROOT) / "scripts/hendrix/check_imagenet.py"
    subprocess.run([str(venv / "bin/python"), str(checker), "latents", str(dataset)], check=True)
    if not args.resume:
        run_root.mkdir(parents=True, exist_ok=False)
        source.mkdir()
        for directory in ("src", "scripts", "configs", "environment"):
            shutil.copytree(ROOT / directory, source / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        write_json(run_root / "experiment.json", {
            "table": "1(b)", "dataset_root": str(dataset), "venv": str(venv),
            "git_revision": revision, "source_snapshot": str(source),
            "settings": [{"name": name, "jvp_tangent_r": r, "jvp_tangent_t": t, "paper_fid": fid}
                         for name, r, t, fid in SETTINGS],
            "note": "Snapshot contains current files, including uncommitted changes."})
    response = subprocess.check_output(command, cwd=source, text=True).strip()
    job_id = response.split(";")[0]
    write_json(run_root / f"submission-{job_id}.json", {
        "job_id": job_id, "command": command, "resume": bool(args.resume),
        "gpu_type": args.gpu_type, "concurrency": args.concurrency, "max_steps": args.max_steps})
    print(f"Submitted array {job_id}. Run root: {run_root}", flush=True)


if __name__ == "__main__":
    main()
