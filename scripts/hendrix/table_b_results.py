#!/usr/bin/env python3
"""Select a common saved step or collect four matched FID results into CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys

from table_b_common import SETTINGS, common_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    selection = subparsers.add_parser("common-step")
    selection.add_argument("run_root")
    collection = subparsers.add_parser("collect")
    collection.add_argument("run_root")
    collection.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if args.command == "common-step":
        print(common_step(args.run_root))
        return
    rows = []
    convention = None
    dataset = None
    for name, r, t, paper_fid in SETTINGS:
        run = Path(args.run_root) / name
        result = json.loads((run / f"eval-{args.step}" / "fid.json").read_text())
        config = json.loads((run / "config.json").read_text())
        current_dataset = json.loads((run / "dataset.json").read_text())
        if (result["step"] != args.step or result["jvp_tangent_r"] != r
                or result["jvp_tangent_t"] != t
                or config["method"]["jvp_tangent_r"] != r
                or config["method"]["jvp_tangent_t"] != t):
            raise ValueError(f"Wrong checkpoint/setting in {name}")
        keys = ("num_samples", "num_steps", "sampling_seed", "sample_batch_size", "fid_reference_sha256")
        current = {key: result[key] for key in keys}
        if current["num_samples"] != 50000 or current["num_steps"] != 1:
            raise ValueError("Table 1(b) requires one-step FID-50K")
        if convention is not None and current != convention:
            raise ValueError("Evaluation settings differ across runs")
        if dataset is not None and current_dataset != dataset:
            raise ValueError("Datasets differ across runs")
        if (config["method"]["guidance_eq"] != "none" or config["method"]["kappa"] != 0
                or config["method"]["omega"] != 1 or config["method"]["norm_p"] != 1):
            raise ValueError("Table 1(b) must disable guidance")
        convention, dataset = current, current_dataset
        rows.append({"jvp_tangent": f"(v,{int(r)},{int(t)})", "paper_fid": paper_fid,
                     "reproduced_fid": result["fid_ema"],
                     "step": args.step, "epochs": args.step / current_dataset["steps_per_epoch"]})
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
