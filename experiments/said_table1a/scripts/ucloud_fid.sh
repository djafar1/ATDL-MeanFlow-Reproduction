#!/bin/bash
set -euo pipefail

RATIO=${1:?Usage: ucloud_fid.sh <ratio> <checkpoint>}
CHECKPOINT=${2:?Usage: ucloud_fid.sh <ratio> <checkpoint>}

REPO=/work/MeanFlow/ATDL-MeanFlow-Reproduction
PYTHON=/work/MeanFlow/meanflow-env/bin/python
CONFIG="$REPO/src/meanflow/configs/_table1a_fid.yml"

"$PYTHON" - "$REPO/experiments/said_table1a/configs/table1a_$RATIO.yml" "$CONFIG" "$CHECKPOINT" <<'PY'
import sys, yaml

src, dst, checkpoint = sys.argv[1:]

with open(src) as f:
    c = yaml.safe_load(f)

c["load_from"] = checkpoint
c["eval_only"] = True
c["dataset"]["root"] = "/tmp/latents_imagenet256"
c["method"]["guidance_eq"] = "none"
c["sampling"]["num_steps"] = 1
c["fid"]["on_training"] = True
c["fid"]["num_samples"] = 50000
c["fid"]["device_batch_size"] = 20
c["fid"]["cache_ref"] = "/work/MeanFlow/data/fid/imagenet256_fid_stats.npz"

with open(dst, "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY

trap 'rm -f "$CONFIG"' EXIT

cd "$REPO/src/meanflow"

"$PYTHON" main.py \
  --workdir="/work/MeanFlow/eval/table1a_$RATIO" \
  --config=configs/load_config.py:_table1a_fid
