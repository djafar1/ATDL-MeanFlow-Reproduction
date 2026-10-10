#!/bin/bash
set -euo pipefail

RATIO=${1:?Usage: ucloud_train.sh <0|25|50|100>}

REPO=/work/MeanFlow/ATDL-MeanFlow-Reproduction
PYTHON=/work/MeanFlow/meanflow-env/bin/python
RUNS=/work/MeanFlow/runs
DATA=/tmp/latents_imagenet256
TAR=/work/MeanFlow/data/latents_imagenet256.tar
CONFIG="$REPO/src/meanflow/configs/_table1a_ucloud.yml"

if [ ! -d "$DATA" ]; then
  tar -xf "$TAR" -C /tmp
fi

cp "$REPO/experiments/said_table1a/configs/table1a_$RATIO.yml" "$CONFIG"

sed -i \
  -e 's|/home/hdn456/projects/ATDL-MeanFlow-data/latents_imagenet256|/tmp/latents_imagenet256|' \
  -e 's|/home/hdn456/projects/ATDL-MeanFlow-data/fid/imagenet256_fid_stats.npz|/work/MeanFlow/data/fid/imagenet256_fid_stats.npz|' \
  "$CONFIG"

trap 'rm -f "$CONFIG"' EXIT

cd "$REPO/src/meanflow"

"$PYTHON" main.py \
  --workdir="$RUNS/table1a_${RATIO}_b200" \
  --config=configs/load_config.py:_table1a_ucloud
