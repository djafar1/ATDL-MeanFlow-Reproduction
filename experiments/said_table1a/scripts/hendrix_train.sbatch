#!/bin/bash
#SBATCH --job-name=mf-table1a
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=mf-table1a-%j.out
#SBATCH --error=mf-table1a-%j.err

set -euo pipefail

RATIO=${RATIO:?Use RATIO=0,25,50,100}

REPO=/home/hdn456/projects/ATDL-MeanFlow-Reproduction
RUNS=/home/hdn456/projects/ATDL-MeanFlow-data/runs
PYTHON=/home/hdn456/.conda/envs/meanflow-jax-run/bin/python

cd "$REPO/src/meanflow"

"$PYTHON" main.py \
  --workdir="$RUNS/table1a_$RATIO" \
  --config="configs/load_config.py:table1a_$RATIO"
