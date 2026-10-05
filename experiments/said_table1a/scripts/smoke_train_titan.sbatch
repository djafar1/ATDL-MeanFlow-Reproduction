#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mf-smoke-titan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=00:15:00
#SBATCH --output=experiments/said_table1a/logs/smoke-titan-%j.out
#SBATCH --error=experiments/said_table1a/logs/smoke-titan-%j.err

set -euo pipefail

export HF_HOME=/home/hdn456/projects/ATDL-MeanFlow-data/hf_cache

cd /home/hdn456/projects/ATDL-MeanFlow-Reproduction/src/meanflow

echo "===== GPU ====="
nvidia-smi

echo "===== TRAINING SMOKE TEST ====="

/home/hdn456/.conda/envs/meanflow-jax-run/bin/python main.py \
  --workdir=/home/hdn456/projects/ATDL-MeanFlow-data/runs/smoke_train_titan_${SLURM_JOB_ID} \
  --config=configs/load_config.py:smoke_train_25
