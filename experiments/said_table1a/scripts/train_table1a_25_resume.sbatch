#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mf-t1a-25-res
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/said_table1a/logs/train-25-res-%j.out
#SBATCH --error=experiments/said_table1a/logs/train-25-res-%j.err

set -e

export HF_HOME=/home/hdn456/projects/ATDL-MeanFlow-data/hf_cache

cd /home/hdn456/projects/ATDL-MeanFlow-Reproduction/src/meanflow

nvidia-smi || true

/home/hdn456/.conda/envs/meanflow-jax-run/bin/python main.py \
  --workdir=/home/hdn456/projects/ATDL-MeanFlow-data/runs/table1a_25 \
  --config=configs/load_config.py:table1a_25_resume
