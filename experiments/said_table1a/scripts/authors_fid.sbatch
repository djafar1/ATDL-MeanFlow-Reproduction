#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mf-author-fid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=06:00:00
#SBATCH --output=experiments/said_table1a/logs/author-fid-%j.out
#SBATCH --error=experiments/said_table1a/logs/author-fid-%j.err

set -e

export HF_HOME=/home/hdn456/projects/ATDL-MeanFlow-data/hf_cache

cd /home/hdn456/projects/ATDL-MeanFlow-Reproduction/src/meanflow

nvidia-smi

/home/hdn456/.conda/envs/meanflow-jax-run/bin/python main.py \
  --workdir=/home/hdn456/projects/ATDL-MeanFlow-data/runs/authors_b4_eval \
  --config=configs/load_config.py:authors_b4_eval
