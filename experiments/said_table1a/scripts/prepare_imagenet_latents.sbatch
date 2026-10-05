#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mf-latents
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=24:00:00
#SBATCH --output=experiments/said_table1a/logs/latents-%j.out
#SBATCH --error=experiments/said_table1a/logs/latents-%j.err

set -e

export HF_HOME=/home/hdn456/projects/ATDL-MeanFlow-data/hf_cache
export IMAGENET_ROOT=/home/hdn456/projects/ATDL-MeanFlow-data/imagenet/CLS-LOC
export LATENT_ROOT=/home/hdn456/projects/ATDL-MeanFlow-data/latents_imagenet256

cd /home/hdn456/projects/ATDL-MeanFlow-Reproduction/src/meanflow

/home/hdn456/.conda/envs/meanflow-jax/bin/python prepare_dataset.py \
    --imagenet_root="$IMAGENET_ROOT" \
    --output_dir="$LATENT_ROOT" \
    --batch_size=128 \
    --vae_type=mse \
    --image_size=256 \
    --compute_latent=True \
    --compute_fid=False \
    --overwrite=False
