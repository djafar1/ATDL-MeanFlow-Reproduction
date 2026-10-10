#!/usr/bin/env zsh
MF_ROOT=/work/meanflow-table1e
cd "$MF_ROOT/meanflow" || exit 1

export MEANFLOW_SINGLE_PROCESS=1
export MEANFLOW_VAE_PATH="$MF_ROOT/models/sd-vae-ft-mse-flax"
export HF_HOME="$MF_ROOT/cache/huggingface"
export JAX_PLATFORMS=cuda,cpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

date -Is > "$MF_ROOT/provenance/latent-preprocessing-start.txt"

"$MF_ROOT/venv/bin/python" -u prepare_dataset.py \
  --imagenet_root="$MF_ROOT/data/raw/ILSVRC/Data/CLS-LOC" \
  --output_dir="$MF_ROOT/data/imagenet256_latents" \
  --vae_type=mse \
  --image_size=256 \
  --batch_size=32 \
  --compute_latent=True \
  --compute_fid=False \
  --overwrite=False

MF_PREPROCESS_STATUS=$?
date -Is > "$MF_ROOT/provenance/latent-preprocessing-end.txt"
printf '%s\n' "$MF_PREPROCESS_STATUS" \
  > "$MF_ROOT/provenance/latent-preprocessing-exit-status.txt"
exit "$MF_PREPROCESS_STATUS"
