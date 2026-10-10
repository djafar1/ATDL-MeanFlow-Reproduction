#!/usr/bin/env bash
#SBATCH --job-name=imagenet-latents
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=imagenet-latents-%j.out

set -euo pipefail
repo_root=${SLURM_SUBMIT_DIR:?Submit from the repository root}
imagenet_root=${1:?Usage: sbatch scripts/hendrix/imagenet_latents.sbatch RAW_ROOT LATENT_ROOT}
latent_root=${2:?Supply a fresh absolute latent output path}
[[ "$imagenet_root" == /* && "$latent_root" == /* ]] || { echo 'Use absolute data paths.' >&2; exit 2; }
cd -- "$repo_root"
if [[ -n ${MEANFLOW_PYTHON_MODULE:-} ]]; then
  module load "$MEANFLOW_PYTHON_MODULE"
fi
venv_dir=${MEANFLOW_VENV:-"$repo_root/.venv-hendrix"}
[[ -x "$venv_dir/bin/python" ]] || { echo 'The Hendrix environment is missing.' >&2; exit 1; }
mkdir -p -- "$latent_root"
exec 9>"$latent_root/.imagenet-latents.lock"
flock -n 9 || { echo 'Another encoder is using this latent output path.' >&2; exit 1; }
"$venv_dir/bin/python" scripts/hendrix/check_imagenet.py raw "$imagenet_root"
if [[ -d "$latent_root/train" ]] && [[ -n $(find "$latent_root/train" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  # Upstream silently skips nonempty outputs. Accept only a verified complete cache.
  if "$venv_dir/bin/python" scripts/hendrix/check_imagenet.py latents "$latent_root"; then
    echo 'The full latent cache already exists; reusing it.'
    exit 0
  fi
  echo 'A partial cache exists. Choose a fresh LATENT_ROOT; this encoder cannot resume partial caches.' >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=${HF_HOME:-"$repo_root/data/huggingface"}
hostname
nvidia-smi
srun "$venv_dir/bin/python" -u src/meanflow/prepare_dataset.py \
  --imagenet_root="$imagenet_root" --output_dir="$latent_root" \
  --batch_size=32 --vae_type=mse --image_size=256 \
  --compute_latent=True --compute_fid=False --overwrite=False
"$venv_dir/bin/python" scripts/hendrix/check_imagenet.py latents "$latent_root"
echo "Training input is ready: $latent_root"
