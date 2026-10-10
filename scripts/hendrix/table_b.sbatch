#!/usr/bin/env bash
#SBATCH --job-name=meanflow-table-b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=table-b-%A_%a.out

set -euo pipefail
source_root=${1:?Source snapshot path required}
run_root=${2:?Run root required}
dataset_root=${3:?Latent dataset root required}
venv_dir=${4:?Shared environment path required}
resume=${5:-false}
max_steps=${6:-0}
cd -- "$source_root"
if [[ -n ${MEANFLOW_PYTHON_MODULE:-} ]]; then
  module load "$MEANFLOW_PYTHON_MODULE"
fi
[[ -x "$venv_dir/bin/python" ]] || { echo 'Cluster environment missing' >&2; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
settings=(v01 v00 v10 v11)
setting=${settings[${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}]}
args=(train --config "configs/table_b/$setting.yml"
      --dataset-root "$dataset_root" --workdir "$run_root/$setting")
if [[ "$resume" == true ]]; then args+=(--resume); fi
if [[ "$max_steps" != 0 ]]; then args+=(--max-steps "$max_steps"); fi
hostname
nvidia-smi
srun "$venv_dir/bin/python" -u scripts/hendrix/table_b.py "${args[@]}"
