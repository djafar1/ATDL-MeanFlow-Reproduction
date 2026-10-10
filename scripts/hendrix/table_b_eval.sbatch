#!/usr/bin/env bash
#SBATCH --job-name=meanflow-table-b-fid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=table-b-fid-%A_%a.out

set -euo pipefail
run_root=${1:?Experiment run root required}
step=${2:?Common checkpoint step required}
reference=${3:?Authors FID reference .npz required}
venv_dir=${4:?Shared environment path required}
cd -- "$run_root/source"
if [[ -n ${MEANFLOW_PYTHON_MODULE:-} ]]; then
  module load "$MEANFLOW_PYTHON_MODULE"
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
settings=(v01 v00 v10 v11)
setting=${settings[${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}]}
srun "$venv_dir/bin/python" -u scripts/hendrix/table_b.py evaluate \
  --run "$run_root/$setting" --step "$step" --fid-reference "$reference" \
  --workdir "$run_root/$setting/eval-$step"
