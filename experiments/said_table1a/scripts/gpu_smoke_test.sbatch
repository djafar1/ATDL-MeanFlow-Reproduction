#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mf-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=00:10:00
#SBATCH --output=experiments/said_table1a/logs/smoke-%j.out
#SBATCH --error=experiments/said_table1a/logs/smoke-%j.err

set -e

echo "===== NODE ====="
hostname

echo "===== GPU ====="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "===== PYTHON ====="
/home/hdn456/.conda/envs/meanflow-jax/bin/python --version

echo "===== JAX TEST ====="
cd /home/hdn456/projects/ATDL-MeanFlow-Reproduction/src/meanflow

/home/hdn456/.conda/envs/meanflow-jax/bin/python - <<'PY'
import jax
import jax.numpy as jnp

print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

@jax.jit
def matmul(x):
    return x @ x

x = jnp.ones((2048, 2048), dtype=jnp.float32)
y = matmul(x)
y.block_until_ready()

print("JAX GPU COMPUTE OK")
PY

echo "===== SMOKE TEST PASSED ====="
