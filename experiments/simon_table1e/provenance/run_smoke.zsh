#!/usr/bin/env zsh
MF_ROOT=/work/meanflow-table1e
cd "$MF_ROOT" || exit 1

export MEANFLOW_SINGLE_PROCESS=1
export MEANFLOW_VAE_PATH="$MF_ROOT/models/sd-vae-ft-mse-flax"
export MEANFLOW_LATENT_MANIFEST="$MF_ROOT/provenance/latent-file-order.txt"
export HF_HOME="$MF_ROOT/cache/huggingface"
mkdir -p "$MF_ROOT/cache/tmp/jax_fid" || exit 1
MF_RUNTIME_TMP=$(mktemp -d /tmp/meanflow.XXXXXX) || exit 1
ln -s "$MF_ROOT/cache/tmp/jax_fid" "$MF_RUNTIME_TMP/jax_fid" || exit 1
export TMPDIR="$MF_RUNTIME_TMP"

"$MF_ROOT/venv/bin/python" - <<'PYIPC' || exit 1
from multiprocessing.connection import Listener
from pathlib import Path
import tempfile
with Listener(family="AF_UNIX") as listener:
    print("TEMPORARY SOCKET CHECK PASSED:", listener.address, flush=True)
weights = Path(tempfile.gettempdir()) / "jax_fid/inception_v3_weights_fid.pickle"
assert weights.is_file(), f"Missing cached weights: {weights}"
print("Reusing FID weights:", weights.resolve(), flush=True)
PYIPC
export JAX_PLATFORMS=cuda,cpu
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$TMPDIR" "$MF_ROOT/runs/smoke_p_1.0" || exit 1
MF_SMOKE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export MEANFLOW_SMOKE_DIR="$MF_ROOT/runs/smoke_p_1.0/$MF_SMOKE_STAMP"
mkdir "$MEANFLOW_SMOKE_DIR" || exit 1
printf '%s\n' "$MEANFLOW_SMOKE_DIR" > provenance/latest-smoke-dir.txt
date -Is > "$MEANFLOW_SMOKE_DIR/start.txt"

"$MF_ROOT/venv/bin/python" -u "$MF_ROOT/provenance/p1_smoke.py" \
  2>&1 | tee "$MEANFLOW_SMOKE_DIR/training.log"
MF_SMOKE_STATUS=${pipestatus[1]}

date -Is > "$MEANFLOW_SMOKE_DIR/end.txt"
printf '%s\n' "$MF_SMOKE_STATUS" > "$MEANFLOW_SMOKE_DIR/exit-status.txt"
printf 'Smoke exit status: %s\n' "$MF_SMOKE_STATUS"
exit "$MF_SMOKE_STATUS"
