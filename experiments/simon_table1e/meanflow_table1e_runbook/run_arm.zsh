#!/usr/bin/env zsh
# Script-local failure handling; no options are applied to your interactive shell.
emulate -L zsh
MF_ROOT=/work/meanflow-table1e
MF_BUNDLE="$MF_ROOT/meanflow_table1e_runbook"
MF_GPU="$1"
MF_P="$2"
export MF_RUN_DIR="$MF_ROOT/runs/p_$MF_P"
cd "$MF_ROOT/meanflow" || exit 1
mkdir "$MF_RUN_DIR/launch.lock" || exit 1

MF_GPU_UUID=$(nvidia-smi --id="$MF_GPU" --query-gpu=uuid --format=csv,noheader) || exit 1
export CUDA_VISIBLE_DEVICES="$MF_GPU_UUID"
export MEANFLOW_SINGLE_PROCESS=1
export MEANFLOW_VAE_PATH="$MF_ROOT/models/sd-vae-ft-mse-flax"
export MEANFLOW_LATENT_MANIFEST="$MF_ROOT/provenance/latent-file-order.txt"
export HF_HOME="$MF_ROOT/cache/huggingface"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MF_RUNTIME_TMP=$(mktemp -d /tmp/meanflow.XXXXXX) || exit 1
ln -s "$MF_ROOT/cache/tmp/jax_fid" "$MF_RUNTIME_TMP/jax_fid" || exit 1
export TMPDIR="$MF_RUNTIME_TMP"
cp "configs/table1e_p_$MF_P.yml" "$MF_RUN_DIR/config.yml" || exit 1
date -Is > "$MF_RUN_DIR/start.txt"
MF_CMD=("$MF_ROOT/venv/bin/python" -u "$MF_BUNDLE/audit_run.py" "--config=configs/load_config.py:table1e_p_$MF_P" "--workdir=$MF_RUN_DIR")
"$MF_ROOT/venv/bin/python" - "$MF_RUN_DIR" "$MF_GPU" "${MF_CMD[@]}" <<'PYMETA' || exit 1
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys
keys = ['CUDA_VISIBLE_DEVICES', 'MEANFLOW_SINGLE_PROCESS', 'MEANFLOW_VAE_PATH',
        'MEANFLOW_LATENT_MANIFEST', 'HF_HOME', 'TMPDIR', 'JAX_PLATFORMS',
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MF_RUN_DIR']
work = Path(sys.argv[1])
work.joinpath('command.json').write_text(json.dumps({
    'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'physical_gpu_index': int(sys.argv[2]), 'working_directory': os.getcwd(),
    'argv': sys.argv[3:], 'environment': {k: os.environ[k] for k in keys},
    'config_sha256': hashlib.sha256(work.joinpath('config.yml').read_bytes()).hexdigest(),
    'entrypoint': 'Recording wrapper calls the existing authors main.py via runpy; no added training steps.'
}, indent=2) + '\n')
PYMETA

nvidia-smi --id="$MF_GPU_UUID" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv --loop=10 > "$MF_RUN_DIR/gpu.csv" 2>&1 &
MF_MONITOR_PID=$!
"${MF_CMD[@]}" > "$MF_RUN_DIR/training.log" 2>&1 &
MF_PYTHON_PID=$!
printf '%s\n' "$MF_PYTHON_PID" > "$MF_RUN_DIR/python-pid.txt"
wait "$MF_PYTHON_PID"
MF_TRAIN_STATUS=$?

kill "$MF_MONITOR_PID" 2>/dev/null
wait "$MF_MONITOR_PID" 2>/dev/null
date -Is > "$MF_RUN_DIR/end.txt"
printf '%s\n' "$MF_TRAIN_STATUS" > "$MF_RUN_DIR/exit-status.txt"
printf 'p=%s, GPU=%s, exit status=%s\n' "$MF_P" "$MF_GPU" "$MF_TRAIN_STATUS"
exit "$MF_TRAIN_STATUS"
