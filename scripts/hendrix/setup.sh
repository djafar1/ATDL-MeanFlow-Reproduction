#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash scripts/hendrix/setup.sh [python3.11-or-module-python] [venv-directory]'
  echo 'Default: Python 3.11, .venv-hendrix, CUDA 12. Set MEANFLOW_BACKEND=cpu for validation.'
  exit 0
fi
if (( $# > 2 )); then
  echo 'Too many arguments. Use --help.' >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_cmd=${1:-python3.11}
venv_dir=${2:-"$repo_root/.venv-hendrix"}
backend=${MEANFLOW_BACKEND:-cuda12}
case "$backend" in
  cpu|cuda12) ;;
  *) echo 'MEANFLOW_BACKEND must be cpu or cuda12.' >&2; exit 2 ;;
esac
"$python_cmd" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
  "$python_cmd" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing venv must use Python 3.11"'
"$venv_dir/bin/python" -m pip install 'pip==24.3.1'
# CPU wheels avoid a second CUDA stack competing with JAX for GPU memory.
"$venv_dir/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu'
requirements="$repo_root/environment/requirements-cluster.txt"
if [[ "$backend" == cuda12 ]]; then
  requirements="$repo_root/environment/requirements-cuda12.txt"
fi
"$venv_dir/bin/python" -m pip install -r "$requirements"
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" -m pip freeze > "$venv_dir/installed-requirements.txt"
echo "Environment ready: $venv_dir"
echo 'Next: activate it, then check imports and GPU access inside a Slurm allocation.'
