# Environment setup

## Hendrix training and evaluation

From the project root on Hendrix, use Python 3.11 to create the separate GPU
environment:

```sh
bash scripts/hendrix/setup.sh python3.11
source .venv-hendrix/bin/activate
```

The setup script installs `requirements-cuda12.txt`, which includes the shared
pins in `requirements-cluster.txt`. JAX runs on the GPU; Torch and TensorFlow
use CPU packages. The script records the installed versions inside the environment
as `installed-requirements.txt`. If Python 3.11 is supplied through a cluster
module, load that module first or pass its Python executable to the script.

Training and evaluation run through Slurm, using the commands in the
[Table 1(b) guide](../experiments/table_b/README.md). The environment and latent
dataset are shared across jobs; each training submission freezes its own code
and configuration snapshot.

For optional CPU integration validation, create a separate environment:

```sh
MEANFLOW_BACKEND=cpu bash scripts/hendrix/setup.sh python3.11 .venv-validation
MEANFLOW_CPU_INTEGRATION=1 .venv-validation/bin/python -m unittest discover -s tests -v
```

## Local editing

The repository's `pyrightconfig.json` points Pyright (including Neovim's language
server) at `.venv` and adds `src/meanflow` to its import search path. Start Neovim
anywhere inside the repository; activating the environment is not required for
Pyright. Restart Neovim after first setting up the environment.

Recreate the environment from the repository root with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements-editor.txt
```

This environment supplies package source and type information for local editing.
It uses CPU PyTorch wheels and the upstream JAX version, without TPU/CUDA support.
It is not the cluster training environment. In particular, binary wheels may
need additional runtime libraries on NixOS, and TensorFlow is not installed.
Pyright reads the installed files without importing these binary libraries.

Keep the cluster's environment separate. The original TPU setup
remains in `src/meanflow/scripts/install.sh`.

Diagnostics are not disabled by this setup. The original code still has typing
issues (for example, layer factories annotated as module instances), and the
optional TensorFlow import in `utils/fid_util.py` remains unresolved locally.
