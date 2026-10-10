"""Shared setup for the project runners; upstream modules stay unchanged."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "src" / "meanflow"


def initialize(*, allow_cpu=False):
    sys.path.insert(0, str(UPSTREAM))
    # Set before importing TensorFlow/JAX, without changing Slurm's GPU visibility.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    import jax

    if "SLURM_JOB_ID" in os.environ:
        required = ("SLURM_STEP_NODELIST", "SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID")
        if any(key not in os.environ for key in required):
            raise RuntimeError("Launch Python with srun so JAX can discover the Slurm task.")
        if int(os.environ["SLURM_NTASKS"]) != 1:
            raise RuntimeError("These introductory runners support one Slurm task only.")
        jax.distributed.initialize()
    devices = jax.local_devices()
    print(f"JAX {jax.__version__}; processes={jax.process_count()}; devices={devices}", flush=True)
    if jax.process_count() != 1:
        raise RuntimeError("These runners require a single JAX process.")
    if not allow_cpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("No JAX GPU detected. Request a GPU with Slurm before running.")
    return jax


def load_config(path):
    import yaml
    from configs.default import enforce_relations, get_config

    config = get_config()
    with Path(path).open() as stream:
        overrides = yaml.safe_load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown configuration section: {key}")
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    enforce_relations(config)
    return config


def new_workdir(path):
    directory = Path(path).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Workdir must be empty; use a new directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory
