"""Run names and checkpoint selection, without importing the ML stack."""

import json
from pathlib import Path
import re

# Paper order: (z, r, t); the upstream JVP uses (z, t, r).
SETTINGS = (('v01', 0.0, 1.0, 61.06), ('v00', 0.0, 0.0, 268.06), ('v10', 1.0, 0.0, 329.22), ('v11', 1.0, 1.0, 137.96))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_steps(directory):
    directory = Path(directory)
    return {int(match.group(1)) for path in directory.glob("checkpoint_*")
            if (match := re.fullmatch(r"checkpoint_(\d+)", path.name))}


def available_steps(run):
    run = Path(run)
    return checkpoint_steps(run / "checkpoints") | checkpoint_steps(run / "ema_snapshots")


def common_step(run_root):
    sets = [available_steps(Path(run_root) / name) for name, _, _, _ in SETTINGS]
    common = set.intersection(*sets)
    if not common or max(common) == 0:
        raise ValueError("No shared trained checkpoint yet. Wait for all four runs to "
                         "reach an EMA milestone (default: every 25,000 steps).")
    return max(common)


def checkpoint_location(run, step):
    run = Path(run)
    for kind in ("ema_snapshots", "checkpoints"):
        if step in checkpoint_steps(run / kind):
            return run / kind, kind
    raise ValueError(f"Step {step} is not saved in {run}")
