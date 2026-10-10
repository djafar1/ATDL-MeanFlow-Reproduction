from pathlib import Path
import subprocess
import sys

root = Path("/work/meanflow-table1e")
repo = root / "meanflow"

def edit(relative, old, new):
    path = repo / relative
    text = path.read_text()
    if old not in text and new in text:
        return
    assert text.count(old) == 1, f"Unexpected source in {relative}"
    updated = text.replace(old, new)
    compile(updated, str(path), "exec")
    path.write_text(updated)
    print("Updated:", relative, flush=True)

edit("train.py", "jax.tree_leaves(", "jax.tree_util.tree_leaves(")
edit("train.py", "jax.tree_map(", "jax.tree_util.tree_map(")
edit("utils/ema_util.py", "jax.tree_map(", "jax.tree_util.tree_map(")

edit(
    "meanflow.py",
    "jnp.clip(t - r, a_min=0.0, a_max=1.0)",
    "jnp.clip(t - r, 0.0, 1.0)",
)

edit(
    "utils/vae_util.py",
    "        Bflops = compiled_decod_fn.cost_analysis()[0]['flops'] / 1e9",
    "        analysis = compiled_decod_fn.cost_analysis()\n"
    "        if isinstance(analysis, (list, tuple)):\n"
    "            analysis = analysis[0]\n"
    "        Bflops = analysis['flops'] / 1e9",
)

edit(
    "utils/jax_fid/fid.py",
    "    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)",
    "    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))",
)

path = repo / "utils/jax_fid/fid.py"
text = path.read_text()
old = "\nfrom keras.preprocessing.image import ImageDataGenerator\n"
new = "\n    from keras.preprocessing.image import ImageDataGenerator\n"
if old in text:
    assert text.count(old) == 1
    marker = "    preprocessing_fn = lambda x: x.astype(float) / 255\n"
    assert text.count(marker) == 1
    text = text.replace(old, "\n")
    text = text.replace(marker, new + "\n" + marker)
    compile(text, str(path), "exec")
    path.write_text(text)
    print("Deferred legacy Keras import.", flush=True)
else:
    assert new in text, "Unexpected Keras import location"

path = repo / "utils/input_pipeline.py"
text = path.read_text()
start = text.index("def create_latent_split(")
end = text.index("\ndef create_split(", start)
section = text[start:end]
old = "      num_workers=dataset_cfg.num_workers,\n"
new = old + (
    "      multiprocessing_context='spawn' "
    "if dataset_cfg.num_workers > 0 else None,\n"
)
if new not in section:
    assert section.count(old) == 1
    section = section.replace(old, new)
    text = text[:start] + section + text[end:]
    compile(text, str(path), "exec")
    path.write_text(text)
    print("Training latent loader now uses spawn.", flush=True)

subprocess.run(["git", "-C", str(repo), "diff", "--check"], check=True)
with (root / "provenance/gpu-compat.patch").open("w") as f:
    subprocess.run(["git", "-C", str(repo), "diff"], stdout=f, check=True)
with (root / "provenance/git-status-current.txt").open("w") as f:
    subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        stdout=f, check=True,
    )

sys.path.insert(0, str(repo))

import importlib.metadata as metadata
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import train
from utils.jax_fid.fid import compute_frechet_distance

for package in [
    "jax", "jaxlib", "flax", "torch", "torchvision",
    "diffusers", "scipy", "tensorflow-cpu", "clu", "orbax-checkpoint",
]:
    print(package, metadata.version(package), flush=True)

print("JAX devices:", jax.devices(), flush=True)
assert jax.local_device_count() == 1
assert jax.default_backend() == "gpu"
print("TensorFlow GPU devices:", tf.config.list_physical_devices("GPU"))

zero = np.zeros(2)
identity = np.eye(2)
same = compute_frechet_distance(zero, zero, identity, identity)
known = compute_frechet_distance(
    zero, np.array([1., 2.]),
    np.diag([1., 4.]), np.diag([9., 16.]),
)
assert np.isclose(same, 0.0), same
assert np.isclose(known, 13.0), known
print("FID math checks:", same, known, flush=True)

jax.jit(lambda x: x + 1)(jnp.ones(1)).block_until_ready()
print("TRAINING IMPORT AND FID MATH CHECKS PASSED", flush=True)
