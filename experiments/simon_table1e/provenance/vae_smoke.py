import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers.models import FlaxAutoencoderKL
from huggingface_hub import HfApi, snapshot_download
from torchvision import transforms

root = Path("/work/meanflow-table1e")
sys.path.insert(0, str(root / "meanflow"))
from utils.input_pipeline import center_crop_arr, loader

repo = "pcuenq/sd-vae-ft-mse-flax"
revision_file = root / "provenance/vae-revision.txt"
revision = (
    revision_file.read_text().strip()
    if revision_file.exists()
    else HfApi().model_info(repo).sha
)
revision_file.write_text(revision + "\n")

model_dir = root / "models/sd-vae-ft-mse-flax"
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=model_dir,
    allow_patterns=["config.json", "diffusion_flax_model.msgpack"],
)

metadata = {"repo_id": repo, "revision": revision, "sha256": {}}
for name in ["config.json", "diffusion_flax_model.msgpack"]:
    path = model_dir / name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata["sha256"][name] = digest.hexdigest()

(root / "provenance/vae-snapshot.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
print("VAE repository:", repo)
print("Pinned revision:", revision)

vae, params = FlaxAutoencoderKL.from_pretrained(
    str(model_dir), local_files_only=True, dtype=jnp.float32
)
assert not getattr(vae, "_missing_keys", set()), "Missing VAE weights"

devices = jax.local_devices()
print("GPU devices:", devices)
assert len(devices) == 5
assert all(device.platform == "gpu" for device in devices)

train = root / "data/raw/ILSVRC/Data/CLS-LOC/train"
first_class = sorted(path for path in train.iterdir() if path.is_dir())[0]
images = sorted(
    path for path in first_class.iterdir()
    if path.suffix.lower() == ".jpeg"
)[:5]
assert len(images) == 5

transform = transforms.Compose([
    transforms.Lambda(lambda image: center_crop_arr(image, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
])
batch = torch.stack([transform(loader(str(path))) for path in images])
batch = batch.numpy().reshape(5, 1, 3, 256, 256)

(root / "provenance/vae-smoke-inputs.txt").write_text(
    "\n".join(str(path.relative_to(root)) for path in images) + "\n"
)

def roundtrip(images):
    posterior = vae.apply(
        {"params": params}, images, method=FlaxAutoencoderKL.encode
    ).latent_dist
    decoded = vae.apply(
        {"params": params}, posterior.mean, method=FlaxAutoencoderKL.decode
    ).sample
    return posterior.mean, posterior.std, decoded

print("Compiling and running the VAE on all five GPUs...", flush=True)
outputs = tuple(np.asarray(x) for x in jax.pmap(roundtrip)(batch))

expected = [(5, 1, 32, 32, 4), (5, 1, 32, 32, 4), (5, 1, 3, 256, 256)]
for name, value, shape in zip(["mean", "std", "decoded"], outputs, expected):
    assert value.shape == shape, (name, value.shape)
    assert np.isfinite(value).all(), f"Non-finite values in {name}"
    print(name, "shape:", value.shape, "dtype:", value.dtype)

assert (outputs[1] > 0).all(), "Invalid posterior standard deviations"
print("VAE SMOKE TEST PASSED")
