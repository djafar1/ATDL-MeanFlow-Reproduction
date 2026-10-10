from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
import yaml

root = Path("/work/meanflow-table1e")
repo = root / "meanflow"
sys.path.insert(0, str(repo))
from configs.default import get_config

def edit(relative, old, new):
    path = repo / relative
    text = path.read_text()
    if new in text:
        return
    assert text.count(old) == 1, f"Unexpected source: {relative}"
    text = text.replace(old, new)
    compile(text, str(path), "exec")
    path.write_text(text)

order = root / "provenance/latent-file-order.txt"
if not order.exists():
    names = [x for x in os.listdir(root / "data/imagenet256_latents/train")
             if x.endswith(".pt")]
    assert len(names) == 1_281_167
    assert len(set(names)) == len(names)
    order.write_text("\n".join(names) + "\n")
names = order.read_text().splitlines()
assert len(names) == 1_281_167 and len(set(names)) == len(names)

old = "        self.file_list = [file for file in os.listdir(root) if file.endswith('.pt')]"
edit(
    "utils/vae_util.py", old,
    "        order_file = os.environ.get('MEANFLOW_LATENT_MANIFEST')\n"
    "        if order_file:\n"
    "            with open(order_file) as f:\n"
    "                self.file_list = f.read().splitlines()\n"
    "        else:\n" + "    " + old,
)
edit(
    "train.py",
    "fid_util.compute_stats(samples_all, inception_net)",
    "fid_util.compute_stats(samples_all, inception_net, num_workers=0)",
)
edit(
    "utils/vis_util.py",
    "n = col * row * max_bz",
    "n = min(n // (col * row), max_bz) * (col * row)",
)

base = get_config().to_dict()
overrides = yaml.safe_load((repo / "configs/run_b4.yml").read_text())
for key, value in overrides.items():
    if isinstance(value, dict):
        base[key].update(value)
    else:
        base[key] = value

base["dataset"]["root"] = str(root / "data/imagenet256_latents")
base["training"]["num_epochs"] = 80
base["method"]["guidance_eq"] = "none"
base["fid"]["cache_ref"] = str(root / "data/fid/imagenet256_authors_ref.npz")

assert base["model"]["cls"] == "DiT_B_4"
assert base["dataset"]["image_size"] == 32
assert base["dataset"]["image_channels"] == 4
assert base["dataset"]["vae"] == "mse"
assert base["training"]["batch_size"] == 256
assert base["training"]["seed"] == 42
assert base["training"]["half_precision"] is False
assert base["training"]["ema_val"] == 0.9999
assert base["sampling"]["num_steps"] == 1
assert base["fid"]["num_samples"] == 50_000
assert base["fid"]["device_batch_size"] == 20
assert base["load_from"] is None
assert base["eval_only"] is False

records = []
for p in [0.0, 0.5, 1.0, 1.5, 2.0]:
    cfg = deepcopy(base)
    cfg["method"]["norm_p"] = p
    normalized = deepcopy(cfg)
    normalized["method"]["norm_p"] = base["method"]["norm_p"]
    assert normalized == base
    path = repo / f"configs/table1e_p_{p:.1f}.yml"
    content = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        assert path.read_text() == content, f"Existing config differs: {path}"
    else:
        path.write_text(content)
    (root / f"runs/p_{p:.1f}").mkdir(parents=True, exist_ok=True)
    records.append({
        "p": p, "config": str(path),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    })
    print(f"Verified p={p:.1f}: 80 epochs, batch 256, no guidance, FID-50K, 1 NFE")

with order.open("rb") as f:
    order_hash = hashlib.file_digest(f, "sha256").hexdigest()
(root / "provenance/table1e-configs.json").write_text(
    json.dumps({"configs": records, "file_order_sha256": order_hash}, indent=2) + "\n"
)
subprocess.run(["git", "-C", str(repo), "diff", "--check"], check=True)
with (root / "provenance/gpu-compat.patch").open("w") as f:
    subprocess.run(["git", "-C", str(repo), "diff"], stdout=f, check=True)
with (root / "provenance/git-status-current.txt").open("w") as f:
    subprocess.run(["git", "-C", str(repo), "status", "--short"], stdout=f, check=True)
print("EXPERIMENT CONFIGS READY — full training has not started.")
