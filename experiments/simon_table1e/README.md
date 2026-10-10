# Simon: Table 1(e), loss metrics

ImageNet-256, MeanFlow/DiT-B/4, 80 epochs (400320 updates), batch 256,
one NVIDIA B200 per run. The five norm_p values (0, 0.5, 1, 1.5, 2) were
trained with seeds 42 and 43. Within each sweep only norm_p changes;
sampling seed is 0, guidance is disabled and norm_eps is 0.01.

[Results](../../results/simon_table1e/processed/table1e_two_seeds.csv) retain
epoch-80 EMA FID-50K at 1 NFE, both individual seeds and mean ± sample SD
(ddof=1, n=2). The second sweep was added after inspecting the first.
The report discusses the persistent p=2 discrepancy and limited seed evidence.

## Files

meanflow/ contains the code used for the runs, based on
[Gsunshine/meanflow](https://github.com/Gsunshine/meanflow) commit
`d70cb55d298ee03c53bf6da67bec281082e4e2d9`, with all ten original configs and
the authors' MIT license. provenance/full-run-gpu-compat.patch shows changes
for single-process GPU startup, local VAE loading, JAX/SciPy APIs, spawned
loader workers and fixed filename order. The source archive and manifest record
the code used. included-files-sha256.json lists the SHA256 hashes of the copied files.

runs/ contains records for ten full 80-epoch runs:

- runs/p_0.0 through runs/p_2.0: training seed 42.
- runs/diagnostic_p_0.0_seed43 through runs/diagnostic_p_2.0_seed43: training seed 43.

The diagnostic label does not mean shorter training. Each folder contains
metrics.csv, fid.csv, configs, timing records and a run summary. The smoke test
is excluded. final-checkpoint-files.json records checkpoint hashes; weights
are not included. environment/ lists the training and preprocessing packages.

meanflow_table1e_runbook/, seed43_runbook/ and preprocessing helpers in
provenance/ are the original scripts, including their UCloud paths and GPU
assignments. Their READMEs describe that setup.
Use [CPU analysis](analysis/README.md) to check the included records and make
the tables and plots. The original seed-43 collector also checks checkpoints.

## Fresh training

scripts/run.py creates local configs from the recorded effective configs,
changes the data paths and calls the original training recorder. It checks
package versions and file hashes. Output directories must be empty and outside
experiments/simon_table1e/. Run these commands from the repository root; replace
/path/to/... with local paths.

### Environment

Use Python 3.12.3 and the recorded GPU environment: JAX/JAXlib 0.11.2 with
CUDA 13 plugins, Flax 0.12.10, Optax 0.2.8, Diffusers 0.35.1, NumPy 2.5.3,
SciPy 1.18.1, TensorFlow CPU 2.21.0 and CPU Torch/Torchvision 2.14.1/0.29.1.
NVIDIA driver 610.57.04 and CUDA UMD 13.3 were recorded. Five B200s ran
concurrently per sweep; one GPU can run the arms sequentially.

```sh
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r experiments/simon_table1e/environment/full-run-packages.txt
.venv-training/bin/python -m pip check
CUDA_VISIBLE_DEVICES=0 .venv-training/bin/python -c \
  'import jax; print(jax.devices()); assert jax.devices()[0].device_kind == "NVIDIA B200"'
```

The package list gives exact versions but has no wheel hashes. If a package
cannot be installed, do not replace it with a newer version without checking it.
Do not use meanflow/scripts/install.sh: it installs upstream's TPU JAX 0.4.27.
The GPU environment differs from the authors' release, so it does not give
bitwise replication. Peak memory in the smoke test was 137752 MiB. Peak memory
for the full runs was not recorded.

### Data and VAE

Get the ILSVRC ImageNet-1K training split through authorized ImageNet or
Kaggle access and follow the dataset terms. We used Kaggle's
imagenet-object-localization-challenge archive. Keep credentials outside Git.
Expected ImageFolder layout:
`/path/to/ILSVRC/Data/CLS-LOC/train/<synset>/<image>.JPEG`, with 1000 classes
and 1281167 images. The dataset and large source archive are excluded.

Download the VAE at the recorded revision:

```sh
.venv-training/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="pcuenq/sd-vae-ft-mse-flax",
    revision="7581b0a0489cc8483c876a728b830b9ce087cf03",
    local_dir="/path/to/vae",
    allow_patterns=["config.json", "diffusion_flax_model.msgpack"],
)
PY
```

Expected hashes are in provenance/vae-snapshot.json. Use this revision:
the original vae_smoke.py can fall back to the latest revision.

Encode the images with the preprocessing script. The latent output must be
empty; the script refuses to append to an existing cache.

```sh
MF_PYTHON="$PWD/.venv-training/bin/python"
export MEANFLOW_SINGLE_PROCESS=1
export MEANFLOW_VAE_PATH=/path/to/vae
export HF_HOME=/path/to/hf_cache
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd experiments/simon_table1e/meanflow
CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$MF_PYTHON" prepare_dataset.py \
  --imagenet_root=/path/to/ILSVRC/Data/CLS-LOC \
  --output_dir=/path/to/imagenet256_latents \
  --vae_type=mse --image_size=256 --batch_size=32 \
  --compute_latent=True --compute_fid=False --overwrite=False
cd ../../..
```

We encoded the images with five GPUs. The code pads batches when the GPU
count differs, but we have not checked whether this gives the same cache.
Encoding uses RGB, ADM center crop, normalization to [-1,1] and FP32 cached
posterior mean/std of shape [8,32,32]. Training samples [4,32,32] latents and
scales by 0.18215. Filenames are 00000000.pt through 01281166.pt.
provenance/latent-file-order.txt fixes their training order. We checked all
filenames and counts, then checked values and labels in 100 tensors.
We did not check the contents of every tensor.

### FID assets

Download the authors' reference from the URL in provenance/fid-reference.json
to /path/to/imagenet256_authors_ref.npz. SHA256:
`a8e1d8cabbc2cd5137e009dc3c5ad8c410506e9dfcf5d1717a84967108e95d06`.
Download Inception weights from the URL in meanflow/utils/jax_fid/inception.py
to /path/to/jax_fid/inception_v3_weights_fid.pickle. SHA256:
`4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77`.
scripts/run.py checks these hashes and the VAE hashes before loading the files.

The authors' loop evaluates every ten epochs: EMA, 50000 decoded 256×256
images, random class labels, sampling seed 0, 1 NFE, generation batch 20,
Inception feature batch 200, released legacy_tensorflow resize and authors'
reference statistics. Report epoch 80; no separate evaluation is needed.

### Configs and launch

Generate ten local configs:

```sh
.venv-training/bin/python experiments/simon_table1e/scripts/run.py prepare-configs \
  --config-dir local/simon_table1e/configs \
  --latents /path/to/imagenet256_latents \
  --fid-reference /path/to/imagenet256_authors_ref.npz
```

Example for p=1, seed 42; --check validates files without training:

```sh
.venv-training/bin/python experiments/simon_table1e/scripts/run.py train \
  --config-dir local/simon_table1e/configs \
  --output-dir local/simon_table1e/runs/p_1.0_seed42 \
  --p 1 --seed 42 --gpu 0 --vae-dir /path/to/vae \
  --inception-dir /path/to/jax_fid --check
```

Remove --check to train. Run p=0, 0.5, 1, 1.5, 2 for each seed 42 and 43,
with distinct empty output directories. --gpu accepts a physical index or
UUID. Give each parallel run its own GPU and keep all other settings fixed.
Write logs outside the included records. The script records the command,
config, metrics, FID timings, summary and exit status using the original
recorder. It checks for a B200 and final checkpoints and does not resume runs.

We checked config generation and recorded-results analysis locally. We have
not tested a fresh GPU installation, downloads, encoding or training with
scripts/run.py. The reported results come from the original completed runs.
