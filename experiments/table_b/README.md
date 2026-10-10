# Table 1(b): JVP tangent ablation

Reproduce the four rows of Table 1(b) in [Mean Flows for One-step Generative
Modeling](https://arxiv.org/html/2505.13447v1#S5.T1).

| Run | Paper tangent `(z,r,t)` | Code tangent `(z,t,r)` | Paper FID, 1-NFE |
| --- | --- | --- | ---: |
| `v01` (normal baseline) | `(v,0,1)` | `(v_g,1,0)` | 61.06 |
| `v00` | `(v,0,0)` | `(v_g,0,0)` | 268.06 |
| `v10` | `(v,1,0)` | `(v_g,0,1)` | 329.22 |
| `v11` | `(v,1,1)` | `(v_g,1,1)` | 137.96 |

The baseline is included: **four training runs total**. Each starts independently
from seed 42. Only `jvp_tangent_r` and `jvp_tangent_t` differ between the four
YAML files in `configs/table_b/`. The paper and code order the time arguments
differently; the adapter maps them explicitly.

## Fixed recipe

- MF-B/4, full ImageNet-1K, 256-pixel images encoded into VAE latents.
- Batch 256, 80 epochs, 5,004 updates per epoch: **400,320 updates**.
- Learning rate 0.0001; Adam betas (0.9, 0.95); weight decay zero.
- Constant EMA 0.9999; logit-normal time sampling (-0.4, 1.0); 25% unequal times.
- No guidance: `guidance_eq: none`, `omega: 1.0`, `kappa: 0.0`.
- Adaptive loss weighting `norm_p: 1.0`, epsilon 0.01.
- One-step EMA FID with exactly 50,000 images, generation batch 20, seed 0.

The completed runs used unchanged upstream source, preserved under
`results/runs/table-b-first/source/src/meanflow/`. The shared `src/meanflow/`
now includes the group's GPU compatibility patches. `table_b_model.py` adapts
the upstream forward loss to change only the JVP time tangents. Optimizer, training update,
latent scaling, EMA, network, and sampling come from the released code. Cached
records are sorted for stable indexing; the upstream single-process sampler's
fixed epoch is retained on resume. The released guided 240-epoch B/4 checkpoint
is a different experiment and cannot supply this table's 80-epoch baseline.

## Prepare ImageNet on Hendrix

First open `hendrix tui` locally: **c** connects, **p** previews changes, and
**s** syncs the updated project files into `REPO/.hendrix/source/`, not the
repository root. The commands below assume updated files in the repository root
(for example after a direct source sync). When using the TUI mirror directly,
run from `.hendrix/source/`, pass `--venv /home/fhz806/projects/ATDL-MeanFlow-Reproduction/.venv-hendrix`
to the launcher, and use absolute paths for the run root and Python environment.
For preparation helpers, export `MEANFLOW_VENV` to that same environment path. Neither
preparation job starts training.

The completed experiment already has a prepared latent cache on Hendrix.
The following commands are for preparing a new dataset. The download helper
retains the archive and resumes class folders marked complete:

```bash
export MEANFLOW_DATA_ROOT=/home/fhz806/meanflow-data
df -h "$MEANFLOW_DATA_ROOT"
quota -s
sbatch scripts/hendrix/imagenet_download.sbatch "$MEANFLOW_DATA_ROOT"
```

This uses four CPUs, 8 GB RAM, and up to 48 hours, without a GPU. It verifies the
archive MD5, unpacks both tar layers, and checks 1,281,167 images in 1,000 class
folders. Watch `imagenet-download-JOB_ID.out`. If the input/output error repeats,
resolve the filesystem issue before retrying again. Allow roughly 450 GB for
archive, images, latents, checkpoints, and snapshots. The latent tensors alone
are about 42 GB; checkpoints across four B/4 runs are roughly 56 GB.

Once the log reports `Raw ImageNet is ready` and the job is `COMPLETED`, choose
an exact GPU model from `sinfo` and encode the latents. This example uses `a100`;
replace it with the model you choose:

```bash
sinfo -p gpu -o '%N %G'
sbatch --gres=gpu:a100:1 scripts/hendrix/imagenet_latents.sbatch \
  "$MEANFLOW_DATA_ROOT/imagenet" "$MEANFLOW_DATA_ROOT/imagenet_latents"
```

Encoding requests four CPUs, 32 GB RAM, and up to 24 hours. It uses the upstream
MSE VAE at image size 256 and batch 32, with FID statistics disabled. It verifies
all 1,281,167 indexed records and representative tensor shapes and labels.
The cache layout is `imagenet_latents/train/00000000.pt`, and so on. Each record
stores unscaled mean/std of shape `(8,32,32)` and a label from 0 to 999.

Interrupted encoding requires a **fresh output directory**; this encoder does
not resume a partial cache. The helper accepts an existing cache only if it
passes validation. Reuse one complete cache for all four training runs.
Download the authors' FID reference from the [upstream README](../../src/meanflow/README.md#inference)
before evaluation. Cache VAE and Inception weights if compute nodes cannot
access their download servers.

## Preview and submit training

Choose `h100`, `a100`, `l40s`, or `a40` if offered by `sinfo`. The completed
experiment ran on H100, A100, and L40S with batch 256. A40 has not been validated.
There is no automatic
GPU fallback, batch reduction, or gradient accumulation. An allocation failure
saves the available state and tells you to resume on a GPU with more memory.

```bash
.venv-hendrix/bin/python scripts/hendrix/submit_table_b.py \
  --dataset-root /home/fhz806/meanflow-data/imagenet_latents \
  --run-root "$PWD/results/runs/table-b-first" \
  --gpu-type a100
```

This is a preview: no files are created and no jobs submitted. Repeat with
`--submit` when ready. The launcher checks the selected GPU resource and full
latent cache before creating a source/config snapshot and submitting Slurm jobs.
The snapshot includes uncommitted project changes; data and the environment
remain shared.

Defaults: **four concurrent jobs**, one selected GPU, eight CPUs, 32 GB host RAM,
and 48 hours per job. Use `--concurrency 2` to limit simultaneous jobs while still
running all four settings. Scheduling can stagger their start times. The internal
47h30m timer and Slurm's pre-expiry signal finish the current update and save
state. Training stops at 80 epochs or its time limit; evaluation is separate.
Four full allocations use up to 192 GPU-hours. A one-day completion would require
about 4.63 updates/second, excluding preparation and evaluation; measure your run
before assuming that timeline.

## Monitor and resume

```bash
squeue -u "$USER"
tail -f results/runs/table-b-first/slurm-ARRAY_ID_0.out
cat results/runs/table-b-first/v01/summary.json
```

Array indices 0–3 correspond to `v01`, `v00`, `v10`, `v11`. Each run stores resolved
configuration, dataset metadata, metrics, rolling full checkpoints, retained EMA
snapshots, and a summary. `complete` means 80 epochs reached; `stopped` means a
graceful time/signal exit. Check the Slurm exit status too. A hard kill can lose
updates after the latest full checkpoint.

Full optimizer/EMA checkpoints save every 1,000 steps (latest three retained).
EMA snapshots save every 25,000 steps and on normal exit, enabling comparison
at the largest step saved by all four runs.

To continue unfinished runs, choosing any supported GPU model:

```bash
.venv-hendrix/bin/python scripts/hendrix/submit_table_b.py \
  --resume "$PWD/results/runs/table-b-first" --gpu-type a100
```

Add `--submit` when ready. Resume reuses the saved source, configuration, and
dataset, restores optimizer/EMA/step, and skips consumed batches in the current
epoch. Completed settings are omitted. Table 1(e) experiments cannot be resumed
as Table 1(b); their checkpoints belong to different experiments.

## Evaluate and collect results

After training has stopped, select one shared checkpoint on the gateway:

```bash
run_root="$PWD/results/runs/table-b-first"
step=$(.venv-hendrix/bin/python scripts/hendrix/table_b_results.py common-step "$run_root")
sbatch --gres=gpu:a100:1 \
  --output="$run_root/fid-%A_%a.out" --error="$run_root/fid-%A_%a.err" \
  "$run_root/source/scripts/hendrix/table_b_eval.sbatch" \
  "$run_root" "$step" /absolute/path/to/authors_fid_reference.npz \
  "$PWD/.venv-hendrix"
```

The four-task evaluation array runs one GPU at a time by default; add
`--array=0-3%4` for four concurrent evaluations. The initial request is four hours
per task; adjust if needed. Sampling uses batch 20 and streams decoded images
into the upstream Inception pipeline. Keep sample batch size identical across
runs because it affects RNG assignment. Results go in `v*/eval-STEP/fid.json`
and `samples.png`. Retrying requires a fresh evaluation output directory.

After all four evaluations complete:

```bash
.venv-hendrix/bin/python scripts/hendrix/table_b_results.py collect "$run_root" --step "$step" \
  > "$run_root/table-b.csv"
```

The collector checks tangent mappings, shared steps, datasets, sampling settings,
reference hashes, and fixed loss/guidance settings. Copy the CSV, grids, configs,
metrics, and summaries to your PC. Report actual epochs and compute time. If
training stops short of 80 epochs, describe a reduced-duration reproduction.
`v01` supplies both your baseline and the first row of the table.

## Validation

Run `python3 -m unittest discover -s tests -v` for launcher/configuration checks.
Set `MEANFLOW_CPU_INTEGRATION=1` in a CPU ML environment for analytical tangent
checks, exact baseline loss/gradient equivalence with upstream, actual tiny-model
updates for all four tangents, exact mid-epoch resume, graceful signal/time-limit
checkpoints, and EMA evaluation with partial sample-batch trimming. VAE/Inception
are mocked in the evaluation data-flow test. Real FID quality and GPU memory/
throughput require cluster execution.
