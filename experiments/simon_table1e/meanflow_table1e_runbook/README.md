# MeanFlow Table 1(e): final-run recording and launch

This bundle calls your existing, smoke-tested author code. It launches **one fresh 80-epoch trajectory for each p**, on five separate B200s. It does not resume the 201-step diagnostic checkpoint, change the loss/optimizer/model, re-encode data, install packages, or run another training benchmark.

Read this before launch. The launch controller starts all five full runs after its preflight succeeds. It refuses to launch if output directories contain files or a previous final-launch lock exists. Do not run the launch command twice or remove locks to bypass a failed check.

## What is being reproduced

Sources: *Mean Flows for One-step Generative Modeling*, Table 1(e) on paper page 8; Appendix A, Table 4 on page 21; the released repository at commit `d70cb55d298ee03c53bf6da67bec281082e4e2d9`; and your supplied Group Assignment 2 instructions. The paper's VAE footnote identifies the same Flax VAE repository used by the code.

| Setting | This experiment |
|---|---|
| Dataset | ImageNet-1K ILSVRC classification training split, 1,281,167 images / 1,000 classes |
| Resolution / cached posterior | RGB 256×256; posterior mean and std stored as float32 [8,32,32] |
| VAE | `pcuenq/sd-vae-ft-mse-flax`, fixed revision `7581b0a0489cc8483c876a728b830b9ce087cf03` |
| Model | MeanFlow / DiT-B/4, depth 12, width 768, 12 heads, patch 4; 131,165,248 parameters |
| Training | 80 epochs from scratch; 5,004 steps/epoch; 400,320 updates/run |
| Batch | Global 256; one GPU/process, so 256/device |
| Optimizer | Released `optax.adamw`, weight decay 0, β1=0.9, β2=0.95, constant lr=1e-4 |
| EMA | Constant 0.9999 |
| Seeds | Training 42; sampling 0; same in all arms |
| Time sampling | Logit-normal (−0.4, 1.0); r=t for 75% of examples |
| Class labels / guidance | Class conditional, class dropout 0.1; `guidance_eq: none` |
| Precision | Released FP32 setting; no mixed-precision modification |
| Varied setting | `norm_p` = 0, 0.5, 1, 1.5, 2; all other effective config values identical |
| Loss stabilizer | Released `norm_eps=0.01` |
| Evaluation | EMA, 50,000 generated images, 1 NFE, author's JAX FID/resize/Inception and reference statistics |
| Evaluation schedule | After epochs 10,20,...,80; primary result is epoch 80 |
| Checkpoints | Every 10 epochs; authors' retention of last three checkpoints |

| p | Published Table 1(e) FID |
|---|---:|
| 0.0 | 79.75 |
| 0.5 | 63.98 |
| 1.0 | 61.06 |
| 1.5 | 66.57 |
| 2.0 | 69.19 |

The adaptive objective is L / stop_gradient((L+c)^p), with L the per-example squared residual summed over latent elements. For p=0 the denominator is 1; no special p=0 branch is needed. The paper gives c=1e-3 as an **example**, but does not specify Table 1(e)'s exact c or its historical seed. Retain and disclose the released code's c=0.01 and seed42. Matching published settings does not establish identical historical execution or guarantee the same FIDs.

## Changes introduced by this bundle

The **only new execution change** is a host-side recording wrapper (`audit_run.py`) around the existing `main.py`, scalar writer, checkpoint saver and FID functions. It forwards their inputs and returns their original outputs. It never replaces the train step, loss, random number generation, optimizer, EMA update, sampler or FID calculations. It adds file writes and timing overhead.

It saves effective configs, scalar metrics, timings, generated FID moments and fixed qualitative grids. The grid is the first16 images of the already generated FID sample array, with no extra sampling. It fails explicitly on non-finite logged metrics or an incorrect evaluation sample count, rather than marking an incomplete result successful. GPU telemetry is sampled every10 seconds. A controller writes the final comparison automatically when all processes finish.

Your earlier disclosed compatibility changes remain in `provenance/gpu-compat.patch`, copied to `full-run-gpu-compat.patch` and archived with actual source. They include single-process JAX startup, pinned local VAE loading, modern JAX/SciPy/Keras API adaptations, spawn workers, nonempty-cache protection, frozen original unsorted latent-file order, FID feature workers=0, and the 16-image preview-grid fix. IPC sockets use local `/tmp`; model weights, data and all important outputs stay under `/work`.

We preserve released single-process data behavior: the distributed sampler is not advanced between epochs when process_count=1, and LatentDataset's `use_flip` argument does not apply flipping. The frozen manifest makes that same order explicit across all five arms; the authors' historical filesystem order is unknown. Describe these facts rather than claiming access to their exact original data ordering.

## Start on UCloud

Upload `meanflow_table1e_runbook.zip` to your persistent `/work/meanflow-table1e` folder using UCloud's file interface. Run in your normal zsh, from that folder. No interactive strict-mode options or shell configuration changes are needed.

```zsh
cd /work/meanflow-table1e
unzip -n meanflow_table1e_runbook.zip
venv/bin/python meanflow_table1e_runbook/preflight.py
```

**Proceed only if it prints `FULL-RUN PREFLIGHT PASSED`.** This check does zero training steps. It checks the successful smoke, unchanged package versions/source patch/configs, all cache filenames, recorded tensor spot checks, VAE files/revision, reference and Inception hashes, five idle B200s, writable local sockets, and recording-function interfaces. Its manifest and source archive are saved in `provenance`. A failed assertion names the issue; send its output for correction, without bypassing it.

Then launch once:

```zsh
nohup venv/bin/python -u meanflow_table1e_runbook/launch_all.py > provenance/final-controller.log 2>&1 < /dev/null &
```

The controller repeats preflight before starting. GPU0 gets p1.0 (the smoke-tested GPU); GPU1 p0.0; GPU2 p0.5; GPU3 p1.5; GPU4 p2.0. Each child sees its single physical GPU UUID as logical CUDA device0.

Inspect early progress:

```zsh
venv/bin/python meanflow_table1e_runbook/status.py
tail -n 30 provenance/final-controller.log
```

Wait until the controller says **all five runs have reached at least100 steps with finite logged metrics**, and `status.py` shows five running arms. Check `nvidia-smi` for five training processes. Compilation can take a few minutes and metric windows differ; a missing line initially is not proof of a crash. If an arm fails, its `training.log`, `launcher.log`, `run-summary.json` and exit status identify it. Nothing automatically restarts training or substitutes a different configuration. The short smoke checked p1; successful completion of all five full configurations is established by these runs themselves.

The controller and child processes use nohup and file logs, so SSH disconnection does not end them. UCloud walltime/node termination still ends processes. All logs/checkpoints already written remain in persistent storage. A checkpoint may support recovery of the same trajectory, but this bundle does not automatically resume or start a second attempt.

## What you will have for the report

| File under `/work/meanflow-table1e` | Purpose |
|---|---|
| `results/table1e.csv` | Final epoch80 FID, paper FID, signed difference, rank, completion status, runtime, throughput and sampled peak VRAM/utilization |
| `results/fid-learning-curves.csv` | FIDs at epochs10–80 and evaluation phase times |
| `results/report-summary.json` | Machine-readable results, runtime scope and limitations |
| `provenance/reproducibility-manifest.json` | SHA/branch/patch, source/config hashes, seeds, versions, GPU UUIDs/driver, VAE and FID provenance, preprocessing and validation scope |
| `provenance/full-run-source.tar.gz` | Actual source/configs and these recording scripts |
| `provenance/full-run-packages.txt`, `full-run-nvidia-smi.txt`, `full-run-lscpu.txt` | Exact environment records; lscpu is physical host information, not proof of job CPU quota |
| `provenance/parallel-run-timing.json` | Five-arm block elapsed time and estimated GPU-hours for that block |
| `runs/p_*/config.yml`, `effective-config.json`, `command.json`, `runtime.json` | Exact per-arm effective settings, arguments, whitelisted runtime variables and GPU selection |
| `runs/p_*/training.log`, TensorBoard event files | Complete native log and native metrics/preview events |
| `runs/p_*/metrics.csv` | loss, v_loss, learning rate, steps/sec and FID with timestamps/steps/epochs |
| `runs/p_*/fid.csv`, `fid/epoch_*.npz` | Per-evaluation FID and generated feature mean/covariance with hashes |
| `runs/p_*/qualitative/epoch_*.png` | Fixed first16 images from that epoch's EMA FID generation |
| `runs/p_*/events.jsonl`, `progress.json`, `run-summary.json` | Timing/progress, compile messages, checkpoint events, final success/failure |
| `runs/p_*/gpu.csv`, `start.txt`, `end.txt`, `exit-status.txt` | Sampled utilization/VRAM/power, dates, process outcome |
| `runs/p_*/checkpoint_*`, `final-checkpoint-files.json` | Author checkpoints and final checkpoint file hashes |

Results are collected automatically after all processes end. To regenerate the tables or inspect currently completed arms, safely run:

```zsh
venv/bin/python meanflow_table1e_runbook/collect_results.py
```

This reads recorded files only. It never trains or evaluates. Failed/incomplete arms stay marked; they do not become fabricated final results.

Keep the separate preprocessing logs, cache-verification JSON, original VAE snapshot record, dataset download/version/count records and FID-reference download record already in `provenance`. Do not publish authentication tokens, credentials, Kaggle account files or old pasted messages containing credentials. This bundle records only selected runtime variables and never reads credentials.

## Runtime and interpretation

Five parallel runs take approximately the **slowest run's elapsed time**, not the sum of five runtimes. The p1 smoke's last100-step window projected about4.96 hours; its combined200-step average projected8.63 hours for training alone. These are warm-up-sensitive short measurements. Five concurrent loaders, p-dependent compute and evaluation can increase elapsed time; **nine hours is not an upper bound**. Eight periodic FID-50K evaluations per run, compilation, previews, checkpointing and logging add time. At9 elapsed hours with all five GPUs occupied, the block is45 GPU-hours; actual allocation billing also includes other time the node remains allocated.

The reported end-to-end Python runtime includes startup/compilation, data I/O, all8 evaluations, preview/checkpoint operations, logging and final hashing. Separately recorded FID-generation, feature/statistics, FID-math and checkpoint times explain where time went. Native logged throughput is retained; its median afterstep500 is a descriptive summary, not a synchronized hardware benchmark. Whole-run GPU peak telemetry includes VAE/Inception compilation/autotuning and can miss peaks shorter than10 seconds. Do not compare B200 timing to TPU timing as a controlled algorithmic speed experiment.

The adaptive weighted training loss has a different scale for each p and is approximately1 for p1. It should not determine which arm is best. `v_loss` is the released velocity-error diagnostic; it is not the exact adaptive target residual or a substitute for FID. Assess the conclusion primarily from final FID ranking and signed differences against Table1(e). Show intermediate FIDs as learning curves, and fixed sample grids as supporting qualitative evidence. Do not choose the minimum intermediate FID as the table's headline result.

There is **one seed per arm**. Report that scope explicitly; do not claim standard deviations, confidence intervals or significance across training runs. Discuss whether p1 remains best or near-best and whether adaptive p values improve on p0, with absolute FID differences. If the ranking/FIDs differ, discuss GPU/software, unknown historical seed/c/data ordering, finite-sample evaluation and any actual execution failure rather than claiming exact reproducibility.

## Assignment report checklist

This is your loss-metric ablation contribution under the **Replication Track**. The supplied assignment requests a subset of paper experiments and one experiment per member; the five-arm ablation is one coherent experiment for your contribution. It does not require reproducing every Table1 panel or specify a minimum seed count. Coordinate the final group report and name each member's concrete contribution.

Use the ISBI2027 template, at most4 pages excluding references/figures/tables. Put group number, names and track on the front page; include a contribution statement and GitHub repository link. Explain the reproduced question, objective and controlled settings; describe dataset/VAE/evaluation provenance and compatibility changes; present epoch80 FIDs, differences/ranking, learning curves and clearly scoped runtime; discuss conclusions and limitations. The course exam/AI-use/evaluation policy must also be reviewed; it was not included in the supplied assignment file and is not established by this bundle.

Before stopping the UCloud job, confirm all five exit statuses are0, all final checkpoints exist, and the collector reports5/5 completed epoch80 results. The bundle does not stop or release your job.

## Validation of this bundle

The scripts were syntax checked, and recording/result selection was exercised with fake author interfaces to verify unchanged forwarding, 1-based epoch labeling, final-FID selection, and rejection of invalid completions. These are checks of the recorder, not training/FID experiments. The UCloud preflight checks the actual installed interfaces without training. The previously completed batch256 B200 smoke remains the computational validation; the full80-epoch/FID50K runs have not yet completed.
