# Loss-metric experiment analysis

This directory supports Simon's Group Assignment 2 experiment. It reads only
`../../meanflow-table1e-light/`; the full and light experiment folders are copies
of the same runs, not independent repetitions. No training, sampling, latent
encoding, or FID computation is performed. Smoke runs are never selected.

From `RAPPORT/`, regenerate the assets with:

```sh
python analysis/analyse_loss_metrics.py
latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Python 3.10+ with NumPy and Matplotlib is required. The verified analysis used
NumPy 2.5.2 and Matplotlib 3.11.1; these are analysis dependencies, separate
from the recorded training environment. Paths are resolved relative to the
script, so it can also be invoked from the assignment root. `--data-root`
selects one refreshed experiment copy; `--output-root` can direct all generated
assets to a separate review directory outside the raw experiment tree.

The recorded input tree is outside the report's Git repository. Compiling the
report uses the checked-in tables and figures; rerunning the verification and
analysis also requires the experiment files. A fresh checkout needs a copy of
`meanflow-table1e-light/`, or an equivalent input tree passed to `--data-root`.
The required files and their SHA256 hashes are listed in
`generated/analysis_summary.json` under `input_sha256`. These small records and
source/provenance files suffice; checkpoints, generated samples and the ImageNet
latent cache are not needed for this analysis.

Keep the section sources, scripts, generated CSV/JSON files, tables and paper
figures in the report repository. `.gitignore` excludes new temporary build files,
Python bytecode and the isolated visualization previews. It does not remove
build files already tracked by the template repository; review those separately
when selecting files for a commit.

## Outputs and update procedure

- `../tables/loss_metrics_two_seeds.tex`: table included by the section.
- `../tables/loss_metrics_seed42.tex`: original seed-42 comparison, retained.
- `generated/table1e_seed42.csv`: full precision paper/ours/signed differences.
- `generated/fid_epochs_seed42.csv`: recorded FID points and evaluation timings.
- `generated/individual_completed_runs.csv`: all ten verified runs, retaining
  both individual scores and signed paper differences, runtime and throughput.
- `generated/fid_epochs_all_verified.csv`: all 80 recorded FID evaluations,
  with training seed, evaluation protocol and timings.
- `generated/run_costs.csv`: measured per-run cost, throughput and timestamps.
- `generated/analysis_summary.json`: effective controls, source checks, software,
  provenance, input hashes, analysis script hash, and seed-43 readiness.
- `../figures/loss_metrics_fid_epochs.{pdf,png}`: measured curves for both seeds,
  included as Figure 1.
- `../figures/loss_metrics_fid_vs_p.{pdf,png}`: available alternative, not included.

The PDFs contain vector lines, markers, text and embedded fonts. PNGs are previews.
The curves connect actual recorded evaluations; they do not extrapolate or insert
missing epochs. The table always uses epoch 80, never the best checkpoint or seed.
The script does not compare adaptive training-loss values across exponents.

All five seed-43 runs are now verified complete, alongside the five original
seed-42 runs. `diagnostic` is a directory label for a full 80-epoch repeat.
The second sweep was added after inspecting the original results, as recorded
in `provenance/seed43-all-plan.json`. Sampling seed remains 0.

The script verifies successful summary and process exit status, 80 epochs and
400,320 updates; final FID agreement among the summary, FID CSV and metrics;
EMA/FID-50K/1-NFE and fixed sampling seed; and matching configurations after
removing only `method.norm_p` and `training.seed`. It also checks the launch
config hashes, identical paired runtime settings, the same physical GPU per
pair, source/manifest provenance and all 33 archived tracked source files.
Both controllers report zero exit codes for every arm.

Only when **all five** seed-43 endpoints pass those checks does the script create:

- `generated/table1e_two_seeds.csv`: paper FID, both individual FIDs and their
  signed differences from the paper, n=2, arithmetic mean, sample SD, signed
  mean difference from the paper, and `paper_minus_mean` for the displayed delta.
  The table uses Delta = paper FID minus our two-seed mean FID; the original
  individual and mean difference columns retain the ours-minus-paper convention. Values retain full recorded precision.
- `../tables/loss_metrics_two_seeds.tex`: both individual FIDs, mean ± SD and
  Delta = paper minus our mean. Minimum FID in each FID column is bold; the SD
  and delta are not ranked. The table keeps the report font size and uses local
  4pt column padding to fit one ISBI column.

SD uses NumPy `std(ddof=1)`, with n=2 training seeds per p. The mean averages
two independently measured epoch-80 FIDs; generated statistics are not pooled.
No best epoch or best seed is selected. Missing or inconsistent transfer
snapshots remain pending, with the exact run path and reason printed and stored
in `analysis_summary.json`. Incomplete copies produce no complete two-seed
aggregate and remove stale generated aggregates; they retain verified individual
results and a measured seed-42 figure. Recheck the section before compiling if
using an incomplete copy, since it currently inputs the completed two-seed table.
Raw experiment files are never rewritten.

Verification additionally checked the generated means, sample SDs and differences
against Python's independent `statistics` implementation. An in-memory incomplete
transfer of the seed-43 p=2 FID CSV was correctly marked pending, withheld the
complete table, and retained nine verified runs. No raw files were changed.

## Verified evidence

| Claim | Local evidence |
|---|---|
| Paper values 79.75, 63.98, 61.06, 66.57, 69.19 | `Mean Flows.pdf`, printed p.8, Table 1(e) |
| 80 epochs, B/4, batch 256, lr 1e-4, Adam (0.9,0.95), EMA 0.9999 | Paper Appendix A/Table 4, printed p.21; all ten effective configs; `meanflow/train.py` |
| Ten completed runs and epoch-80 FIDs | Both run families: `run-summary.json`, `exit-status.txt`, `fid.csv`, `metrics.csv`; 400,320 updates; both controllers |
| Only norm_p varies within a sweep; only training.seed varies between paired runs | Normalized effective configs checked by the script; seed-43 `seed-check-provenance.json` |
| Squared residual sum per example; stopped target/weight; c=0.01 | Archived and local `meanflow/meanflow.py`, forward method; Appendix B.2, printed p.22 |
| Official JAX source | Repository README; manifest commit `d70cb55d298ee03c53bf6da67bec281082e4e2d9` |
| Source used at execution | `provenance/full-run-source.tar.gz`, `full-run-gpu-compat.patch`, `reproducibility-manifest.json`; script verifies hashes and key local files against archive |
| ImageNet: 1,281,167 images / 1,000 classes | Manifest preprocessing/count records; `latent-cache-verification.json` |
| Cache structure and validation scope | Full filename coverage; 100 tensor spot checks, float32 [8,32,32] storing means/stds; not a complete tensor-content audit |
| VAE and posterior sampling | `vae-snapshot.json`, revision `7581b0a0489cc8483c876a728b830b9ce087cf03`; `utils/data_util.py`, `utils/vae_util.py`, `train.py`; scale 0.18215 |
| Evaluation | `fid.csv` and source: EMA, FID-50K every 10 epochs, 1-NFE, fixed sampling seed 0, class labels sampled uniformly |
| FID reference | `fid-reference.json`: supplied author reference, SHA256 `a8e1d8cabbc2cd5137e009dc3c5ad8c410506e9dfcf5d1717a84967108e95d06`; reference moments 2048 and 2048×2048 |
| Resize/Inception implementation | `utils/fid_util.py`, `utils/jax_fid/`: released `legacy_tensorflow` resizing and JAX Inception; FID features on one process with workers=0 |
| One B200 per run; two sweeps of five concurrent runs | `runtime.json`, GPU selection in `command.json`, manifest GPU mapping, paired seed provenance, both controllers |
| Run runtime and throughput | Summary end-to-end seconds, FID CSV timing sums, metrics median at logged step >=500; no smoke extrapolations |

The verified epoch-80 results are below (rounded to two decimals); generated
CSVs preserve full precision and both signed differences.

| p | Paper | Seed 42 | Seed 43 | Mean ± sample SD (n=2) | Paper − our mean |
|---|---:|---:|---:|---:|---:|
| 0.0 | 79.75 | 80.23 | 81.47 | 80.85 ± 0.88 | -1.10 |
| 0.5 | 63.98 | 64.60 | 65.22 | 64.91 ± 0.44 | -0.93 |
| 1.0 | 61.06 | 61.36 | 61.44 | 61.40 ± 0.06 | -0.34 |
| 1.5 | 66.57 | 67.32 | 66.62 | 66.97 ± 0.49 | -0.40 |
| 2.0 | 69.19 | 97.61 | 97.95 | 97.78 ± 0.24 | -28.59 |

Both seeds rank p=1 best. For p<=1.5, individual differences from the paper range
from +0.052666 to +1.721977. For p=2, they are +28.418740 and +28.759139: the large
discrepancy persists under the second training seed. Every recorded curve
strictly decreases, and p=2 has the highest FID at every checkpoint in each seed.
This supports partial numerical replication, without identifying the discrepancy's
cause. Two seeds provide limited descriptive evidence and do not establish
robustness, significance or outlier status. Seed 42 matches the released default
in `meanflow/configs/default.py` and `configs/run_b4.yml`; the paper does not
specify the historical Table 1(e) training seed. The report therefore does not
claim that either run reproduces the authors' exact historical random seed.

The discussion distinguishes the observed p=2 gap (mean 28.59 above the paper)
from hypotheses. At p=2, the implemented weight `(q + 0.01)**(-2)` favors small
residuals more strongly than at p=1. Stabilizer/trajectory sensitivity, the
GPU/software port, and unknown historical file order are possible contributors;
none was isolated by an ablation. The paper's illustrative c=1e-3 does not
establish that its historical Table 1(e) runs used a different c from our 0.01.

| Cost scope | Seed 42 | Seed 43 | Both sweeps |
|---|---:|---:|---:|
| Summed one-GPU run hours | 31.610764 | 31.417578 | 63.028342 |
| Per-run hours | 6.319280–6.324975 | 6.280888–6.286125 | 6.280888–6.324975 |
| Parallel sweep elapsed hours | 6.326310 | 6.288893 | 12.615203 (sum of active sweep intervals) |
| Five-GPU occupied-block estimate | 31.631548 | 31.444466 | 63.076014 |
| Per-run total FID evaluation minutes (eight evaluations) | 21.9245–25.0688 | 21.1008–22.9061 | 21.1008–25.0688 |
| Range of per-run median updates/s (logged step >=500) | 19.0282–19.0680 | 18.9720–18.9839 | 18.9720–19.0680 |

Seed-42 parallel elapsed time is the recorded monotonic controller duration.
Seed 43 has no recorded controller start/duration; its elapsed time is the UTC
interval from the earliest `command.json` launch (2026-10-05T16:38:18.006521+00:00)
to `seed43-all-controller-summary.json` completion
(2026-10-05T22:55:38.022393+00:00). The five-GPU occupied-block figures estimate
five GPUs retained for each sweep interval; they are not invoices. Summed run
GPU-hours and parallel elapsed walltime have different scopes.

Run time includes compilation, I/O, evaluations, previews, checkpointing, logging
and final hashing. It excludes preprocessing, the smoke check and idle allocation
time. Allocated/billed duration is unavailable; retained idle GPUs or time between
sweeps may have been billed. These figures are not a controlled GPU/TPU speed
comparison. Throughput is the range of per-run medians from the recorded training
metrics, cross-checked against the summaries, rather than an end-to-end average.
Full precision and timing scopes are in `analysis_summary.json` and `run_costs.csv`.

## Environment and compatibility differences

The official repository is described as written/tested on TPUs; Appendix B.4
benchmarks B/4 on v4-8 TPUs, which does not establish the hardware of every
historical Table 1 run. The released installation script pins JAX/JAXlib 0.4.27,
TensorFlow 2.15.0 and Orbax 0.4.4 and permits Flax >=0.8. The executed environment
records Python 3.12.3, JAX/JAXlib 0.11.2, Flax 0.12.10, Optax 0.2.8,
Diffusers 0.35.1, SciPy 1.18.1, NumPy 2.5.3, TensorFlow CPU 2.21.0,
Torch 2.14.1+cpu, Torchvision 0.29.1+cpu and Orbax 0.12.6. The manifest records
NVIDIA B200 GPUs and driver 610.57.04. Model parameters are float32.

The recorded patch bypasses distributed startup for one process, updates JAX
tree/clip/cost-analysis APIs and SciPy sqrtm's return API, defers unused legacy
Keras imports, loads a pinned local VAE, uses spawn workers for preprocessing
and latent loading, sets FID feature workers to zero, protects nonempty caches,
freezes latent filename order, and fixes a preview-grid size. The separate IPC
record moves temporary sockets to local `/tmp`. The host recorder wraps the
authors' entry point, writers, checkpoint and FID functions and adds timing/file
output, without adding training updates.

The single-process sampler stays at epoch 0 because released `set_epoch` is
conditional on multiple processes. Its default data-order seed is independent
of `training.seed`. The accepted `use_flip` argument does not actually flip
latents. Those released behaviors are preserved, but the frozen filesystem
order cannot be equated with the authors' historical order. The paper's c=1e-3
is illustrative; the exact historical ablation stabilizer and training seeds
are unspecified. The differences prevent a claim of bitwise historical
replication. Close endpoints for p<=1.5 and the best p=1 in both seeds support partial
numerical replication; p=2's persistent discrepancy has no demonstrated cause.

## Layout and submission framing

`main.tex` retains the usual introduction/experiment/discussion/conclusion order.
The complete loss-metric experiment occupies page 2 with its equation, two-seed
table and recorded learning-curve figure. The
ISBI style file, fonts, margins and line spacing are unchanged.

The shared abstract, introduction, discussion and conclusion are concise enough
to fit one rendered page together with the front matter and contribution statement,
verified in `../build/shared-budget/shared_layout_check.pdf` (shared text, including
Simon's limitations subsection, on page 1;
references on page 2). That is a separate layout check, not an appended report
section. In `../build/main.pdf`, shared material remains before and after the
experiment; the current report has three pages including references. Hjalte's and
Djafar's one-page experiment allocations have source TODO comments and no invented
placeholder results or empty pages. Check pagination again when their sections
and shared findings are integrated.

Group 11 and the assignments were confirmed by Simon: Hjalte Table 1(b), Simon
Table 1(e), Djafar Table 1(a). The front page preserves the existing author names,
states Replication Track and Group 11, links the actual assignment repository
`https://github.com/simonwinther/ATDL_GROUP_ASSIGNMENT_2` from the report's origin
remote, and gives specific contributions. The authors' `Gsunshine/meanflow`
repository remains a separate reference.

The completed experiment is ImageNet-256/JAX. The abstract reports five exponents,
ten runs, n=2 sample SD, p=1 at 61.40 +/- 0.06 and p=2 at 97.78 +/- 0.24 versus
69.19, and explicitly limits these findings to the loss-metric study. The earlier
CIFAR-10/PyTorch-only scope is not supported by any member protocol or experiment
file in this checkout and was removed. No dataset or implementation is assumed
for the other members; the confirmed ablation targets are named separately.

Required remaining information is Hjalte's and Djafar's actual experiment sections,
datasets, implementation repositories/configurations and numerical results, with
their interpretation for the shared sections. Source TODO comments identify these
integration points. Their confirmed experiment contributions are credited without
inventing additional implementation or writing roles. There are no group-number,
track, repository or contribution placeholders in the rendered front matter.
The supplied template README identifies ISBI 2021 while the assignment names
ISBI 2027; the existing template settings have been preserved as instructed.

### Simon's limitations subsection

`../sections/loss_metrics_limitations.tex` contributes three paragraphs to the
shared discussion: FID's limited diagnostic value, the scope of model comparisons,
and the two-dataset coverage. The existing p=2 discussion is shorter to avoid
repeating details already reported on the experiment page. All experimental
results, the full experiment section, tables and figures remain unchanged.

The MeanFlow paper's Section 5 and Tables 2–3 use FID-50K for quantitative
generation quality on ImageNet-256 and CIFAR-10. Table 2 includes Shortcut,
IMM, DiT and SiT, with different NFE and model settings; a standalone baseline
explicitly labeled Rectified Flow is absent. The subsection does not suggest
that the paper omits flow-based methods or compares only weak models. Its model
rankings are conditional on the reported settings, and Simon's B/4 ablation
does not reproduce the larger-model comparisons.

FID's inability to separate fidelity from coverage is supported by Kynkäänniemi
et al., *Improved Precision and Recall Metric for Assessing Generative Models*
(NeurIPS 2019, Sections 1–1.1). Zhai et al., *A Large-scale Study of Representation
Learning with the Visual Task Adaptation Benchmark* (arXiv:1910.04867v2,
Sections 2–3) defines VTAB's 19 tasks and the 1,000-label protocol. VTAB-1k is
used as an example of domain breadth, not as a direct generative-quality test.
Both sources are cited in `../refs.bib`.

Performance outside the evaluated datasets and under controlled background or
noise changes remains untested. No claim of deliberate cherry-picking, hidden
experiments or demonstrated background shortcuts is made; the available results
do not establish those explanations.

## Recorded two-seed figures

`analyse_loss_metrics.py` generates Figure 1 using the established paper-style
renderer in `paper_seed_figure.py`. Both seeds share one linear FID axis with
range 55–160. Colour and marker shape identify p; solid lines and larger filled
markers identify seed 42, dashed lines and smaller open markers identify seed 43.
The measured paired curves largely overlap. A horizontal p key sits below the
plot, with a compact seed key in the unused upper-right corner. The single-column
canvas uses 10pt serif labels, vector lines and markers, and embedded fonts.
No synthetic data or preview page is included in the report.

The endpoint alternative `../figures/loss_metrics_fid_vs_p.{pdf,png}` plots both
seeds and the paper's five Table 1(e) values. The paper has no intermediate epoch
curves in this table, so no paper learning curve is invented.

`paper_seed_figure.py` now defaults to recorded mode and refuses incomplete
seed-43 endpoints. It can also save a separate recorded figure/provenance copy
under `visualisation-previews/RECORDED_SEEDS/paper_figures/`. Historical synthetic
design assets remain isolated under `SYNTHETIC_SEED43/`; they are neither read
by the analysis nor included by LaTeX. Synthetic mode requires an explicit
`--mode synthetic`. The optional log-axis comparisons and stacked 2D alternative
remain outside the report. No 3D assets are used.
