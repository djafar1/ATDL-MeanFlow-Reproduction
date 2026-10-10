# ATDL Group Assignment: MeanFlow Reproduction

Reproducibility project for the course Advanced Topics in Deep Learning at the University of Copenhagen.

## Group members

- Said Djafar Said
- Simon Winther Albertsen
- Hjalte Svaneborg Bjørnstrup

## Paper

**Mean Flows for One-step Generative Modeling**
Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He

Paper: https://arxiv.org/abs/2505.13447
Official code: https://github.com/Gsunshine/meanflow

## Group experiments

Group 11 follows the Replication Track. Each experiment has its own guide,
configurations, launchers, and result records:

| Experiment | Contributor | Guide / code | Results |
| --- | --- | --- | --- |
| Table 1(a): unequal-time sampling ratio | Said Djafar Said | `experiments/said_table1a/` | `results/said_table1a/` |
| Table 1(b): JVP tangent computation | Hjalte Svaneborg Bjørnstrup | [Run guide](experiments/table_b/README.md), `configs/table_b/`, `scripts/hendrix/table_b*` | `results/runs/table-b-first/` (synced separately) |
| Table 1(e): adaptive loss metrics | Simon Winther Albertsen | [Run guide](experiments/simon_table1e/README.md) | `results/simon_table1e/`, compact records in `experiments/simon_table1e/runs/` |

Table 1(e) includes two training seeds per loss exponent. Table 1(b) includes
one training seed per tangent. Use each experiment's own protocol and dependency
versions rather than treating the three experiments as one shared run setup.

The shared `src/meanflow/` now includes the group's GPU compatibility patches
and Table 1(a) configs. The completed Table 1(b) jobs used the original upstream
code plus our adapter, preserved in their frozen source snapshot. Future Table
1(b) submissions from this checkout snapshot the current shared source.

## Completed experiment: Table 1(b)

We trained four JVP tangent settings to 400,320 steps (80 epochs) and evaluated
each EMA model with 50,000 generated images at one sampling step.

| Paper tangent `(v,r,t)` | Paper FID | Our FID |
| --- | ---: | ---: |
| `(v,0,1)` | 61.06 | 60.91 |
| `(v,0,0)` | 268.06 | 260.45 |
| `(v,1,0)` | 329.22 | 259.70 |
| `(v,1,1)` | 137.96 | 119.77 |

The baseline closely matches the paper. The other rows are a partial numerical
reproduction; the `(v,1,0)` row differs substantially. These are single-seed
results, so we cannot attribute the differences to seed variation alone.
The full-precision output is [table-b.csv](results/runs/table-b-first/table-b.csv).
Generated results are ignored by Git and must be synced separately.

## Reading order

1. [Experiment guide](experiments/table_b/README.md): recipe, tangent mapping,
   and commands to prepare data, train, resume, and evaluate.
2. [Four configurations](configs/table_b/): identical settings apart from the
   two JVP time tangents; start with `v01.yml` for the baseline.
3. [Model adapter](scripts/hendrix/table_b_model.py): the experiment's change
   to the upstream loss. Compare it with [MeanFlow](src/meanflow/meanflow.py).
4. [Training and evaluation](scripts/hendrix/table_b.py): update loop,
   checkpoint/resume, EMA sampling, and FID evaluation.
5. [Result collector](scripts/hendrix/table_b_results.py): checks that all four
   evaluations are comparable before producing the CSV.

## Supporting files

| Path | Purpose |
| --- | --- |
| `scripts/hendrix/submit_table_b.py` | Preview/submit jobs and freeze a source snapshot |
| `scripts/hendrix/table_b.sbatch`, `table_b_eval.sbatch` | Slurm training and evaluation launchers |
| `scripts/hendrix/table_b_common.py`, `runtime.py` | Run metadata, checkpoints, configuration, and JAX setup |
| `scripts/hendrix/imagenet_download.sbatch`, `imagenet_latents.sbatch`, `check_imagenet.py` | Download, encode, and validate the shared dataset |
| [Environment setup](environment/README.md) | Cluster dependencies and local editor environment |
| `tests/test_table_b.py` | Configuration, launcher, collector, and optional CPU integration checks |
| `src/meanflow/` | Shared released code with the group's GPU compatibility patches and original documentation |

## Evidence from the completed runs

`results/runs/table-b-first/` contains the four run directories (`v01`, `v00`,
`v10`, `v11`), resolved configs, dataset metadata, metrics, summaries, evaluation
JSON files, sample grids, and Slurm logs. Its `source/` directory is the frozen
source used by those jobs. Read that snapshot when checking historical behavior;
it intentionally still contains older helper files that were present at submission.
Checkpoints excluded during download remain on Hendrix.

`results/runs/hendrix-5602/` and `cpu-smoke-validation-20261003/` are earlier
validation artifacts, not inputs to the four reported FIDs. They are retained
as validation evidence.

## Assignment report and template

The report uses the ISBI/IEEE template. Integrate all three experiment sections
and update the abstract, discussion, and conclusion from their recorded results.
The repository's experiment guides document execution; they do not establish
which results have already been included in the current report.

Template guidance: [IEEE Author Center](https://ieeeauthorcenter.ieee.org/publish-with-ieee/author-education-resources/guidelines-and-policies/policy-posting-your-article/)
and [ISBI 2021](https://biomedicalimaging.org/2021/full-paper-instructions/).
Template material is subject to the [LaTeX Project Public License](https://www.latex-project.org/lppl/);
the authors' code retains its [upstream license](src/meanflow/LICENSE).
