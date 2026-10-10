# Table 1(e) analysis

Run these commands from the repository root:

```sh
python -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r experiments/simon_table1e/analysis/requirements.txt
.venv-analysis/bin/python experiments/simon_table1e/analysis/analyse_loss_metrics.py
```

The recorded analysis environment used Python 3.14.7, NumPy 2.5.2 and
Matplotlib 3.11.1. Analysis runs on CPU using the included records.

The script reads experiments/simon_table1e/ and writes results/simon_table1e/.
Paths are relative to the script. --data-root and --output-root let you use
other locations; outputs must be outside the experiment records.

All ten runs must pass before results are written. Checks cover successful
exits, epoch 80, step 400320, EMA FID-50K with 1 NFE, sampling seed 0, matching
FID values in CSVs and summaries, config controls, runtimes, GPU/software
records, controller exits and source hashes. Configs may differ only in
norm_p and training.seed. The smoke test is excluded. This check does not
need checkpoints, models or datasets. The original seed-43 collector also
checks checkpoint files.

processed/table1e_two_seeds.csv contains both seed FIDs, their mean and
sample SD (ddof=1, n=2). The individual and mean difference columns use
ours minus paper. paper_minus_mean uses paper minus our mean, as in the
report table. The other CSVs contain individual results, all 80 recorded FID
checkpoints and run costs. analysis_summary.json records the checks and
input hashes. GPU-hours add each run's measured duration. Parallel elapsed
time comes from the controller records. Allocated/billed time is unknown.

figures/loss_metrics_fid_epochs.{pdf,png} shows both seeds' learning curves.
figures/loss_metrics_fid_vs_p.{pdf,png} shows both final FIDs and the paper's
values. PDF figures are vector files; PNGs are previews. paper_seed_figure.py
sets the curve figure's layout. Plots use only recorded FID points. Adaptive
training-loss values have different scales across p and are not compared.
