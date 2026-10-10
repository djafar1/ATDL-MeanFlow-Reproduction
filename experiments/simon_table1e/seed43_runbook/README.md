# MeanFlow Table 1(e): complete second training-seed sweep

This adds one fresh run for each p = 0, 0.5, 1, 1.5, 2 using training seed 43.
The five original seed-42 runs are preserved. Compared with the completed run
for each p, the only experimental change is training.seed 42 -> 43.
Sampling seed stays 0. Each p uses the same physical B200 as its first run.

All runs retain DiT-B/4, ImageNet-256 cached latents, 80 epochs from scratch,
global/per-device batch 256, norm_eps 0.01, no CFG, the same VAE and FID reference,
EMA FID-50K and 1-NFE sampling. No dependency or authors-source changes are made.
The recorder is the original audit_run.py; a separate entrypoint checks the
loaded effective config before delegating to it. Logs, checkpoints, FID CSVs,
Inception statistics, GPU telemetry and runtime summaries use the same recorder.

## Upload from your own computer

```sh
scp -P 2239 ~/Downloads/meanflow_seed43_all.zip \
  ucloud@ssh.cloud.sdu.dk:/work/meanflow-table1e/
```

## Launch on UCloud

```sh
cd /work/meanflow-table1e
unzip -n meanflow_seed43_all.zip
nohup venv/bin/python -u seed43_runbook/repeat_all.py --launch \
  >> provenance/seed43-all-controller.log 2>&1 < /dev/null &
```

Do not also launch the earlier p=2-only script. This launcher refuses existing
seed-43 output/config directories and duplicate controller launches. It checks
all five arms before launching any of them; preflight failures launch none.
It makes no changes to your interactive shell options or tmux configuration.

After a few minutes:

```sh
venv/bin/python seed43_runbook/repeat_all.py --status
tail -n 30 provenance/seed43-all-controller.log
nvidia-smi
```

Check that all five arms have finite logged training metrics and roughly the
previous throughput before leaving them unattended. Training runs independently
of the SSH session. About 6.3 hours is an estimate based on the first full sweep,
not a time limit. Five parallel one-GPU runs cost about 31.6 additional measured
run GPU-hours; the allocation can also bill idle time.

## Separate outputs

- runs/diagnostic_p_0.0_seed43
- runs/diagnostic_p_0.5_seed43
- runs/diagnostic_p_1.0_seed43
- runs/diagnostic_p_1.5_seed43
- runs/diagnostic_p_2.0_seed43

Each has training.log, metrics.csv, fid.csv, effective-config.json,
seed-check-provenance.json, command.json, runtime.json, gpu.csv,
run-summary.json and the native checkpoints, including checkpoint_400320.
The authors' checkpoint retention policy is unchanged.

## Results and reporting

At completion, the controller automatically writes results/seed43/:

- individual-runs.csv: all ten individual runs, their seeds, FIDs and runtimes.
- table1e-two-seeds.csv: paper FID, seed-42 FID, seed-43 FID, n completed,
  two-seed mean, sample SD (ddof=1), signed difference from the paper.
- report-summary.json: endpoint, variability definition, completion and limitations.

Original results/table1e.csv is never overwritten. To recollect from files only:

```sh
venv/bin/python seed43_runbook/repeat_all.py --collect
cat results/seed43/table1e-two-seeds.csv
```

Report n=2 training seeds per p only after both runs completed. Keep both
individual FIDs visible. Sample SD summarizes training-seed variation under
the fixed data/cache/environment and FID sampling seed; it is not a confidence
interval. Two seeds provide a limited variability check, not proof of stability
or unbiasedness. Disclose that the second sweep was chosen after inspecting the
seed-42 results. Use epoch-80 FID rather than the best intermediate FID or seed.
