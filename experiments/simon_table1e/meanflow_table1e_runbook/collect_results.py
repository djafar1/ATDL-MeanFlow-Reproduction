"""Read recorded files only; never train or evaluate. Epoch-80 FID is primary."""
from pathlib import Path
import csv
import datetime as dt
import json
import math

ROOT = Path('/work/meanflow-table1e')
PAPER = {'0.0': 79.75, '0.5': 63.98, '1.0': 61.06, '1.5': 66.57, '2.0': 69.19}


def telemetry(path):
    values = {'peak_vram_mib': [], 'peak_gpu_util_percent': []}
    if not path.exists():
        return {key: None for key in values}
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            for prefix, key in [('memory.used', 'peak_vram_mib'),
                                ('utilization.gpu', 'peak_gpu_util_percent')]:
                for column, value in row.items():
                    if column and column.startswith(prefix) and value:
                        try:
                            values[key].append(float(value.strip().split()[0]))
                        except (ValueError, AttributeError):
                            pass
    return {key: max(items) if items else None for key, items in values.items()}


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    rows, all_fids = [], []
    for p, target in PAPER.items():
        work = ROOT / f'runs/p_{p}'
        path = work / 'run-summary.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        exit_file = work / 'exit-status.txt'
        final = data.get('FID_evaluations', [])[-1:] or [{}]
        fid = final[0]
        valid = (data.get('status') == 'completed' and
                 data.get('p') == float(p) and data.get('final_train_step') == 400320 and
                 exit_file.exists() and exit_file.read_text().strip() == '0' and
                 fid.get('epoch') == 80 and fid.get('train_step') == 400320 and
                 fid.get('num_samples') == 50000 and fid.get('NFE') == 1 and
                 fid.get('EMA') is True and math.isfinite(fid.get('FID_ema', float('nan'))))
        ours = fid['FID_ema'] if valid else None
        wall = data.get('end_to_end_wall_seconds')
        row = {
            'p': float(p), 'status': 'completed' if valid else 'incomplete_or_failed',
            'paper_FID': target, 'replicated_FID_epoch80': ours,
            'FID_delta_ours_minus_paper': ours - target if valid else None,
            'epochs_completed': data.get('epochs_completed'),
            'final_train_step': data.get('final_train_step'),
            'end_to_end_hours': wall / 3600 if wall is not None else None,
            'one_gpu_run_GPU_hours': data.get('gpu_hours_for_this_one_gpu_run'),
            'FID_evaluation_hours': data.get('total_FID_evaluation_seconds', 0) / 3600,
            'checkpoint_save_seconds': data.get('total_checkpoint_save_seconds'),
            'median_logged_steps_per_second_after_step500': data.get('median_logged_steps_per_second_after_step500'),
            **telemetry(work / 'gpu.csv'), 'rank': None,
        }
        rows.append(row)
        for item in data.get('FID_evaluations', []):
            all_fids.append({'p': float(p), **item})
    ranked = sorted((r for r in rows if r['status'] == 'completed'),
                    key=lambda r: r['replicated_FID_epoch80'])
    for rank, row in enumerate(ranked, start=1):
        row['rank'] = rank
    with (out / 'table1e.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'fid-learning-curves.csv').open('w', newline='') as stream:
        fields = ['p', 'epoch', 'train_step', 'num_samples', 'sampling_seed', 'NFE', 'EMA',
                  'FID_ema', 'generation_seconds', 'inception_and_stats_seconds',
                  'fid_math_seconds', 'evaluation_seconds', 'stats_file', 'stats_sha256',
                  'utc', 'wall_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_fids)
    timing = ROOT / 'provenance/parallel-run-timing.json'
    report = {
        'collected_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'primary_endpoint': 'EMA FID-50K at epoch80, 1NFE, no CFG',
        'completed_runs': len(ranked), 'rows': rows,
        'sum_measured_one_gpu_run_hours': sum(r['one_gpu_run_GPU_hours'] or 0 for r in rows),
        'parallel_timing': json.loads(timing.read_text()) if timing.exists() else None,
        'interpretation': 'One seed per arm. Compare p ranking and signed FID differences; no training variance/error bars can be estimated.',
        'runtime_scope': 'Python entrypoint startup through final checkpoint hashing, includes data I/O, compilation, previews, eight FID evaluations, checkpointing and logging.',
        'throughput_scope': 'Median of authors logged 100-step windows after step500; not a hardware-normalized benchmark. Native epoch-boundary windows may be partial.',
        'memory_scope': 'Peak whole-process GPU memory sampled every10s; may include VAE/Inception compilation workspaces, and can miss shorter peaks.',
        'billing_scope': 'Run GPU-hours and five-GPU occupied block estimates are not UCloud billing totals.',
    }
    (out / 'report-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Collected {len(ranked)}/5 completed epoch-80 FID-50K results. Missing runs remain explicitly marked.', flush=True)


if __name__ == '__main__':
    main()
