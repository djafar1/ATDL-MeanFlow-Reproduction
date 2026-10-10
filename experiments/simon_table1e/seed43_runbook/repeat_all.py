"""Repeat all five MeanFlow loss exponents with training seed 43, once each.

The completed seed-42 runs and their results are read-only inputs.
Use --launch to check then launch five fresh arms; --status reads progress;
--collect writes a separate comparison with both seeds and descriptive mean/SD.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

import run_seed43 as arm

ROOT = arm.ROOT
PLAN = ROOT / 'provenance/seed43-all-plan.json'
LOCK = ROOT / 'provenance/seed43-all-launch.lock'
RESULTS = ROOT / 'results/seed43'
PAPER = {'0.0': 79.75, '0.5': 63.98, '1.0': 61.06, '1.5': 66.57, '2.0': 69.19}


def verify_bundle(plan):
    for name, expected in plan['launcher_bundle_sha256'].items():
        arm.require(arm.checksum(Path(__file__).parent / name) == expected,
                    f'Seed-43 launcher changed after preflight: {name}')


def make_plan():
    # Check every destination first, before any GPU work or training.
    for p in arm.P_VALUES:
        arm.configure(p)
        arm.require(not arm.WORK.exists() and not arm.CONFIG_PATH.exists(),
                    f'Seed-43 output/config already exists for p={p}; refusing to overwrite or duplicate.')
    arm.require(not PLAN.exists() and not RESULTS.exists(),
                'A seed-43 plan/results directory already exists. Inspect it; do not relaunch.')
    entries = {}
    for p in arm.P_VALUES:
        arm.configure(p)
        print(f'Checking p={p}, seed=43 before launching any arm...', flush=True)
        config, record, freeze = arm.preflight()
        entries[p] = {'config': config, 'record': record, 'packages_freeze': freeze}
    return {
        'created_utc': arm.utc(), 'training_seeds': [42, 43], 'sampling_seed': 0,
        'p_values': arm.P_VALUES, 'arms': entries,
        'launcher_bundle_sha256': {name: arm.checksum(Path(__file__).parent / name)
                                  for name in ['repeat_all.py', 'run_seed43.py']},
        'only_experimental_change_per_arm': 'training.seed: 42 -> 43',
        'selection': 'Second sweep chosen after observing the original five results; all five p values repeated, no original scores replaced.',
        'output_directories': [str(ROOT / f'runs/diagnostic_p_{p}_seed43') for p in arm.P_VALUES],
        'reporting': 'Report both seeds and descriptive mean +/- sample standard deviation for n=2 completed training runs per p. Evaluation seed remains 0; this is not an FID sampling-seed study.',
    }


def collect():
    run_rows = []
    comparison = []
    complete_new = 0
    for p, target in PAPER.items():
        values = {}
        for seed in [42, 43]:
            work = ROOT / (f'runs/p_{p}' if seed == 42 else f'runs/diagnostic_p_{p}_seed43')
            row = {'p': float(p), 'training_seed': seed, 'sampling_seed': 0,
                   'status': 'not_completed', 'FID_epoch80': None,
                   'end_to_end_hours': None, 'one_gpu_run_GPU_hours': None,
                   'median_logged_steps_per_second': None,
                   'work_directory': str(work), 'error': None}
            try:
                summary, cfg = arm.completed_record(work, seed)
                arm.require(summary['p'] == float(p) and cfg['method']['norm_p'] == float(p),
                            f'Wrong p in {work}')
                if seed == 43:
                    expected = arm.read_json(work / 'seed-check-provenance.json')['expected_effective_config']
                    arm.require(cfg == expected, f'Unexpected seed-43 effective config: {p}')
                    original = arm.read_json(ROOT / f'runs/p_{p}/effective-config.json')
                    original['training']['seed'] = 43
                    arm.require(cfg == original, f'Config differs beyond training.seed: {p}')
                fid = summary['final_FID_epoch80']
                arm.require(math.isfinite(fid), f'Invalid final FID: {work}')
                values[seed] = fid
                row.update(status='completed', FID_epoch80=fid,
                           end_to_end_hours=summary['end_to_end_wall_seconds'] / 3600,
                           one_gpu_run_GPU_hours=summary.get('gpu_hours_for_this_one_gpu_run',
                                                            summary['end_to_end_wall_seconds'] / 3600),
                           median_logged_steps_per_second=summary['median_logged_steps_per_second_after_step500'])
                if seed == 43:
                    complete_new += 1
            except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
                row['error'] = str(error)
            run_rows.append(row)
        both = len(values) == 2
        mean = statistics.mean(values.values()) if both else None
        comparison.append({
            'p': float(p), 'paper_FID': target,
            'FID_training_seed42': values.get(42), 'FID_training_seed43': values.get(43),
            'n_completed_training_seeds': len(values),
            'mean_FID_two_seeds': mean,
            'sample_SD_FID_two_seeds_ddof1': statistics.stdev(values.values()) if both else None,
            'mean_FID_minus_paper': mean - target if both else None,
            'seed43_minus_seed42': values[43] - values[42] if both else None,
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, rows in [('individual-runs.csv', run_rows), ('table1e-two-seeds.csv', comparison)]:
        with (RESULTS / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    arm.write_json(RESULTS / 'report-summary.json', {
        'collected_utc': arm.utc(), 'completed_seed43_runs': complete_new,
        'training_seeds': [42, 43], 'sampling_seed': 0,
        'endpoint': 'Epoch-80 EMA FID-50K, ImageNet-256, 1 NFE, no CFG.',
        'individual_runs': run_rows, 'comparison': comparison,
        'variability_definition': 'Sample standard deviation across the two training-seed FIDs (ddof=1), not standard error or a confidence interval; FID sampling seed is fixed.',
        'limitations': 'Two training seeds give only a limited descriptive variability check. They do not establish unbiasedness, statistical stability or whether a score is an outlier. The second sweep was selected after observing the seed-42 results.',
        'original_results': str(ROOT / 'results/table1e.csv'),
        'billing_scope': 'Measured one-GPU run hours do not include allocated GPUs sitting idle before/after the runs.',
    })
    print(f'Collected {complete_new}/5 completed seed-43 runs. Separate results: {RESULTS}', flush=True)
    return complete_new


def launch_all():
    arm.require(Path(sys.prefix).resolve() == (ROOT / 'venv').resolve(), 'Use venv/bin/python.')
    LOCK.mkdir(exist_ok=False)
    (LOCK / 'controller-pid.txt').write_text(str(os.getpid()) + '\n')
    launched = []
    prepared = False
    try:
        plan = make_plan()
        arm.write_json(PLAN, plan)
        prepared = True
        verify_bundle(plan)
        for p in arm.P_VALUES:
            log_path = ROOT / f'provenance/seed43-p_{p}-launcher.log'
            with log_path.open('x') as log:
                child = subprocess.Popen(
                    [sys.executable, '-u', str(Path(__file__).resolve()), '--arm', p],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            launched.append((p, child))
            print(f'LAUNCHED seed43 p={p}, launcher PID={child.pid}', flush=True)
        finished = {}
        while len(finished) < len(launched):
            for p, child in launched:
                code = child.poll()
                if code is not None and p not in finished:
                    finished[p] = code
                    print(f'FINISHED seed43 p={p}, launcher exit={code}', flush=True)
            if len(finished) < len(launched):
                time.sleep(10)
        count = collect()
        # Both the original table and original individual records must be unchanged.
        for name, expected in next(iter(plan['arms'].values()))['record']['primary_files_to_preserve_sha256'].items():
            arm.require(arm.checksum(Path(name)) == expected, f'Primary record changed: {name}')
        success = count == 5 and all(value == 0 for value in finished.values())
        arm.write_json(ROOT / 'provenance/seed43-all-controller-summary.json', {
            'finished_utc': arm.utc(), 'success': success, 'launcher_exit_codes': finished,
            'completed_seed43_runs': count, 'original_primary_files_preserved': True,
        })
        return 0 if success else 1
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        # Do not kill healthy training arms if a controller fails. Their launchers
        # and recording continue independently; --status/--collect remain usable.
        if launched:
            print('Already-launched arms continue. Inspect --status and individual logs; do not relaunch.', flush=True)
        if not prepared:
            (LOCK / 'controller-pid.txt').unlink()
            LOCK.rmdir()
            print('Preflight stopped before training; no seed-43 arms were launched.', flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--launch', action='store_true')
    actions.add_argument('--status', action='store_true')
    actions.add_argument('--collect', action='store_true')
    actions.add_argument('--arm', choices=arm.P_VALUES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        for p in arm.P_VALUES:
            arm.configure(p)
            arm.status()
        return 0
    if args.collect:
        collect()
        return 0
    if args.arm:
        plan = arm.read_json(PLAN)
        verify_bundle(plan)
        arm.configure(args.arm)
        entry = plan['arms'][args.arm]
        arm.require(entry['record']['p'] == float(args.arm), 'Wrong p in controller plan.')
        return arm.launch((entry['config'], entry['record'], entry['packages_freeze']))
    return launch_all()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
