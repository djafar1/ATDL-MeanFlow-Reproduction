"""One launch, five independent processes; collect results when all finish."""
from pathlib import Path
import datetime as dt
import json
import subprocess
import sys
import time

ROOT = Path('/work/meanflow-table1e')
BUNDLE = Path(__file__).resolve().parent
MAPPING = [(0, '1.0'), (1, '0.0'), (2, '0.5'), (3, '1.5'), (4, '2.0')]


def main():
    print('Running preflight before launching any full run...', flush=True)
    subprocess.run([sys.executable, str(BUNDLE / 'preflight.py')], check=True)
    # Exclusive mkdir prevents concurrent controllers and any automatic re-run.
    (ROOT / 'provenance/full-launch.lock').mkdir()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    start_clock = time.monotonic()
    processes = []
    try:
        for gpu, p in MAPPING:
            out = ROOT / f'runs/p_{p}'
            log = (out / 'launcher.log').open('w')
            process = subprocess.Popen(['zsh', str(BUNDLE / 'run_arm.zsh'), str(gpu), p],
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT)
            log.close()
            (out / 'launcher-pid.txt').write_text(str(process.pid) + '\n')
            processes.append((p, process))
            print(f'LAUNCHED p={p} on physical GPU {gpu}; launcher PID {process.pid}', flush=True)
        completed = {}
        first_windows = set()
        while len(completed) < len(processes):
            for p, process in processes:
                if p in completed:
                    continue
                code = process.poll()
                if code is not None:
                    completed[p] = code
                    print(f'FINISHED p={p}, exit={code}', flush=True)
                    continue
                progress = ROOT / f'runs/p_{p}/progress.json'
                if progress.exists():
                    row = json.loads(progress.read_text())
                    if row.get('kind') == 'train' and row.get('train_step', 0) >= 100:
                        if p not in first_windows:
                            first_windows.add(p)
                            print(f'FIRST METRICS p={p}: step={row["train_step"]}, loss={row.get("loss")}, steps/sec={row.get("steps_per_second")}', flush=True)
            if len(first_windows) == 5:
                print('ALL FIVE RUNS HAVE REACHED AT LEAST 100 STEPS WITH FINITE LOGGED METRICS.', flush=True)
                first_windows.add('announced')
            if len(completed) < len(processes):
                time.sleep(15)
    finally:
        (ROOT / 'provenance/parallel-run-timing.json').write_text(json.dumps({
            'started_utc': started, 'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'controller_wall_seconds': time.monotonic() - start_clock,
            'launcher_exit_codes': {p: process.poll() for p, process in processes},
            'five_gpu_block_hours': 5 * (time.monotonic() - start_clock) / 3600,
            'billing_note': 'Five-GPU occupied block estimate; UCloud allocation billing may include idle time before/after.'
        }, indent=2) + '\n')
    subprocess.run([sys.executable, str(BUNDLE / 'collect_results.py')], check=True)
    print('Results written to /work/meanflow-table1e/results/table1e.csv', flush=True)
    if any(process.returncode for _, process in processes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
