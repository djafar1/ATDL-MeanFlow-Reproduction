"""Read-only progress; safe to run any time after launch."""
from pathlib import Path
import json
import os

ROOT = Path('/work/meanflow-table1e')


def main():
    ready = 0
    for p in ['0.0', '0.5', '1.0', '1.5', '2.0']:
        work = ROOT / f'runs/p_{p}'
        progress = work / 'progress.json'
        row = json.loads(progress.read_text()) if progress.exists() else {}
        exit_file = work / 'exit-status.txt'
        state = 'not started'
        pid_file = work / 'python-pid.txt'
        if pid_file.exists():
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, 0)
                state = 'running'
            except ProcessLookupError:
                state = 'process ended; inspect summary'
        if exit_file.exists():
            state = 'finished exit=' + exit_file.read_text().strip()
        step = row.get('train_step', 0)
        if state == 'running' and step >= 100:
            ready += 1
        print(f'p={p}  {state:30}  step={step:6}  epoch={row.get("epoch", "-")}  '
              f'loss={row.get("loss", "-")}  sps={row.get("steps_per_second", "-")}  '
              f'FID={row.get("FID_ema", "-")}')
    print(f'{ready}/5 runs are running and have logged at least100 steps.')
    print('Logs: /work/meanflow-table1e/runs/p_*/training.log')


if __name__ == '__main__':
    main()
