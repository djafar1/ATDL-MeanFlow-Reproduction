"""Internal per-arm launcher for the complete seed-43 ablation.

Use repeat_all.py to launch, inspect or collect the complete second sweep.
This module checks and records each arm using the original training recorder.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = Path('/work/meanflow-table1e')
REPO = ROOT / 'meanflow'
WORK = ROOT / 'runs/diagnostic_p_2.0_seed43'
BUNDLE = ROOT / 'meanflow_table1e_runbook'
CONFIG_NAME = 'table1e_p_2.0_seed43'
CONFIG_PATH = REPO / f'configs/{CONFIG_NAME}.yml'
GIT_SHA = 'd70cb55d298ee03c53bf6da67bec281082e4e2d9'
GPU_UUID = 'GPU-fa390f93-b365-d803-679e-743cdf5f68df'
P = '2.0'
GPU_BY_P = {
    '0.0': 'GPU-60696e56-351d-2b86-d0fb-55e4ff9a6199',
    '0.5': 'GPU-b887cdf4-f601-f229-6cb6-d67867be093a',
    '1.0': 'GPU-fc69ea19-3c17-a4c0-28bb-095ecf542429',
    '1.5': 'GPU-c4d54d6f-311e-aa27-ddaa-9992d7daf823',
    '2.0': 'GPU-fa390f93-b365-d803-679e-743cdf5f68df',
}
AUDIT_SHA = '2f776ebe3261c7fced674f85ee845cbb4963c08a71ab049187219f4c5fe69e32'
REF_SHA = 'a8e1d8cabbc2cd5137e009dc3c5ad8c410506e9dfcf5d1717a84967108e95d06'
P_VALUES = ['0.0', '0.5', '1.0', '1.5', '2.0']
GUARDED_ENTRYPOINT = '''"""Validate the actual effective config before any training; reuse original recorder."""
from pathlib import Path
import json
import os
import sys

work = Path(os.environ['MF_RUN_DIR'])
expected = json.loads((work / 'seed-check-provenance.json').read_text())['expected_effective_config']
sys.path.insert(0, str(work.parents[1] / 'meanflow_table1e_runbook'))
import audit_run as audit

original_install = audit.install
def guarded_install(recorder, check_only=False):
    original_install(recorder, check_only=check_only)
    if not check_only:
        import train
        recorded_train = train.train_and_evaluate
        def checked_train(config, workdir):
            if config.to_dict() != expected:
                raise RuntimeError('Effective config differs beyond training.seed; zero training steps executed.')
            print('EFFECTIVE CONFIG VERIFIED: seed=43, all other settings match this p in the primary run.', flush=True)
            return recorded_train(config, workdir)
        train.train_and_evaluate = checked_train
audit.install = guarded_install
audit.main()
'''


def configure(p):
    global P, WORK, CONFIG_NAME, CONFIG_PATH, GPU_UUID
    require(p in P_VALUES, f'Unexpected p: {p}')
    P = p
    WORK = ROOT / f'runs/diagnostic_p_{p}_seed43'
    CONFIG_NAME = f'table1e_p_{p}_seed43'
    CONFIG_PATH = REPO / f'configs/{CONFIG_NAME}.yml'
    GPU_UUID = GPU_BY_P[p]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(*argv):
    return subprocess.check_output(list(argv), text=True).strip()


def completed_record(work, seed):
    summary = read_json(work / 'run-summary.json')
    config = read_json(work / 'effective-config.json')
    require(summary['status'] == 'completed' and summary['exit_code'] == 0,
            f'Run was not successful: {work}')
    require((work / 'exit-status.txt').read_text().strip() == '0',
            f'Launcher did not record success: {work}')
    require(summary['final_train_step'] == 400320 and summary['epochs_completed'] == 80,
            f'Run did not finish 80 epochs: {work}')
    require(config['training']['seed'] == seed, f'Wrong training seed: {work}')
    fid = summary['FID_evaluations'][-1]
    require(fid['epoch'] == 80 and fid['train_step'] == 400320
            and fid['num_samples'] == 50000 and fid['NFE'] == 1
            and fid['sampling_seed'] == 0 and fid['EMA'] is True
            and math.isfinite(fid['FID_ema']), f'Wrong final FID procedure: {work}')
    require((work / 'checkpoint_400320').exists(), f'Missing final checkpoint: {work}')
    return summary, config


def environment(runtime_tmp):
    env = os.environ.copy()
    env.update(
        MF_RUN_DIR=str(WORK), CUDA_VISIBLE_DEVICES=GPU_UUID,
        MEANFLOW_SINGLE_PROCESS='1',
        MEANFLOW_VAE_PATH=str(ROOT / 'models/sd-vae-ft-mse-flax'),
        MEANFLOW_LATENT_MANIFEST=str(ROOT / 'provenance/latent-file-order.txt'),
        HF_HOME=str(ROOT / 'cache/huggingface'), JAX_PLATFORMS='cuda,cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', TMPDIR=str(runtime_tmp),
    )
    return env


def preflight():
    import yaml
    require(Path(sys.prefix).resolve() == (ROOT / 'venv').resolve(),
            'Use /work/meanflow-table1e/venv/bin/python.')
    require(not WORK.exists(), f'Refusing to reuse or overwrite output: {WORK}')
    require(not CONFIG_PATH.exists(), f'Refusing to overwrite config: {CONFIG_PATH}')
    manifest_path = ROOT / 'provenance/reproducibility-manifest.json'
    manifest = read_json(manifest_path)
    require(command('git', '-C', str(REPO), 'rev-parse', 'HEAD') == GIT_SHA
            == manifest['git_sha'], 'Repository commit changed.')
    subprocess.run(['git', '-C', str(REPO), 'diff', '--check'], check=True)
    patch = subprocess.check_output(['git', '-C', str(REPO), 'diff'])
    require(hashlib.sha256(patch).hexdigest() == manifest['source_patch_sha256'],
            'Source patch changed since the primary runs.')
    for name, expected in manifest['tracked_source_sha256'].items():
        require(checksum(REPO / name) == expected, f'Source changed: {name}')
    require(checksum(BUNDLE / 'audit_run.py') == AUDIT_SHA,
            'Recording wrapper changed; do not launch with different recording code.')
    primary_configs = {float(item['p']): item['sha256'] for item in manifest['configs']}
    baseline_summary, baseline = completed_record(ROOT / f'runs/p_{P}', 42)
    for p in P_VALUES:
        summary, cfg = completed_record(ROOT / f'runs/p_{p}', 42)
        require(summary['p'] == float(p) and cfg['method']['norm_p'] == float(p),
                f'Wrong p in primary run {p}.')
        cfg['method']['norm_p'] = float(P)
        require(cfg == baseline, f'Primary effective config differs beyond p: {p}')
        path = REPO / f'configs/table1e_p_{p}.yml'
        require(checksum(path) == primary_configs[float(p)], f'Primary config changed: {p}')
    primary_yaml = yaml.safe_load((REPO / f'configs/table1e_p_{P}.yml').read_text())
    require(primary_yaml['training']['seed'] == 42, 'Original YAML seed changed.')
    diagnostic_yaml = deepcopy(primary_yaml)
    diagnostic_yaml['training']['seed'] = 43
    normalized = deepcopy(diagnostic_yaml)
    normalized['training']['seed'] = 42
    require(normalized == primary_yaml, 'Unexpected additional config change.')
    expected_effective = deepcopy(baseline)
    expected_effective['training']['seed'] = 43
    require(baseline['method']['norm_p'] == float(P) and baseline['load_from'] is None
            and baseline['eval_only'] is False, 'Original run is not from scratch or has wrong p.')
    require(baseline['training']['num_epochs'] == 80
            and baseline['training']['batch_size'] == 256
            and baseline['method']['guidance_eq'] == 'none'
            and baseline['fid']['num_samples'] == 50000
            and baseline['sampling']['num_steps'] == 1
            and baseline['sampling']['seed'] == 0, 'Original experimental settings differ.')
    freeze = command(sys.executable, '-m', 'pip', 'freeze')
    old_freeze = (ROOT / 'provenance/full-run-packages.txt').read_text()
    require(set(freeze.splitlines()) == set(old_freeze.splitlines()),
            'Installed package set changed since the five primary runs.')
    order = ROOT / 'provenance/latent-file-order.txt'
    require(checksum(order) == manifest['latent_manifest_sha256'], 'Latent file order changed.')
    names = order.read_text().splitlines()
    require(len(names) == 1281167 and len(set(names)) == 1281167,
            'Latent filename manifest is incomplete.')
    require(set(os.listdir(ROOT / 'data/imagenet256_latents/train')) == set(names),
            'Cached latent filenames changed.')
    del names
    for name, expected in manifest['vae_file_sha256'].items():
        require(checksum(ROOT / 'models/sd-vae-ft-mse-flax' / name) == expected,
                f'VAE weights/config changed: {name}')
    require(checksum(ROOT / 'data/fid/imagenet256_authors_ref.npz') == REF_SHA
            == manifest['fid_reference_sha256'], 'FID reference changed.')
    require(checksum(ROOT / 'cache/tmp/jax_fid/inception_v3_weights_fid.pickle')
            == manifest['inception_weights_sha256'], 'FID Inception weights changed.')
    rows = [[x.strip() for x in line.split(',')] for line in command(
        'nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version',
        '--format=csv,noheader').splitlines()]
    gpu = next((row for row in rows if row[2] == GPU_UUID), None)
    require(gpu is not None and gpu[1] == 'NVIDIA B200', 'Original B200 is unavailable.')
    old_gpu = next(item for item in manifest['gpu_p_mapping'] if item['p'] == float(P))
    require(old_gpu['gpu_uuid'] == GPU_UUID and old_gpu['driver'] == gpu[4],
            'Physical GPU or driver differs from this original run.')
    require(not command('nvidia-smi', f'--id={GPU_UUID}', '--query-compute-apps=pid',
                        '--format=csv,noheader'), 'The selected B200 is busy.')
    # The existing wrapper imports training and checks interfaces, without training.
    with tempfile.TemporaryDirectory(prefix='meanflow-seed43-check-', dir='/tmp') as folder:
        runtime = Path(folder)
        (runtime / 'jax_fid').symlink_to(ROOT / 'cache/tmp/jax_fid', target_is_directory=True)
        checked = subprocess.run(
            [sys.executable, str(BUNDLE / 'audit_run.py'), '--check-hooks'],
            env=environment(runtime), cwd=REPO, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    print(checked.stdout, flush=True)
    require(checked.returncode == 0, 'Recording interface check failed; no training launched.')
    preserved_paths = [ROOT / 'results/table1e.csv', ROOT / 'results/report-summary.json',
                       manifest_path, ROOT / 'provenance/final-controller.log']
    for p in P_VALUES:
        work = ROOT / f'runs/p_{p}'
        preserved_paths.extend(work / name for name in [
            'run-summary.json', 'effective-config.json', 'fid.csv', 'final-checkpoint-files.json'])
    preserved = {str(path): checksum(path) for path in preserved_paths}
    record = {
        'checked_utc': utc(), 'purpose': 'Second training seed for every loss exponent.',
        'selection_reason': 'Post hoc repetition of all five p values after inspecting the seed-42 results.',
        'only_experimental_change': {'training.seed': {'original': 42, 'additional': 43}},
        'p': float(P), 'training_seed': 43, 'sampling_seed': 0,
        'from_scratch': True, 'resume_from_primary': False,
        'primary_FID_epoch80': baseline_summary['final_FID_epoch80'],
        'primary_directory': str(ROOT / f'runs/p_{P}'),
        'output_directory': str(WORK), 'git_sha': GIT_SHA,
        'source_patch_sha256': manifest['source_patch_sha256'],
        'primary_manifest_sha256': checksum(manifest_path),
        'same_physical_gpu_uuid': GPU_UUID, 'physical_gpu_index': int(gpu[0]),
        'gpu_name': gpu[1], 'gpu_vram': gpu[3], 'driver_version': gpu[4],
        'global_batch_size': 256, 'per_device_batch_size': 256, 'number_of_GPUs': 1,
        'expected_effective_config': expected_effective,
        'primary_files_to_preserve_sha256': preserved,
        'reporting': 'Keep both seeds for every p. Report individual FIDs and mean/sample SD across seeds 42 and 43 only when both runs completed; two seeds provide a limited variability check.',
        'billing_scope': 'One GPU executes this run; all GPUs retained in the UCloud allocation may continue to be billed.',
    }
    print('SEED-43 PREFLIGHT PASSED; zero training steps. Only training.seed changes.', flush=True)
    return diagnostic_yaml, record, freeze


def compare_results():
    old, _ = completed_record(ROOT / f'runs/p_{P}', 42)
    new, config = completed_record(WORK, 43)
    expected = read_json(WORK / 'seed-check-provenance.json')['expected_effective_config']
    require(config == expected, 'Actual diagnostic config differs beyond the training seed.')
    rows = []
    for seed, result, work in [(42, old, ROOT / f'runs/p_{P}'), (43, new, WORK)]:
        rows.append({'p': float(P), 'training_seed': seed, 'sampling_seed': 0,
                     'FID_epoch80': result['final_FID_epoch80'], 'num_samples': 50000,
                     'NFE': 1, 'epochs': 80, 'work_directory': str(work),
                     'end_to_end_hours': result['end_to_end_wall_seconds'] / 3600,
                     'median_logged_steps_per_second': result['median_logged_steps_per_second_after_step500']})
    with (WORK / 'seed-comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(WORK / 'seed-comparison.json', {
        'created_utc': utc(), 'p': float(P), 'n_training_seeds': 2,
        'paper_FID': {'0.0': 79.75, '0.5': 63.98, '1.0': 61.06, '1.5': 66.57, '2.0': 69.19}[P],
        'runs': rows, 'mean_FID': sum(row['FID_epoch80'] for row in rows) / 2,
        'interpretation': 'Descriptive mean for this p over training seeds 42 and 43; post hoc repetition, not proof of an outlier or statistically established stability.',
    })
    print(f'Completed p={P}. Both seeds saved in seed-comparison.csv; primary results preserved.', flush=True)


def launch(prechecked=None):
    import yaml
    diagnostic, record, freeze = preflight() if prechecked is None else prechecked
    require(not WORK.exists() and not CONFIG_PATH.exists(), 'Refusing to reuse an existing arm.')
    require(not command('nvidia-smi', f'--id={GPU_UUID}', '--query-compute-apps=pid',
                        '--format=csv,noheader'), 'The selected B200 became busy.')
    require(checksum(BUNDLE / 'audit_run.py') == AUDIT_SHA, 'Recorder changed after preflight.')
    # Exclusive creation: a second launcher cannot reuse this experiment directory.
    WORK.mkdir(exist_ok=False)
    monitor = None
    child = None
    runtime = None
    code = 1
    try:
        with CONFIG_PATH.open('x') as stream:
            yaml.safe_dump(diagnostic, stream, sort_keys=False)
        shutil.copy2(CONFIG_PATH, WORK / 'config.yml')
        shutil.copy2(Path(__file__).resolve(), WORK / 'launcher.py')
        record.update(config_sha256=checksum(CONFIG_PATH), launcher_sha256=checksum(Path(__file__)),
                      recorder_sha256=checksum(BUNDLE / 'audit_run.py'))
        write_json(WORK / 'seed-check-provenance.json', record)
        entrypoint = WORK / 'guarded-recording-entrypoint.py'
        entrypoint.write_text(GUARDED_ENTRYPOINT)
        (WORK / 'packages.txt').write_text(freeze + '\n')
        runtime = Path(tempfile.mkdtemp(prefix='meanflow-seed43-', dir='/tmp'))
        (runtime / 'jax_fid').symlink_to(ROOT / 'cache/tmp/jax_fid', target_is_directory=True)
        env = environment(runtime)
        argv = [sys.executable, '-u', str(entrypoint),
                f'--config=configs/load_config.py:{CONFIG_NAME}', f'--workdir={WORK}']
        keys = ['MF_RUN_DIR', 'CUDA_VISIBLE_DEVICES', 'MEANFLOW_SINGLE_PROCESS',
                'MEANFLOW_VAE_PATH', 'MEANFLOW_LATENT_MANIFEST', 'HF_HOME', 'TMPDIR',
                'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']
        write_json(WORK / 'command.json', {
            'started_utc': utc(), 'working_directory': str(REPO), 'argv': argv,
            'physical_gpu_index': record['physical_gpu_index'],
            'environment': {key: env[key] for key in keys},
            'config_sha256': record['config_sha256'],
            'entrypoint_sha256': checksum(entrypoint),
            'entrypoint': 'Checks actual effective config before training, then delegates to unchanged original audit_run recorder and authors main.py.',
        })
        (WORK / 'start.txt').write_text(utc() + '\n')
        with (WORK / 'gpu.csv').open('w') as gpu_log, (WORK / 'training.log').open('w') as train_log:
            monitor = subprocess.Popen([
                'nvidia-smi', f'--id={GPU_UUID}',
                '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv', '--loop=10'], stdout=gpu_log, stderr=subprocess.STDOUT)
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=train_log, stderr=subprocess.STDOUT)
            (WORK / 'python-pid.txt').write_text(str(child.pid) + '\n')
            print(f'LAUNCHED p={P}, training seed=43, GPU={record["physical_gpu_index"]}, PID={child.pid}.', flush=True)
            print(f'Logs and checkpoints: {WORK}', flush=True)
            code = child.wait()
        if code == 0:
            actual = read_json(WORK / 'effective-config.json')
            require(actual == record['expected_effective_config'],
                    'Diagnostic effective config differs beyond training.seed.')
            for name, expected in record['primary_files_to_preserve_sha256'].items():
                require(checksum(Path(name)) == expected, f'Primary record changed: {name}')
            (WORK / 'exit-status.txt').write_text('0\n')
            compare_results()
    except BaseException:
        code = 1
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        (WORK / 'launcher-error.txt').write_text(failure)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if runtime is not None:
            shutil.rmtree(runtime)
        (WORK / 'end.txt').write_text(utc() + '\n')
        (WORK / 'exit-status.txt').write_text(str(code) + '\n')
        print(f'SEED-43 LAUNCHER FINISHED, exit={code}.', flush=True)
    return code


def status():
    if not WORK.exists():
        print(f'p={P} seed=43: not launched yet. Check provenance/seed43-all-controller.log for preflight.')
        return
    summary = WORK / 'run-summary.json'
    if summary.exists():
        result = read_json(summary)
        print(f"p={P} seed=43: {result['status']}, step={result['final_train_step']}, "
              f"epoch={result['epochs_completed']}, final FID={result['final_FID_epoch80']}")
        print(f"Runtime: {result['end_to_end_wall_seconds'] / 3600:.3f} h")
    else:
        path = WORK / 'progress.json'
        if path.exists():
            progress = read_json(path)
            print(f"p={P} seed=43: step={progress['train_step']}, epoch={progress['epoch']}, "
                  f"loss={progress.get('loss', '-')}, steps/sec={progress.get('steps_per_second', '-')}, "
                  f"FID={progress.get('FID_ema', '-')}")
        else:
            print('Seed-43 directory created; initializing or compiling. See training.log.')
        pid_file = WORK / 'python-pid.txt'
        if pid_file.exists():
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, 0)
                print(f'Training process PID {pid} exists.')
            except ProcessLookupError:
                print(f'Training process PID {pid} no longer exists; inspect logs.')
    exit_file = WORK / 'exit-status.txt'
    if exit_file.exists():
        print('Launcher exit status:', exit_file.read_text().strip())
    print('Logs/checkpoints:', WORK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--launch', action='store_true')
    action.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    if args.check:
        preflight()
        return 0
    return launch()


if __name__ == '__main__':
    raise SystemExit('Use venv/bin/python seed43_runbook/repeat_all.py --launch, --status or --collect.')
