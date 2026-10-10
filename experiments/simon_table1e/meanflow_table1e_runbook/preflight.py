"""Read-only experiment checks plus persistent provenance; zero training steps."""
from pathlib import Path
from copy import deepcopy
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import yaml

ROOT = Path('/work/meanflow-table1e')
REPO = ROOT / 'meanflow'
BUNDLE = Path(__file__).resolve().parent
SHA = 'd70cb55d298ee03c53bf6da67bec281082e4e2d9'
VAE_REVISION = '7581b0a0489cc8483c876a728b830b9ce087cf03'
REF_SHA = 'a8e1d8cabbc2cd5137e009dc3c5ad8c410506e9dfcf5d1717a84967108e95d06'
MAPPING = [(0, 1.0), (1, 0.0), (2, 0.5), (3, 1.5), (4, 2.0)]


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def command(*argv):
    return subprocess.check_output(list(argv), text=True).strip()


def main():
    assert Path(sys.prefix).resolve() == (ROOT / 'venv').resolve(), 'Use venv/bin/python.'
    provenance = ROOT / 'provenance'
    assert not (provenance / 'full-launch.lock').exists(), 'A final launch has already been attempted.'
    for name, expected in read_json(BUNDLE / 'bundle-sha256.json').items():
        assert checksum(BUNDLE / name) == expected, f'Changed bundle file: {name}'
    smoke = Path((provenance / 'latest-smoke-dir.txt').read_text().strip())
    result = read_json(smoke / 'result.json')
    assert result['passed'] is True
    assert (smoke / 'exit-status.txt').read_text().strip() == '0'
    for package, version in result['packages'].items():
        assert metadata.version(package) == version, f'Packages changed after smoke: {package}'
    recorded = read_json(provenance / 'table1e-configs.json')
    hashes = {float(item['p']): item['sha256'] for item in recorded['configs']}
    baseline = yaml.safe_load((REPO / 'configs/table1e_p_1.0.yml').read_text())
    assert baseline == read_json(smoke / 'config.json'), 'Config differs from the successful smoke.'
    required = {
        'model': {'cls': 'DiT_B_4'},
        'dataset': {'name': 'imgnet_latent', 'root': str(ROOT / 'data/imagenet256_latents'),
                    'image_size': 32, 'image_channels': 4, 'num_classes': 1000, 'vae': 'mse',
                    'num_workers': 4},
        'training': {'num_epochs': 80, 'batch_size': 256, 'seed': 42,
                     'learning_rate': 0.0001, 'adam_b2': 0.95, 'ema_val': 0.9999,
                     'ema_type': 'const', 'half_precision': False, 'log_per_step': 100,
                     'checkpoint_per_epoch': 10, 'fid_per_epoch': 10, 'sample_per_epoch': 10},
        'method': {'guidance_eq': 'none', 'norm_p': 1.0, 'norm_eps': 0.01,
                   'noise_dist': 'logit_normal', 'P_mean': -0.4, 'P_std': 1.0,
                   'data_proportion': 0.75, 'class_dropout_prob': 0.1,
                   't_start': 0.0, 't_end': 1.0},
        'sampling': {'seed': 0, 'num_steps': 1},
        'fid': {'on_training': True, 'num_samples': 50000, 'device_batch_size': 20,
                'cache_ref': str(ROOT / 'data/fid/imagenet256_authors_ref.npz')},
    }
    for section, values in required.items():
        for key, value in values.items():
            assert baseline[section][key] == value, f'Unexpected {section}.{key}'
    assert baseline['load_from'] is None and baseline['eval_only'] is False
    for gpu, p in MAPPING:
        cfg_path = REPO / f'configs/table1e_p_{p:.1f}.yml'
        assert checksum(cfg_path) == hashes[p]
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg['method']['norm_p'] == p
        cfg['method']['norm_p'] = 1.0
        assert cfg == baseline, f'Unrelated setting differs for p={p}'
        out = ROOT / f'runs/p_{p:.1f}'
        out.mkdir(parents=True, exist_ok=True)
        assert not any(out.iterdir()), f'Output directory contains files: {out}'
    assert checksum(provenance / 'latent-file-order.txt') == recorded['file_order_sha256']
    names = (provenance / 'latent-file-order.txt').read_text().splitlines()
    assert len(names) == 1281167 and len(set(names)) == 1281167
    cache_names = set(os.listdir(ROOT / 'data/imagenet256_latents/train'))
    assert cache_names == set(names), 'Latent cache no longer matches its frozen manifest.'
    verification = read_json(provenance / 'latent-cache-verification.json')
    assert verification['file_count'] == 1281167 and verification['complete_filename_coverage']
    assert len(verification['class_to_idx']) == 1000
    assert verification['stored_shape'] == [8, 32, 32] and verification['dtype'] == 'float32'
    del names, cache_names
    ref = ROOT / 'data/fid/imagenet256_authors_ref.npz'
    assert checksum(ref) == REF_SHA
    with np.load(ref, allow_pickle=False) as stats:
        assert stats['ref_mu'].shape == (2048,) and stats['ref_sigma'].shape == (2048, 2048)
        assert all(np.isfinite(stats[key]).all() for key in ['ref_mu', 'ref_sigma'])
    weights = ROOT / 'cache/tmp/jax_fid/inception_v3_weights_fid.pickle'
    assert checksum(weights) == result['inception_weights_sha256']
    vae_dir = ROOT / 'models/sd-vae-ft-mse-flax'
    vae_files = {name: checksum(vae_dir / name) for name in
                 ['config.json', 'diffusion_flax_model.msgpack']}
    vae_records = {}
    for name in ['vae-snapshot.json', 'vae-revision.txt']:
        path = provenance / name
        if path.exists():
            vae_records[name] = path.read_text()
    assert VAE_REVISION in '\n'.join(vae_records.values()), 'Pinned VAE revision record is missing.'
    git_sha = command('git', '-C', str(REPO), 'rev-parse', 'HEAD')
    assert git_sha == SHA == (provenance / 'git-sha.txt').read_text().strip()
    subprocess.run(['git', '-C', str(REPO), 'diff', '--check'], check=True)
    patch = subprocess.check_output(['git', '-C', str(REPO), 'diff'])
    assert patch == (provenance / 'gpu-compat.patch').read_bytes(), 'Source changed after smoke.'
    (provenance / 'full-run-gpu-compat.patch').write_bytes(patch)
    tracked = command('git', '-C', str(REPO), 'ls-files').splitlines()
    source_hashes = {name: checksum(REPO / name) for name in tracked if (REPO / name).is_file()}
    for path in REPO.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    inventory = command('nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version',
                        '--format=csv,noheader')
    rows = [[part.strip() for part in line.split(',')] for line in inventory.splitlines()]
    assert len(rows) == 5 and all(row[1] == 'NVIDIA B200' for row in rows)
    assert {int(row[0]) for row in rows} == set(range(5))
    gpu_rows = {int(row[0]): row for row in rows}
    active = command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader')
    assert not active, f'GPU processes are still running: {active}'
    for name, argv in {
        'full-run-nvidia-smi.txt': ['nvidia-smi'],
        'full-run-lscpu.txt': ['lscpu'],
        'full-run-packages.txt': [sys.executable, '-m', 'pip', 'freeze'],
    }.items():
        (provenance / name).write_text(command(*argv) + '\n')
    (provenance / 'full-run-gpus.txt').write_text(inventory + '\n')
    # Only IPC lives in /tmp. Inception weights remain in persistent storage.
    runtime = Path(tempfile.mkdtemp(prefix='meanflow-preflight-', dir='/tmp'))
    try:
        with socket.socket(socket.AF_UNIX) as probe:
            probe.bind(str(runtime / 'probe.sock'))
        (runtime / 'jax_fid').symlink_to(weights.parent, target_is_directory=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu_rows[0][2], MEANFLOW_SINGLE_PROCESS='1',
                   MEANFLOW_VAE_PATH=str(vae_dir),
                   MEANFLOW_LATENT_MANIFEST=str(provenance / 'latent-file-order.txt'),
                   JAX_PLATFORMS='cuda,cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(runtime))
        checked = subprocess.run([sys.executable, str(BUNDLE / 'audit_run.py'), '--check-hooks'],
                                 env=env, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        (provenance / 'recording-interface-check.log').write_text(checked.stdout)
        print(checked.stdout, flush=True)
        assert checked.returncode == 0, 'Recording interface check failed; no training was launched.'
    finally:
        shutil.rmtree(runtime)
    # Archive the actual code/configs, not raw images, caches, credentials or environment secrets.
    archive = provenance / 'full-run-source.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name in tracked:
            path = REPO / name
            if path.is_file():
                output.add(path, arcname=f'meanflow/{name}')
        for path in sorted((REPO / 'configs').glob('table1e_p_*.yml')):
            output.add(path, arcname=f'meanflow/configs/{path.name}')
        for path in sorted(BUNDLE.iterdir()):
            if path.is_file() and path.suffix in ['.py', '.zsh', '.md', '.json']:
                output.add(path, arcname=f'meanflow_table1e_runbook/{path.name}')
        for name in ['prepare_latents.zsh', 'vae-snapshot.json', 'vae-revision.txt',
                     'latent-cache-verification.json', 'fid-reference.json', 'ipc-workaround.json']:
            path = provenance / name
            if path.is_file():
                output.add(path, arcname=f'provenance/{name}')
    preprocessing_records = {}
    for name in ['latent-preprocessing-start.txt', 'latent-preprocessing-end.txt',
                 'latent-preprocessing-exit-status.txt', 'imagenet-class-count.txt',
                 'imagenet-image-count.txt', 'kaggle-version.txt', 'imagenet-download-start.txt']:
        path = provenance / name
        if path.exists():
            preprocessing_records[name] = path.read_text().strip()
    cgroup = {}
    for name in ['cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.high']:
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            cgroup[name] = path.read_text().strip()
    data_archive = ROOT / 'data/downloads/imagenet-object-localization-challenge.zip'
    archive_record = {'path': str(data_archive),
                      'size_bytes': data_archive.stat().st_size if data_archive.exists() else None,
                      'note': 'Source and counts verified; entire155GB archive was not hashed.'}
    manifest = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'job_hostname': socket.gethostname(), 'python': sys.version,
        'kernel': platform.release(), 'cpu_affinity_count': len(os.sched_getaffinity(0)),
        'host_logical_cpu_count': os.cpu_count(), 'cgroup_allocation_records': cgroup,
        'repository': 'https://github.com/Gsunshine/meanflow',
        'git_sha': git_sha, 'branch': command('git', '-C', str(REPO), 'branch', '--show-current'),
        'git_status': command('git', '-C', str(REPO), 'status', '--short'),
        'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
        'tracked_source_sha256': source_hashes, 'source_archive_sha256': checksum(archive),
        'bundle_sha256': read_json(BUNDLE / 'bundle-sha256.json'),
        'validated_smoke_directory': str(smoke), 'smoke_result': result,
        'gpu_p_mapping': [{'gpu_index': gpu, 'gpu_uuid': gpu_rows[gpu][2], 'p': p,
                          'gpu_name': gpu_rows[gpu][1], 'vram': gpu_rows[gpu][3],
                          'driver': gpu_rows[gpu][4]} for gpu, p in MAPPING],
        'training_steps_per_run': 400320, 'steps_per_epoch': 5004,
        'global_batch_size': 256, 'batch_per_device': 256,
        'resume_from_smoke': False, 'training_seed': 42, 'sampling_seed': 0,
        'effective_baseline_config': deepcopy(baseline), 'configs': recorded['configs'],
        'latent_manifest_sha256': recorded['file_order_sha256'],
        'data_source': 'Kaggle imagenet-object-localization-challenge, ILSVRC/Data/CLS-LOC/train',
        'download_archive': archive_record, 'preprocessing_records': preprocessing_records,
        'latent_cache_verification': verification,
        'cache_verification_scope': 'All filenames/counts; numerical/label checks on 100 sampled tensors.',
        'preprocessing': 'RGB, authors ADM center_crop_arr to 256, ToTensor, normalize mean/std 0.5; FP32 VAE posterior mean/std cached once.',
        'cached_tensor': 'float32 [8,32,32]: mean then std; training samples posterior and multiplies by 0.18215.',
        'vae_repo': 'pcuenq/sd-vae-ft-mse-flax', 'vae_revision': VAE_REVISION,
        'vae_file_sha256': vae_files, 'vae_original_records': vae_records,
        'fid_reference_sha256': REF_SHA, 'fid_reference_record': read_json(provenance / 'fid-reference.json'),
        'inception_weights_sha256': checksum(weights),
        'fid_procedure': 'Released JAX FID/Inception, legacy_tensorflow resize, author reference; EMA; random class labels; seed 0; 50000 images; batch20 generation; batch200 features; 1 NFE.',
        'primary_comparison': 'Epoch-80 EMA FID, not best intermediate FID.',
        'paper_fid_by_p': {'0.0': 79.75, '0.5': 63.98, '1.0': 61.06, '1.5': 66.57, '2.0': 69.19},
        'limitations': [
            'One training seed per p; no estimate of training variability.',
            'Paper Table1(e) does not identify its historical seed or exact loss-stabilizer c. Released seed42 and norm_eps0.01 retained; paper mentions 1e-3 only as an example.',
            'B200/GPU and modern dependency stack differ from authors TPU stack.',
            'Frozen released unsorted data-file order is shared by all arms; historical authors file order is unknown.',
            'Released single-process sampler repeats its epoch0 order; latent use_flip is not applied. Both behaviors are preserved.',
            'Short smoke throughput is a projection, not a guaranteed runtime; simultaneous I/O and eight FID evaluations add time.',
        ],
    }
    (provenance / 'reproducibility-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (provenance / 'full-run-plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('FULL-RUN PREFLIGHT PASSED; configs/source/data/reference/versions checked; zero training steps.')


if __name__ == '__main__':
    main()
