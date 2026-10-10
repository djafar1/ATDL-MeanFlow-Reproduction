"""Run the authors' main.py with host-side recording hooks; no extra training."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import inspect
import json
import logging
import math
import os
import re
import resource
import runpy
import sys
import time
import traceback

ROOT = Path('/work/meanflow-table1e')
REPO = ROOT / 'meanflow'
sys.path.insert(0, str(REPO))


def atomic_json(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def append_csv(path, fields, data):
    new = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(data)


class Recorder:
    def __init__(self, work):
        self.work = work
        self.started = time.perf_counter()
        self.fid = None
        self.fids = []
        self.checkpoints = []
        self.training_points = []
        self.writers = []
        self.final_step = None
        self.config = None
        self.extra = {}

    def stamp(self):
        return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'wall_seconds': time.perf_counter() - self.started}

    def event(self, kind, **values):
        with (self.work / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': kind, **self.stamp(), **values},
                                    default=str) + '\n')

    def scalars(self, step, scalars):
        values = {key: float(value) for key, value in scalars.items()}
        if not all(math.isfinite(x) for x in values.values()):
            self.event('nonfinite_metrics', logged_step=int(step), values=values)
            raise RuntimeError('Non-finite logged metrics; see events.jsonl.')
        is_fid = 'FID_ema' in values
        epoch = int(step) if is_fid else int(values.get('ep', 0)) + 1
        record = {**self.stamp(), 'kind': 'fid' if is_fid else 'train',
                  'epoch': epoch, 'train_step': epoch * 5004 if is_fid else int(step),
                  **values}
        append_csv(self.work / 'metrics.csv',
                   ['utc', 'wall_seconds', 'kind', 'epoch', 'train_step',
                    'loss', 'v_loss', 'lr', 'steps_per_second', 'FID_ema'], record)
        if is_fid:
            assert self.fid is not None
            self.fid['FID_ema'] = values['FID_ema']
        else:
            self.training_points.append(record)
        atomic_json(self.work / 'progress.json', record)


class RecordingWriter:
    def __init__(self, writer, recorder):
        self.writer = writer
        self.recorder = recorder

    def write_scalars(self, step, scalars):
        self.recorder.scalars(step, scalars)
        return self.writer.write_scalars(step, scalars)

    def __getattr__(self, name):
        return getattr(self.writer, name)


def install(recorder, check_only=False):
    import jax
    import numpy as np
    import train
    from PIL import Image

    original_writer = train.metric_writers.create_default_writer
    original_train = train.train_and_evaluate
    original_create = train.create_train_state
    original_save = train.save_checkpoint
    original_evaluator = train.get_fid_evaluator
    original_samples = train.sample_util.generate_fid_samples
    original_stats = train.fid_util.compute_stats
    original_fid = train.fid_util.compute_fid

    # Check interfaces before any training starts. This does not compile or train.
    expected_interfaces = [
        (original_train, {'config', 'workdir'}),
        (original_create, {'rng', 'config', 'model', 'image_size', 'lr_value'}),
        (original_save, {'state', 'workdir'}),
        (original_evaluator, {'workdir', 'config', 'writer', 'p_sample_step', 'latent_manager'}),
    ]
    for function, names in expected_interfaces:
        assert names <= set(inspect.signature(function).parameters), function.__name__
    assert jax.local_device_count() == 1 and jax.devices()[0].device_kind == 'NVIDIA B200'
    if check_only:
        print('RECORDING INTERFACES CHECK PASSED (zero training steps)', flush=True)
        return

    def recorded_writer(*args, **kwargs):
        real = original_writer(*args, **kwargs)
        recorder.writers.append(real)
        return RecordingWriter(real, recorder)

    def recorded_create(*args, **kwargs):
        start = time.perf_counter()
        state = original_create(*args, **kwargs)
        leaves = jax.tree_util.tree_leaves(state.params)
        recorder.extra.update(initial_state_seconds=time.perf_counter() - start,
                              parameter_count=sum(x.size for x in leaves),
                              parameter_dtypes=sorted({str(x.dtype) for x in leaves}))
        recorder.event('initial_state_created', **recorder.extra)
        return state

    def recorded_save(state, workdir):
        step = int(np.asarray(jax.device_get(state.step)).reshape(-1)[0])
        start = time.perf_counter()
        result = original_save(state, workdir)
        row = {'step': step, 'seconds': time.perf_counter() - start}
        recorder.checkpoints.append(row)
        recorder.event('checkpoint_saved', **row)
        return result

    def recorded_samples(*args, **kwargs):
        start = time.perf_counter()
        samples = original_samples(*args, **kwargs)
        assert recorder.fid is not None
        recorder.fid['generation_seconds'] = time.perf_counter() - start
        recorder.fid['num_samples'] = len(samples)
        assert samples.shape == (50000, 256, 256, 3)
        assert samples.dtype == np.uint8
        epoch = recorder.fid['epoch']
        folder = recorder.work / 'qualitative'
        folder.mkdir(exist_ok=True)
        grid = samples[:16].reshape(4, 4, 256, 256, 3)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(1024, 1024, 3)
        Image.fromarray(grid).save(folder / f'epoch_{epoch:03d}.png')
        recorder.event('fid_samples_generated', epoch=epoch, num_samples=len(samples),
                       seconds=recorder.fid['generation_seconds'],
                       sample_grid_indices=list(range(16)))
        return samples

    def recorded_stats(*args, **kwargs):
        start = time.perf_counter()
        mu, sigma = original_stats(*args, **kwargs)
        if recorder.fid is not None:
            recorder.fid['inception_and_stats_seconds'] = time.perf_counter() - start
            assert mu.shape == (2048,) and sigma.shape == (2048, 2048)
            assert np.isfinite(mu).all() and np.isfinite(sigma).all()
            folder = recorder.work / 'fid'
            folder.mkdir(exist_ok=True)
            path = folder / f'epoch_{recorder.fid["epoch"]:03d}.npz'
            np.savez(path, gen_mu=mu, gen_sigma=sigma)
            recorder.fid.update(stats_file=str(path.relative_to(recorder.work)),
                                stats_sha256=checksum(path))
        return mu, sigma

    def recorded_fid(*args, **kwargs):
        start = time.perf_counter()
        result = original_fid(*args, **kwargs)
        if recorder.fid is not None:
            recorder.fid['fid_math_seconds'] = time.perf_counter() - start
        return result

    def recorded_evaluator(*args, **kwargs):
        start = time.perf_counter()
        evaluate = original_evaluator(*args, **kwargs)
        recorder.extra['inception_setup_seconds'] = time.perf_counter() - start

        def evaluate_and_record(state, epoch):
            step = int(np.asarray(jax.device_get(state.step)).reshape(-1)[0])
            context = {'epoch': int(epoch) + 1, 'train_step': step,
                       'sampling_seed': 0, 'NFE': 1, 'EMA': True}
            recorder.fid = context
            start = time.perf_counter()
            recorder.event('fid_started', **context)
            try:
                result = evaluate(state, epoch)
                assert context.get('num_samples') == 50000
                assert math.isfinite(context['FID_ema'])
                context.update(recorder.stamp(),
                               evaluation_seconds=time.perf_counter() - start)
                append_csv(recorder.work / 'fid.csv',
                           ['utc', 'wall_seconds', 'epoch', 'train_step', 'num_samples',
                            'sampling_seed', 'NFE', 'EMA', 'FID_ema', 'generation_seconds',
                            'inception_and_stats_seconds', 'fid_math_seconds',
                            'evaluation_seconds', 'stats_file', 'stats_sha256'], context)
                recorder.fids.append(dict(context))
                recorder.event('fid_completed', **context)
                return result
            except BaseException:
                recorder.event('fid_failed', epoch=int(epoch) + 1,
                               elapsed_seconds=time.perf_counter() - start)
                raise
            finally:
                recorder.fid = None
        return evaluate_and_record

    def recorded_train(config, workdir):
        recorder.config = config.to_dict()
        atomic_json(recorder.work / 'effective-config.json', recorder.config)
        assert config.training.num_epochs == 80 and config.training.batch_size == 256
        assert config.load_from is None and not config.eval_only
        assert jax.process_count() == 1 and jax.local_device_count() == 1
        assert jax.default_backend() == 'gpu'
        atomic_json(recorder.work / 'runtime.json', {
            'python': sys.version,
            'jax_configuration': jax.config.values,
            'devices': [{'platform': d.platform, 'kind': d.device_kind,
                         'logical_id': d.id} for d in jax.devices()],
            'packages': {p: metadata.version(p) for p in [
                'jax', 'jaxlib', 'flax', 'optax', 'torch', 'torchvision', 'diffusers',
                'numpy', 'scipy', 'tensorflow-cpu', 'clu', 'orbax-checkpoint']},
            'data_order': 'Frozen manifest; released single-process sampler stays at epoch 0.',
            'augmentation': 'LatentDataset accepts use_flip but does not apply flipping.',
            'output_resolution': [256, 256, 3],
            'cached_posterior_shape': [8, 32, 32],
            'vae_latent_scaling': 0.18215,
        })
        recorder.event('training_function_entered')
        result = original_train(config, workdir)
        recorder.final_step = int(np.asarray(jax.device_get(result.step)).reshape(-1)[0])
        assert recorder.final_step == 400320
        assert recorder.fids and recorder.fids[-1]['epoch'] == 80
        recorder.event('training_function_completed', final_step=recorder.final_step)
        checkpoint = recorder.work / 'checkpoint_400320'
        assert checkpoint.exists(), 'Final checkpoint was not found.'
        files = [checkpoint] if checkpoint.is_file() else sorted(
            path for path in checkpoint.rglob('*') if path.is_file())
        atomic_json(recorder.work / 'final-checkpoint-files.json', [
            {'path': str(path.relative_to(recorder.work)), 'size': path.stat().st_size,
             'sha256': checksum(path)} for path in files])
        return result

    class CompilationHandler(logging.Handler):
        def emit(self, record):
            found = re.search(r'p_train_step compiled in ([\d.]+)s', record.getMessage())
            if found:
                recorder.extra['training_compilation_reported_seconds'] = float(found[1])
                recorder.event('training_compilation_report', seconds=float(found[1]))

    train.metric_writers.create_default_writer = recorded_writer
    train.create_train_state = recorded_create
    train.save_checkpoint = recorded_save
    train.sample_util.generate_fid_samples = recorded_samples
    train.fid_util.compute_stats = recorded_stats
    train.fid_util.compute_fid = recorded_fid
    train.get_fid_evaluator = recorded_evaluator
    train.train_and_evaluate = recorded_train
    logging.getLogger().addHandler(CompilationHandler())


def main():
    work = Path(os.environ['MF_RUN_DIR'])
    recorder = Recorder(work)
    code = 1
    failure = None
    try:
        install(recorder)
        os.chdir(REPO)
        sys.argv[0] = str(REPO / 'main.py')
        runpy.run_path(str(REPO / 'main.py'), run_name='__main__')
        code = 0
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        if code:
            failure = str(error)
    except BaseException:
        failure = traceback.format_exc()
        recorder.event('run_failed', traceback=failure)
        print(failure, file=sys.stderr, flush=True)
    finally:
        for writer in recorder.writers:
            try:
                writer.flush()
                writer.close()
            except Exception:
                code = 1
                failure = (failure or '') + '\n' + traceback.format_exc()
                print(failure, file=sys.stderr, flush=True)
        complete = bool(code == 0 and recorder.final_step == 400320 and
                        recorder.fids and recorder.fids[-1]['epoch'] == 80)
        if not complete and code == 0:
            code = 1
            failure = failure or 'The author loop did not complete all required steps and final FID.'
        wall = time.perf_counter() - recorder.started
        summaries = recorder.training_points
        usable = [row['steps_per_second'] for row in summaries
                  if row['train_step'] >= 500 and row['steps_per_second'] > 0]
        import statistics
        atomic_json(work / 'run-summary.json', {
            'status': 'completed' if complete else 'failed_or_incomplete',
            'exit_code': code, 'error': failure,
            'p': recorder.config['method']['norm_p'] if recorder.config else None,
            'final_train_step': recorder.final_step,
            'epochs_completed': recorder.final_step // 5004 if recorder.final_step else 0,
            'final_FID_epoch80': recorder.fids[-1]['FID_ema'] if complete else None,
            'end_to_end_wall_seconds': wall,
            'gpu_hours_for_this_one_gpu_run': wall / 3600,
            'FID_evaluations': recorder.fids,
            'total_FID_evaluation_seconds': sum(x['evaluation_seconds'] for x in recorder.fids),
            'checkpoint_saves': recorder.checkpoints,
            'total_checkpoint_save_seconds': sum(x['seconds'] for x in recorder.checkpoints),
            'median_logged_steps_per_second_after_step500': statistics.median(usable) if usable else None,
            'process_peak_RSS_linux_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            **recorder.extra,
        })
        print(f'RECORDED RUN STATUS: {"completed" if complete else "failed_or_incomplete"}', flush=True)
        if code and failure is None:
            print(f'Process exit code: {code}', file=sys.stderr, flush=True)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-hooks']:
        install(None, check_only=True)
    else:
        main()
