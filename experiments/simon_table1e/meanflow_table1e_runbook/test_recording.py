"""Recorder tests with fake author interfaces; no JAX/model/training execution."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest import mock
import contextlib
import csv
import io
import json
import logging
import os
import tempfile
import unittest

import numpy as np
import audit_run as audit
import collect_results as collect


class RecordingTests(unittest.TestCase):
    def test_scalar_epoch_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = audit.Recorder(Path(directory))
            recorder.scalars(5000, {'ep': 0, 'loss': 1, 'steps_per_second': 20})
            recorder.fid = {}
            recorder.scalars(80, {'FID_ema': 61.06})
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['epoch'], '1')
            self.assertEqual(rows[1]['epoch'], '80')
            self.assertEqual(rows[1]['train_step'], '400320')
            with self.assertRaises(RuntimeError):
                recorder.scalars(5001, {'loss': float('nan')})

    def test_forwarding_and_fid_recording(self):
        jax = ModuleType('jax')
        jax.tree_util = SimpleNamespace(tree_leaves=lambda x: list(x.values()))
        jax.local_device_count = lambda: 1
        jax.devices = lambda: [SimpleNamespace(device_kind='NVIDIA B200')]
        jax.device_get = lambda x: x
        train = ModuleType('train')
        spy = {}
        original_step = object()
        train.train_step_with_vae = original_step
        sample_array = np.broadcast_to(np.zeros((1, 256, 256, 3), np.uint8),
                                       (50000, 256, 256, 3))
        mu, sigma = np.zeros(2048), np.eye(2048)
        returned = object()

        class Writer:
            def write_scalars(self, step, values):
                spy['writer'] = (step, values)
                return returned
        real_writer = Writer()
        def writer_factory(*args, **kwargs):
            spy['writer_factory'] = (args, kwargs)
            return real_writer
        def train_function(config, workdir):
            return returned
        def create(rng, config, model, image_size, lr_value):
            spy['create'] = (rng, config, model, image_size, lr_value)
            return SimpleNamespace(params={'a': np.zeros((2, 3), np.float32)})
        def save(state, workdir):
            spy['save'] = (state, workdir)
            return returned
        def samples(*args, **kwargs):
            spy['samples'] = (args, kwargs)
            return sample_array
        def stats(*args, **kwargs):
            self.assertIs(args[0], sample_array)
            spy['stats'] = (args, kwargs)
            return mu, sigma
        def fid_math(*args, **kwargs):
            self.assertIs(args[0], mu)
            spy['fid_math'] = (args, kwargs)
            return 61.06
        def evaluator(workdir, config, writer, p_sample_step, latent_manager):
            def evaluate(state, epoch):
                images = train.sample_util.generate_fid_samples('unchanged', ema=True)
                a, b = train.fid_util.compute_stats(images, 'inception', num_workers=0)
                score = train.fid_util.compute_fid(a, 'reference', b, 'ref_sigma')
                writer.write_scalars(epoch + 1, {'FID_ema': score})
                return returned
            return evaluate
        train.metric_writers = SimpleNamespace(create_default_writer=writer_factory)
        train.train_and_evaluate = train_function
        train.create_train_state = create
        train.save_checkpoint = save
        train.get_fid_evaluator = evaluator
        train.sample_util = SimpleNamespace(generate_fid_samples=samples)
        train.fid_util = SimpleNamespace(compute_stats=stats, compute_fid=fid_math)
        handlers = list(logging.getLogger().handlers)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict('sys.modules', {'jax': jax, 'train': train}):
            recorder = audit.Recorder(Path(directory))
            audit.install(recorder)
            writer = train.metric_writers.create_default_writer('log', just_logging=False)
            self.assertEqual(spy['writer_factory'], (('log',), {'just_logging': False}))
            self.assertIs(train.train_step_with_vae, original_step)
            created = train.create_train_state('rng', 'config', 'model', 32, 0.0001)
            self.assertEqual(spy['create'], ('rng', 'config', 'model', 32, 0.0001))
            self.assertEqual(recorder.extra['parameter_count'], 6)
            state = SimpleNamespace(step=np.array([400320]))
            self.assertIs(train.save_checkpoint(state, directory), returned)
            self.assertIs(spy['save'][0], state)
            evaluate = train.get_fid_evaluator(directory, 'config', writer, 'sampler', 'vae')
            self.assertIs(evaluate(state, 79), returned)
            self.assertEqual(recorder.fids[0]['epoch'], 80)
            self.assertEqual(recorder.fids[0]['num_samples'], 50000)
            self.assertEqual(recorder.fids[0]['FID_ema'], 61.06)
            self.assertEqual(spy['samples'], (('unchanged',), {'ema': True}))
            self.assertEqual(spy['stats'][1], {'num_workers': 0})
            with np.load(Path(directory) / 'fid/epoch_080.npz') as recorded:
                np.testing.assert_array_equal(recorded['gen_mu'], mu)
                np.testing.assert_array_equal(recorded['gen_sigma'], sigma)
            self.assertTrue((Path(directory) / 'qualitative/epoch_080.png').exists())
        logging.getLogger().handlers = handlers

    def test_final_result_is_epoch80_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(collect, 'ROOT', Path(directory)), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / 'runs').mkdir()
            for p in collect.PAPER:
                work = root / f'runs/p_{p}'
                work.mkdir()
                (work / 'exit-status.txt').write_text('0')
                fids = [{'epoch': 10, 'FID_ema': 20, 'evaluation_seconds': 1},
                        {'epoch': 80, 'train_step': 400320, 'FID_ema': 30,
                         'num_samples': 50000, 'NFE': 1, 'EMA': True,
                         'evaluation_seconds': 1}]
                summary = {'status': 'completed', 'p': float(p), 'final_train_step': 400320,
                           'epochs_completed': 80, 'FID_evaluations': fids}
                if p == '2.0':
                    summary['final_train_step'] = 400319
                (work / 'run-summary.json').write_text(json.dumps(summary))
            collect.main()
            with (root / 'results/table1e.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['replicated_FID_epoch80'], '30')
            self.assertEqual(rows[-1]['status'], 'incomplete_or_failed')
            self.assertEqual(rows[-1]['replicated_FID_epoch80'], '')

    def test_writer_failure_produces_failed_process_status(self):
        class BrokenWriter:
            def flush(self):
                raise OSError('fixture disk error')
        def installer(recorder):
            recorder.writers.append(BrokenWriter())
            recorder.final_step = 400320
            recorder.config = {'method': {'norm_p': 1}}
            recorder.fids = [{'epoch': 80, 'FID_ema': 61, 'evaluation_seconds': 1}]
        cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {'MF_RUN_DIR': directory}), mock.patch.object(audit, 'REPO', Path(directory)), mock.patch.object(audit, 'install', installer), mock.patch.object(audit.runpy, 'run_path', side_effect=SystemExit(0)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    audit.main()
                self.assertEqual(stopped.exception.code, 1)
                summary = json.loads((Path(directory) / 'run-summary.json').read_text())
                self.assertEqual(summary['status'], 'failed_or_incomplete')
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
