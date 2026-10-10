"""Control-plane tests; opt-in tiny CPU integration tests exercise real MeanFlow."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "hendrix"))
import submit_table_b
from table_b_common import SETTINGS, common_step, write_json


class ControlTests(unittest.TestCase):
    def test_configs_change_only_jvp_tangents(self):
        configs = []
        for name, r, t, _ in SETTINGS:
            text = (ROOT / "configs" / "table_b" / (name + ".yml")).read_text()
            self.assertIn(f"  jvp_tangent_r: {r}\n", text)
            self.assertIn(f"  jvp_tangent_t: {t}\n", text)
            self.assertIn("  norm_p: 1.0\n", text)
            self.assertIn("  guidance_eq: none\n", text)
            self.assertIn("  batch_size: 256\n", text)
            self.assertIn("  num_epochs: 80\n", text)
            configs.append("\n".join(line for line in text.splitlines() if "jvp_tangent_" not in line))
        self.assertTrue(all(config == configs[0] for config in configs))

    def test_common_step_uses_retained_ema_when_rolling_steps_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, (name, _, _, _) in enumerate(SETTINGS):
                (root / name / "ema_snapshots" / "checkpoint_25000").mkdir(parents=True)
                (root / name / "checkpoints" / f"checkpoint_{31000 + i * 1000}").mkdir(parents=True)
                (root / name / "ema_snapshots" / "checkpoint_50000.orbax-checkpoint-tmp-1").mkdir()
            self.assertEqual(common_step(root), 25000)
            (root / "v11" / "ema_snapshots" / "checkpoint_25000").rmdir()
            with self.assertRaisesRegex(ValueError, "No shared trained checkpoint"):
                common_step(root)

    def test_launcher_preview_has_no_side_effects(self):
        with tempfile.TemporaryDirectory(prefix="table b ") as directory:
            dataset = Path(directory) / "latents"
            (dataset / "train").mkdir(parents=True)
            run = Path(directory) / "run"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/hendrix/submit_table_b.py"),
                "--dataset-root", str(dataset), "--run-root", str(run), "--gpu-type", "a100"],
                check=True, capture_output=True, text=True)
            self.assertIn("PREVIEW ONLY", result.stdout)
            self.assertIn("--array=0-3%4", result.stdout)
            self.assertIn("--time=48:00:00", result.stdout)
            self.assertFalse(run.exists())

    def test_submission_snapshot_and_arguments_with_mock_scheduler(self):
        with tempfile.TemporaryDirectory(prefix="table b ") as directory:
            root = Path(directory)
            for name in ("src", "configs", "scripts", "environment", "latents/train", "venv/bin"):
                (root / name).mkdir(parents=True)
            (root / "src" / "marker.txt").write_text("version one")
            (root / "venv" / "bin" / "python").touch()
            run = root / "output"
            argv = ["submit_table_b.py", "--dataset-root", str(root / "latents"),
                    "--run-root", str(run), "--venv", str(root / "venv"), "--gpu-type", "a100", "--submit"]
            # No Slurm command is executed; only its argument construction is tested.
            with patch.object(submit_table_b, "ROOT", root), patch.object(sys, "argv", argv), \
                 patch.object(submit_table_b.subprocess, "check_output",
                              side_effect=["gpu:a100:4(S:0)\n", "abc123\n", "12345\n"]) as command, \
                 patch.object(submit_table_b.subprocess, "run") as preflight, \
                 redirect_stdout(io.StringIO()):
                submit_table_b.main()
            self.assertEqual((run / "source" / "src" / "marker.txt").read_text(), "version one")
            self.assertIn("--array=0-3%4", command.call_args_list[2].args[0])
            manifest = json.loads((run / "experiment.json").read_text())
            self.assertEqual(len(manifest["settings"]), 4)
            self.assertEqual(manifest["git_revision"], "abc123")
            self.assertEqual(manifest["table"], "1(b)")
            preflight.assert_called_once_with([
                str(root / "venv/bin/python"), str(root / "scripts/hendrix/check_imagenet.py"),
                "latents", str(root / "latents")], check=True)

    def test_submission_rejects_partial_cache_before_snapshot_or_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latents/train").mkdir(parents=True)
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").touch()
            argv = ["submit_table_b.py", "--dataset-root", str(root / "latents"),
                    "--run-root", str(root / "output"), "--venv", str(root / "venv"),
                    "--gpu-type", "a100", "--submit"]
            with patch.object(submit_table_b, "ROOT", root), patch.object(sys, "argv", argv), \
                 patch.object(submit_table_b.subprocess, "check_output", return_value="gpu:a100:4") as calls, \
                 patch.object(submit_table_b.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "check")), \
                 redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                submit_table_b.main()
            self.assertEqual(calls.call_count, 1)
            self.assertFalse((root / "output").exists())

    def test_resume_rejects_table_e(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "experiment.json", {"table": "1(e)"})
            result = subprocess.run([sys.executable, str(ROOT / "scripts/hendrix/submit_table_b.py"),
                                    "--resume", str(root), "--gpu-type", "a100"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Only Table 1(b)", result.stderr)

    def test_resume_preview_omits_completed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latents/train").mkdir(parents=True)
            write_json(root / "experiment.json", {"table": "1(b)", "dataset_root": str(root / "latents"),
                                                  "venv": str(root / "venv")})
            for index, (name, _, _, _) in enumerate(SETTINGS):
                step = 80 if index == 2 else 50
                (root / name / "checkpoints" / f"checkpoint_{step}").mkdir(parents=True)
                write_json(root / name / "config.json", {"training": {"num_epochs": 80}})
                write_json(root / name / "dataset.json", {"steps_per_epoch": 1})
            result = subprocess.run([sys.executable, str(ROOT / "scripts/hendrix/submit_table_b.py"),
                                    "--resume", str(root), "--gpu-type", "a40"], capture_output=True, text=True, check=True)
            self.assertIn("--array=0,1,3%4", result.stdout)
            self.assertIn("PREVIEW ONLY", result.stdout)

    def test_collector_rejects_different_evaluation_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, r, t, fid in SETTINGS:
                run = root / name
                (run / "eval-25000").mkdir(parents=True)
                write_json(run / "config.json", {"method": {"guidance_eq": "none", "kappa": 0, "omega": 1, "norm_p": 1, "jvp_tangent_r": r, "jvp_tangent_t": t}})
                write_json(run / "dataset.json", {"steps_per_epoch": 5004})
                write_json(run / "eval-25000" / "fid.json", {
                    "jvp_tangent_r": r, "jvp_tangent_t": t, "step": 25000, "fid_ema": fid,
                    "num_samples": 50000, "num_steps": 1, "sampling_seed": 0,
                    "sample_batch_size": 20, "fid_reference_sha256": "same"})
            command = [sys.executable, str(ROOT / "scripts/hendrix/table_b_results.py"),
                       "collect", str(root), "--step", "25000"]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(len(result.stdout.splitlines()), 5)
            path = root / "v11" / "eval-25000" / "fid.json"
            data = json.loads(path.read_text())
            data["sample_batch_size"] = 10
            write_json(path, data)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Evaluation settings differ", result.stderr)


@unittest.skipUnless(os.environ.get("MEANFLOW_CPU_INTEGRATION") == "1", "opt-in CPU ML test")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import table_b
        table_b.initialize(allow_cpu=True)
        from models import models_dit
        # Actual DiT/JVP/loss/optimizer, with a tiny architecture for fast local checks.
        models_dit.DiT_Test = lambda **kwargs: models_dit.DiT(
            depth=1, hidden_size=32, patch_size=8, num_heads=2, **kwargs)

    def test_jvp_matches_analytical_derivatives_in_paper_order(self):
        import jax.numpy as jnp
        import numpy as np
        from table_b_model import jvp_with_time_tangents
        # u(z,t,r)=2z+3t+5r, velocity=7: expected 2*7+3*tangent_t+5*tangent_r.
        expected = {"v01": 17.0, "v00": 14.0, "v10": 19.0, "v11": 22.0}
        for name, r, t, _ in SETTINGS:
            _, derivative = jvp_with_time_tangents(lambda z, t, r: 2*z+3*t+5*r,
                jnp.array(1.0), jnp.array(2.0), jnp.array(3.0), jnp.array(7.0), r, t)
            np.testing.assert_array_equal(derivative, expected[name])

    def test_baseline_loss_and_gradients_match_upstream(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from meanflow import MeanFlow
        from table_b_model import TableBMeanFlow
        method = dict(model_str="DiT_Test", model_config={}, guidance_eq="none", kappa=0.0)
        upstream = MeanFlow(**method)
        adapter = TableBMeanFlow(**method, jvp_tangent_r=0.0, jvp_tangent_t=1.0)
        images = jax.random.normal(jax.random.key(12), (4, 32, 32, 4))
        labels = jnp.arange(4)
        variables = upstream.init({"params": jax.random.key(13)}, images, jnp.ones(4), labels)
        # Activate normally zero-initialized gates/output weights to make this a nontrivial check.
        leaves, tree = jax.tree_util.tree_flatten(variables["params"])
        keys = jax.random.split(jax.random.key(14), len(leaves))
        params = jax.tree_util.tree_unflatten(tree, [
            value + 0.01 * jax.random.normal(key, value.shape) for value, key in zip(leaves, keys)])
        def loss(model, parameters):
            return model.apply({"params": parameters}, images, labels,
                               method=model.forward, rngs={"gen": jax.random.key(15)})[0]
        a, ga = jax.value_and_grad(lambda p: loss(adapter, p))(params)
        b, gb = jax.value_and_grad(lambda p: loss(upstream, p))(params)
        np.testing.assert_array_equal(a, b)
        for actual, expected in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb), strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(any(np.any(np.asarray(value) != 0) for value in jax.tree_util.tree_leaves(ga)))

    def fixture(self, root, r=0.0, t=1.0):
        import numpy as np
        import torch
        import yaml
        dataset = root / "data" / "train"
        dataset.mkdir(parents=True)
        rng = np.random.default_rng(4)
        for index in range(8):
            latent = np.concatenate((rng.normal(size=(4, 32, 32)).astype("float32"),
                                     np.full((4, 32, 32), 0.1, dtype="float32")))
            torch.save({"image": torch.from_numpy(latent), "label": torch.tensor(index)},
                       dataset / f"{index:08d}.pt")
        config = yaml.safe_load((ROOT / "configs/table_b/v01.yml").read_text())
        config["model"]["cls"] = "DiT_Test"
        config["method"]["jvp_tangent_r"] = r
        config["method"]["jvp_tangent_t"] = t
        config["dataset"]["root"] = str(root / "data")
        config["dataset"]["num_workers"] = 0
        config["dataset"]["expected_examples"] = 8
        config["training"].update(batch_size=4, num_epochs=3, log_per_step=1,
                                  checkpoint_per_step=1, ema_snapshot_per_step=2)
        path = root / "config.yml"
        path.write_text(yaml.safe_dump(config))
        return path

    def run_train(self, config, run, steps, resume=False, seconds=60):
        import table_b
        arguments = ["table_b.py", "train", "--config", str(config), "--workdir", str(run),
                     "--allow-cpu", "--max-steps", str(steps), "--max-seconds", str(seconds)]
        if resume:
            arguments.append("--resume")
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            table_b.main()

    def test_mid_epoch_resume_recovers_identical_training_state(self):
        import jax
        import numpy as np
        from flax.training import checkpoints
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            self.run_train(config, root / "continuous", 3)
            self.run_train(config, root / "resumed", 1)
            self.run_train(config, root / "resumed", 3, resume=True)
            actual = checkpoints.restore_checkpoint(str(root / "resumed/checkpoints"), None, step=3)
            expected = checkpoints.restore_checkpoint(str(root / "continuous/checkpoints"), None, step=3)
            for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(json.loads((root / "resumed/summary.json").read_text())["step"], 3)

    def test_all_four_tangent_settings_make_finite_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, r, t, _ in SETTINGS:
                root = Path(directory) / name
                config = self.fixture(root, r, t)
                self.run_train(config, root / "run", 1)
                summary = json.loads((root / "run/summary.json").read_text())
                self.assertEqual(summary["step"], 1)
                self.assertEqual(summary["jvp_tangent_r"], r)
                self.assertEqual(summary["jvp_tangent_t"], t)

    def test_time_budget_saves_a_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            self.run_train(config, root / "run", 3, seconds=0.000001)
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["stop_reason"], "time_budget")
            self.assertTrue((root / "run/checkpoints/checkpoint_0").exists())
            self.run_train(config, root / "run", 1, resume=True)
            self.assertEqual(json.loads((root / "run/summary.json").read_text())["step"], 1)

    def test_allocation_failure_preserves_checkpoint_and_config_for_resume(self):
        import train as upstream_train
        def allocation_failure(*args, **kwargs):
            raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with patch.object(upstream_train, "train_step_with_vae", allocation_failure):
                with self.assertRaisesRegex(RuntimeError, "batch size was not reduced"):
                    self.run_train(config, root / "run", 1)
            self.assertTrue((root / "run/checkpoints/checkpoint_0").exists())
            self.assertEqual(json.loads((root / "run/config.json").read_text())["training"]["batch_size"], 4)
            self.run_train(config, root / "run", 1, resume=True)
            self.assertEqual(json.loads((root / "run/summary.json").read_text())["step"], 1)

    def test_signal_during_update_finishes_and_saves(self):
        import signal
        import train as upstream_train
        original = upstream_train.train_step_with_vae
        def signaled_update(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGUSR1)
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with patch.object(upstream_train, "train_step_with_vae", signaled_update):
                self.run_train(config, root / "run", 3)
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["stop_reason"], "SIGUSR1")
            self.assertEqual(summary["step"], 1)
            self.assertTrue((root / "run/checkpoints/checkpoint_1").exists())

    def test_evaluation_restores_ema_and_trims_partial_sample_batch(self):
        import table_b
        import numpy as np
        import yaml
        from utils import fid_util
        import train as upstream_train
        from utils import vae_util
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            data = yaml.safe_load(config.read_text())
            data["fid"]["num_samples"] = 37
            config.write_text(yaml.safe_dump(data))
            self.run_train(config, root / "run", 1)
            reference = root / "reference.npz"
            np.savez(reference, ref_mu=np.zeros(2), ref_sigma=np.eye(2))
            seen = []
            def samples(*_args):
                # 20 samples per invocation; a partial final batch must keep only 17.
                offset = 20 * len(seen)
                seen.append(offset)
                return np.stack([np.full((256, 256, 3), offset + i, dtype="uint8") for i in range(20)])
            def features(images, *_args):
                values = images[:, 0, 0, 0].astype("float32")
                return np.column_stack((values, values ** 2))
            arguments = ["table_b.py", "evaluate", "--run", str(root / "run"), "--step", "1",
                         "--fid-reference", str(reference), "--workdir", str(root / "eval")]
            original_init = table_b.initialize
            with patch.object(sys, "argv", arguments), \
                 patch.object(table_b, "initialize", lambda: original_init(allow_cpu=True)), \
                 patch.object(vae_util, "LatentManager"), \
                 patch.object(upstream_train, "run_p_sample_step", samples), \
                 patch.object(fid_util, "build_jax_inception", return_value={"fn": None, "params": None}), \
                 patch.object(fid_util, "compute_batch_features", features), \
                 patch.object(fid_util, "compute_fid", lambda mu, *_args: mu[0]), \
                 redirect_stdout(io.StringIO()):
                table_b.main()
            result = json.loads((root / "eval/fid.json").read_text())
            self.assertEqual(result["num_samples"], 37)
            self.assertEqual(result["step"], 1)
            self.assertEqual(result["fid_ema"], np.mean(np.arange(37)))
            self.assertTrue((root / "eval/samples.png").exists())


if __name__ == "__main__":
    unittest.main()
