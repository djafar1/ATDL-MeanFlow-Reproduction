from pathlib import Path
from functools import partial
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path("/work/meanflow-table1e")
sys.path.insert(0, str(ROOT / "meanflow"))

def digest_state(state):
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        value = np.asarray(jax.device_get(leaf))
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

def summarize_gpu(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    def peak(prefix):
        values = []
        for row in rows:
            for key, value in row.items():
                if key is not None and key.startswith(prefix) and value:
                    try:
                        values.append(float(value.strip().split()[0]))
                    except ValueError:
                        pass
        return max(values) if values else None
    return {
        "peak_memory_mib": peak("memory.used"),
        "peak_utilization_percent": peak("utilization.gpu"),
    }

def main():
    import importlib.metadata as metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    from absl import logging
    from clu import metric_writers
    from flax import jax_utils
    from flax.training import common_utils
    from configs.load_config import get_config
    import train

    logging.set_verbosity(logging.INFO)
    work = Path(os.environ["MEANFLOW_SMOKE_DIR"])
    work.mkdir(parents=True, exist_ok=True)
    assert not (work / "result.json").exists(), "Use a fresh smoke directory"

    cfg = get_config("table1e_p_1.0")
    assert cfg.training.num_epochs == 80 and cfg.training.batch_size == 256
    assert cfg.method.norm_p == 1.0 and cfg.method.guidance_eq == "none"
    assert cfg.sampling.num_steps == 1 and cfg.fid.num_samples == 50_000
    assert cfg.load_from is None
    assert jax.process_count() == 1 and jax.local_device_count() == 1
    assert jax.default_backend() == "gpu"

    print("SMOKE ONLY: 1 compilation step + 200 benchmark steps; GPU 0.", flush=True)
    print("Output:", work, flush=True)
    (work / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (work / "pid.txt").write_text(str(os.getpid()) + "\n")

    gpu_file = (work / "gpu.csv").open("w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--id=0",
        "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv", "--loop=2",
    ], stdout=gpu_file, stderr=subprocess.STDOUT)

    writer = None
    result = None
    try:
        writer = metric_writers.create_default_writer(
            logdir=str(work), just_logging=False
        )
        rng = jax.random.key(cfg.training.seed)
        loader, steps_per_epoch = train.input_pipeline.create_split(
            cfg.dataset, cfg.training.batch_size, split="train"
        )
        assert len(loader.dataset) == 1_281_167 and steps_per_epoch == 5004

        model_cfg = cfg.model.to_dict()
        model = train.MeanFlow(
            model_str=model_cfg.pop("cls"), model_config=model_cfg,
            **cfg.sampling, **cfg.method
        )
        state = train.create_train_state(
            rng, cfg, model, cfg.dataset.image_size, cfg.training.learning_rate
        )
        state = jax_utils.replicate(state)
        latent = train.LatentManager(
            cfg.dataset.vae, cfg.fid.device_batch_size, cfg.dataset.image_size
        )

        print("Preparing the authors' FID Inception model...", flush=True)
        inception = train.fid_util.build_jax_inception()
        train.fid_util.get_reference(cfg.fid.cache_ref, inception)

        step_fn = jax.pmap(partial(
            train.train_step_with_vae, rng_init=rng, config=cfg,
            lr=cfg.training.learning_rate, ema_fn=train.ema_schedules(cfg),
            latent_mnger=latent
        ), axis_name="batch")

        batches = iter(loader)
        batch = train.input_pipeline.prepare_batch_data(next(batches))
        assert batch["image"].shape == (1, 256, 32, 32, 8)

        print("Compiling batch-256 MeanFlow training...", flush=True)
        start = time.perf_counter()
        state, metrics = step_fn(state, batch)
        jax.block_until_ready((state, metrics))
        compile_seconds = time.perf_counter() - start
        assert all(
            np.isfinite(np.asarray(x)).all()
            for x in jax.tree_util.tree_leaves(metrics)
        )
        print(f"Compilation + first update: {compile_seconds:.1f} seconds", flush=True)

        timings = []
        for block in range(2):
            metrics_list = []
            start = time.perf_counter()
            for _ in range(100):
                batch = train.input_pipeline.prepare_batch_data(next(batches))
                state, metrics = step_fn(state, batch)
                metrics_list.append(metrics)
            jax.block_until_ready((state, metrics))
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            gathered = common_utils.get_metrics(metrics_list)
            assert all(
                np.isfinite(np.asarray(x)).all()
                for x in jax.tree_util.tree_leaves(gathered)
            )
            summary = {
                key: float(np.asarray(value).mean())
                for key, value in gathered.items()
            }
            summary["steps_per_second"] = 100 / elapsed
            step = int(np.asarray(state.step)[0])
            writer.write_scalars(step, summary)
            writer.flush()
            print(
                f"Step {step}: loss={summary['loss']:.6f}, "
                f"v_loss={summary['v_loss']:.3f}, "
                f"steps/sec={summary['steps_per_second']:.3f}",
                flush=True,
            )

        finite = jax.jit(lambda s: jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(x))
            for x in jax.tree_util.tree_leaves(s)
        ])))
        assert bool(finite(state)), "Non-finite parameters or optimizer state"
        assert int(np.asarray(state.step)[0]) == 201

        print("Testing checkpoint save and restore...", flush=True)
        checkpoint_dir = str(work / "checkpoints")
        train.save_checkpoint(state, checkpoint_dir)
        unreplicated = jax_utils.unreplicate(state)
        before_hash = digest_state(unreplicated)
        restore_target = jax.tree_util.tree_map(
            lambda x: np.zeros_like(np.asarray(jax.device_get(x))),
            unreplicated,
        )
        restored = train.restore_checkpoint(restore_target, checkpoint_dir)
        assert digest_state(restored) == before_hash, "Checkpoint contents differ"
        print("Checkpoint round trip passed.", flush=True)

        print("Testing 1-NFE EMA sampling and FID features on 20 images...", flush=True)
        sample_fn = jax.pmap(partial(
            train.sample_step, model=model,
            rng_init=jax.random.PRNGKey(cfg.sampling.seed),
            device_batch_size=cfg.fid.device_batch_size, config=cfg
        ), axis_name="batch")
        samples = train.run_p_sample_step(
            sample_fn, state, jnp.arange(1), latent, ema=True
        )
        samples = np.asarray(samples)
        assert samples.shape == (20, 256, 256, 3)
        assert samples.dtype == np.uint8

        grid = train.make_grid_visualization(jnp.asarray(samples), grid=4)
        assert grid.shape == (1, 1024, 1024, 3)
        writer.write_images(201, {"smoke_preview": np.asarray(grid)})

        mu, sigma = train.fid_util.compute_stats(
            samples, inception, num_workers=0
        )
        assert mu.shape == (2048,) and sigma.shape == (2048, 2048)
        assert np.isfinite(mu).all() and np.isfinite(sigma).all()
        writer.flush()
        print("Sampling, preview and FID feature checks passed.", flush=True)

        sps = 200 / sum(timings)
        total_steps = steps_per_epoch * cfg.training.num_epochs
        hours = total_steps / sps / 3600
        weight_file = (
            Path(os.environ["TMPDIR"])
            / "jax_fid/inception_v3_weights_fid.pickle"
        )
        with weight_file.open("rb") as f:
            inception_hash = hashlib.file_digest(f, "sha256").hexdigest()

        result = {
            "passed": True,
            "diagnostic_training_steps": 201,
            "compile_and_first_update_seconds": compile_seconds,
            "benchmark_block_seconds": timings,
            "steps_per_second": sps,
            "training_steps_per_80_epoch_run": total_steps,
            "training_only_hours_per_run": hours,
            "training_only_hours_five_parallel_assuming_same_throughput": hours,
            "training_only_gpu_hours_five_runs": hours * 5,
            "checkpoint_state_sha256": before_hash,
            "inception_weights_sha256": inception_hash,
            "fid50k_computed": False,
            "diagnostic_sample_count": 20,
            "gpu": jax.devices()[0].device_kind,
            "python": sys.version,
            "packages": {p: metadata.version(p) for p in [
                "jax", "jaxlib", "flax", "diffusers", "torch", "torchvision",
                "scipy", "tensorflow-cpu", "clu", "tensorboard",
                "orbax-checkpoint",
            ]},
        }
        (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Steady throughput: {sps:.3f} steps/sec", flush=True)
        print(
            f"Training-only estimate: {hours:.2f} h per run; "
            f"five parallel: about {hours:.2f} h, {hours * 5:.2f} GPU-hours.",
            flush=True,
        )
        print("FID evaluations, startup and shared-node contention add time.", flush=True)
        print("MEANFLOW SMOKE TEST PASSED", flush=True)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            gpu_file.close()
            gpu_summary = summarize_gpu(work / "gpu.csv")
            print("GPU telemetry over the whole smoke:", gpu_summary, flush=True)
            if result is not None:
                result["gpu_telemetry"] = gpu_summary
                (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
