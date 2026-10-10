#!/usr/bin/env python3
"""Train or evaluate one Table 1(b) setting using the upstream MeanFlow methods."""

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from runtime import initialize, load_config, new_workdir
from table_b_common import checkpoint_location, checkpoint_steps, write_json


def make_model(config):
    from table_b_model import TableBMeanFlow
    if (config.method.jvp_tangent_r, config.method.jvp_tangent_t) not in ((0, 1), (0, 0), (1, 0), (1, 1)):
        raise ValueError("Table 1(b) requires tangent_r and tangent_t to be 0 or 1")
    return TableBMeanFlow(model_str=config.model.cls, model_config={},
                    **config.method, **config.sampling)


def train(args):
    jax = initialize(allow_cpu=args.allow_cpu)
    if jax.local_device_count() != 1:
        raise ValueError("Table-b jobs require exactly one local device per run.")
    import numpy as np
    import torch
    from absl import logging
    from flax import jax_utils
    from flax.training import checkpoints
    from train import create_train_state, train_step_with_vae
    from utils import input_pipeline
    from utils.ema_util import ema_schedules
    from utils.vae_util import LatentManager

    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(1)
    config = load_config(args.config)
    config.dataset.root = str(Path(args.dataset_root or config.dataset.root).expanduser().resolve())
    if config.eval_only or config.fid.on_training or config.training.sample_on_training:
        raise ValueError("Use the separate evaluation command; training must disable FID/sampling.")
    if config.training.batch_size < 1 or config.training.num_epochs < 1:
        raise ValueError("batch_size and num_epochs must be positive")
    for field in ("log_per_step", "checkpoint_per_step", "ema_snapshot_per_step"):
        if config.training[field] < 1:
            raise ValueError(f"training.{field} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")

    stop = {"reason": None}
    def request_stop(signum, _frame):
        stop["reason"] = signal.Signals(signum).name
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    started = time.monotonic()

    loader, steps_per_epoch = input_pipeline.create_split(
        config.dataset, config.training.batch_size, split="train")
    # Stable indexing for all settings and resumed jobs; retain upstream sampler behavior.
    loader.dataset.file_list.sort()
    count = len(loader.dataset)
    expected = config.dataset.get("expected_examples", 1281167)
    if count != expected and not args.allow_small_dataset:
        raise ValueError(f"Expected {expected} ImageNet latents, found {count}. "
                         "--allow-small-dataset is only for pipeline validation.")
    if not steps_per_epoch:
        raise ValueError("Dataset must contain at least one full batch")
    # Check representative records before paying the compilation cost.
    for index in sorted({0, count // 2, count - 1}):
        image, label = loader.dataset[index]
        if tuple(image.shape) != (8, 32, 32) or not torch.isfinite(image).all():
            raise ValueError(f"Invalid latent record at index {index}")
        if not 0 <= int(label) < config.dataset.num_classes:
            raise ValueError(f"Invalid class label at index {index}")
    manifest_hash = hashlib.sha256("\n".join(loader.dataset.file_list).encode()).hexdigest()
    data_info = {"examples": count, "steps_per_epoch": steps_per_epoch,
                 "filenames_sha256": manifest_hash}
    target = config.training.num_epochs * steps_per_epoch
    if args.max_steps is not None:
        target = min(target, args.max_steps)
    workdir = Path(args.workdir).expanduser().resolve()
    if args.resume:
        previous = json.loads((workdir / "config.json").read_text())
        current = config.to_dict()
        for section in ("model", "method", "sampling", "dataset"):
            if previous[section] != current[section]:
                raise ValueError(f"Resume would change {section}; use the original configuration")
        for field in ("batch_size", "seed", "learning_rate", "adam_b2", "ema_val"):
            if previous["training"][field] != current["training"][field]:
                raise ValueError(f"Resume would change training.{field}")
        if json.loads((workdir / "dataset.json").read_text()) != data_info:
            raise ValueError("Dataset membership changed since the original run")
        if not checkpoint_steps(workdir / "checkpoints"):
            raise ValueError("No resumable checkpoint in this run")
    else:
        workdir = new_workdir(workdir)
        write_json(workdir / "config.json", config.to_dict())
        write_json(workdir / "dataset.json", data_info)

    torch.manual_seed(config.training.seed)
    key = jax.random.key(config.training.seed)
    model = make_model(config)
    state = create_train_state(key, config, model, 32, config.training.learning_rate)
    if args.resume:
        state = checkpoints.restore_checkpoint(str(workdir / "checkpoints"), state)
    step = initial_step = int(state.step)
    if step >= target:
        raise ValueError(f"Checkpoint already at step {step}; target is {target}. "
                         "Increase --max-steps/num_epochs if extending a short run.")
    state = jax_utils.replicate(state)

    class CachedEncoder:
        cached_encode = LatentManager.cached_encode

    update = jax.pmap(partial(train_step_with_vae, rng_init=key, config=config,
                            lr=config.training.learning_rate, ema_fn=ema_schedules(config),
                            latent_mnger=CachedEncoder()), axis_name="batch")
    last_saved = max(checkpoint_steps(workdir / "checkpoints"), default=-1)

    def save(full=True, ema=False):
        nonlocal last_saved
        host = jax.device_get(jax_utils.unreplicate(state))
        if full and step != last_saved:
            checkpoints.save_checkpoint(str(workdir / "checkpoints"), host, step, keep=3)
            last_saved = step
        if ema and step not in checkpoint_steps(workdir / "ema_snapshots"):
            checkpoints.save_checkpoint(str(workdir / "ema_snapshots"),
                                        {"ema_params": host.ema_params, "step": host.step},
                                        step, keep=1000000)
        print(f"Checkpoint saved at step {step}", flush=True)

    if not args.resume:
        save(ema=True)
    reason = "target_reached"
    pending_metrics = []
    last_log = time.monotonic()
    # Upstream uses a fixed sampler epoch on a single process. Replay its index order
    # and skip consumed batches, preserving exact next-batch semantics on resume.
    with (workdir / "metrics.jsonl").open("a") as metrics_file:
        for epoch in range(step // steps_per_epoch, config.training.num_epochs):
            consumed = step % steps_per_epoch
            for batch_index, batch in enumerate(loader):
                if stop["reason"] or time.monotonic() - started >= args.max_seconds:
                    reason = stop["reason"] or "time_budget"
                    break
                if batch_index < consumed:
                    continue
                try:
                    candidate, metrics = update(state, input_pipeline.prepare_batch_data(batch))
                    # Synchronize here so asynchronous device allocation errors are caught too.
                    values = {k: float(np.asarray(v).mean()) for k, v in jax.device_get(metrics).items()}
                except Exception as exc:
                    if any(word in str(exc).lower() for word in ("out of memory", "resource_exhausted", "failed to allocate")):
                        save(ema=True)
                        raise RuntimeError("Batch 256 does not fit this GPU. Select a larger GPU with "
                                           "--gpu-type and resume this run; batch size was not reduced.") from exc
                    raise
                if not all(math.isfinite(value) for value in values.values()):
                    save(ema=True)
                    raise RuntimeError(f"Non-finite metrics at step {step + 1}: {values}")
                state = candidate
                step += 1
                pending_metrics.append(values)
                if step % config.training.log_per_step == 0 or step == target:
                    now = time.monotonic()
                    record = {k: sum(m[k] for m in pending_metrics) / len(pending_metrics)
                              for k in values}
                    record.update(step=step, epochs=step / steps_per_epoch,
                                  steps_per_second=len(pending_metrics) / (now - last_log),
                                  elapsed_seconds=now - started)
                    metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
                    metrics_file.flush()
                    print(json.dumps(record), flush=True)
                    pending_metrics.clear()
                    last_log = now
                milestone = step % config.training.ema_snapshot_per_step == 0
                if step % config.training.checkpoint_per_step == 0 or milestone:
                    save(ema=milestone)
                if step >= target:
                    break
            if step >= target or reason != "target_reached":
                break
    save(ema=True)
    summary = {"status": "complete" if step >= target else "stopped", "stop_reason": reason,
               "step": step, "initial_step": initial_step, "target_step": target,
               "epochs": step / steps_per_epoch,
               "jvp_tangent_r": config.method.jvp_tangent_r,
               "jvp_tangent_t": config.method.jvp_tangent_t,
               "elapsed_seconds": time.monotonic() - started,
               "devices": [str(device) for device in jax.local_devices()], **data_info}
    write_json(workdir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args):
    jax = initialize()
    if jax.local_device_count() != 1:
        raise ValueError("Evaluation requires one local GPU")
    import numpy as np
    from absl import logging
    from flax import jax_utils
    from flax.training import checkpoints
    from PIL import Image
    from train import create_train_state, sample_step, run_p_sample_step
    from utils import fid_util
    from utils.vae_util import LatentManager

    logging.set_verbosity(logging.INFO)
    run = Path(args.run).expanduser().resolve()
    config = load_config(run / "config.json")
    config.fid.cache_ref = str(Path(args.fid_reference).expanduser().resolve())
    with np.load(config.fid.cache_ref) as cached:
        reference = {"mu": cached["ref_mu"], "sigma": cached["ref_sigma"]}
    if args.sample_batch_size < 1:
        raise ValueError("sample-batch-size must be positive")
    config.fid.device_batch_size = args.sample_batch_size
    step = args.step
    if step < 1:
        raise ValueError("Select a trained checkpoint step")
    directory, kind = checkpoint_location(run, step)
    output = new_workdir(args.workdir)
    model = make_model(config)
    state = create_train_state(jax.random.key(config.training.seed), config, model,
                               32, config.training.learning_rate)
    if kind == "ema_snapshots":
        restored = checkpoints.restore_checkpoint(str(directory),
            {"ema_params": state.ema_params, "step": state.step}, step=step)
        state = state.replace(ema_params=restored["ema_params"], step=restored["step"])
    else:
        state = checkpoints.restore_checkpoint(str(directory), state, step=step)
    if int(state.step) != step:
        raise RuntimeError("Checkpoint restore did not recover the requested step")
    state = jax_utils.replicate(state)
    latent_manager = LatentManager(config.dataset.vae, args.sample_batch_size, 32)
    sampler = jax.pmap(partial(sample_step, model=model,
        rng_init=jax.random.PRNGKey(config.sampling.seed),
        device_batch_size=args.sample_batch_size, config=config), axis_name="batch")
    # Keep the upstream Inception weights, resize, normalization and FID formula.
    inception = fid_util.build_jax_inception(batch_size=200)
    num_samples = config.fid.num_samples
    features = []
    buffered = []
    generated = 0
    for index in range(math.ceil(num_samples / args.sample_batch_size)):
        images = jax.device_get(run_p_sample_step(sampler, state, np.array([index]), latent_manager))
        images = images[:min(len(images), num_samples - generated)]
        if index == 0:
            count = min(16, len(images))
            columns = min(4, count)
            grid = Image.new("RGB", (256 * columns, 256 * math.ceil(count / columns)))
            for i, item in enumerate(images[:count]):
                grid.paste(Image.fromarray(item), (256 * (i % columns), 256 * (i // columns)))
            grid.save(output / "samples.png")
        generated += len(images)
        buffered.extend(images)
        while len(buffered) >= 200:
            features.append(fid_util.compute_batch_features(np.stack(buffered[:200]),
                            inception["fn"], inception["params"], 200))
            del buffered[:200]
        if (index + 1) % 50 == 0:
            print(f"Generated {generated}/{num_samples} images", flush=True)
    if buffered:
        features.append(fid_util.compute_batch_features(np.stack(buffered),
                        inception["fn"], inception["params"], 200))
    features = np.concatenate(features)
    if len(features) != num_samples:
        raise RuntimeError("FID must use exactly the configured number of samples")
    mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
    fid = float(fid_util.compute_fid(mu, reference["mu"], sigma, reference["sigma"]))
    if not math.isfinite(fid):
        raise RuntimeError("Non-finite FID")
    result = {"run": str(run), "step": step,
              "jvp_tangent_r": config.method.jvp_tangent_r,
              "jvp_tangent_t": config.method.jvp_tangent_t,
              "num_samples": num_samples, "num_steps": config.sampling.num_steps,
              "sampling_seed": config.sampling.seed, "sample_batch_size": args.sample_batch_size,
              "fid_reference": config.fid.cache_ref,
              "fid_reference_sha256": hashlib.sha256(Path(config.fid.cache_ref).read_bytes()).hexdigest(),
              "fid_ema": fid}
    write_json(output / "fid.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--workdir", required=True)
    training.add_argument("--dataset-root")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--max-steps", type=int)
    training.add_argument("--max-seconds", type=float, default=171000,
                          help="47h30m by default, reserving 30m before Slurm's 48h limit")
    training.add_argument("--allow-cpu", action="store_true", help="Developer validation only")
    training.add_argument("--allow-small-dataset", action="store_true", help="Pipeline validation only")
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--step", type=int, required=True)
    evaluation.add_argument("--fid-reference", required=True)
    evaluation.add_argument("--workdir", required=True)
    evaluation.add_argument("--sample-batch-size", type=int, default=20)
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
