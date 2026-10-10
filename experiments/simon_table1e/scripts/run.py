#!/usr/bin/env python3
"""Create local configs or start a run with the original training recorder."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

EXPERIMENT = Path(__file__).resolve().parents[1]
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0)


def get_config(filename):
    # ml_collections loads this file again to supply the authors' --config flag.
    from ml_collections import ConfigDict
    import yaml
    return ConfigDict(yaml.safe_load(Path(filename).read_text()))


def baseline(p, seed):
    name = f"p_{p:.1f}" if seed == 42 else f"diagnostic_p_{p:.1f}_seed43"
    return json.loads((EXPERIMENT / "runs" / name / "effective-config.json").read_text())


def outside_evidence(path):
    if path.is_relative_to(EXPERIMENT):
        raise ValueError("Local outputs must be outside experiments/simon_table1e")


def verify_hash(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def prepare(args):
    import yaml
    output = args.config_dir.resolve()
    outside_evidence(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Config directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43):
        for p in POWERS:
            config = baseline(p, seed)
            config["dataset"]["root"] = str(args.latents.resolve())
            config["fid"]["cache_ref"] = str(args.fid_reference.resolve())
            (output / f"table1e_p_{p:.1f}_seed{seed}.yml").write_text(
                yaml.safe_dump(config, sort_keys=False))
    print(f"Ten local configs written to {output}")


def launch(args):
    import yaml
    config_path = args.config_dir.resolve() / f"table1e_p_{args.p:.1f}_seed{args.seed}.yml"
    config = yaml.safe_load(config_path.read_text())
    expected = baseline(args.p, args.seed)
    for group, key in (("dataset", "root"), ("fid", "cache_ref")):
        expected[group][key] = config[group][key]
    if config != expected:
        raise ValueError("Local config differs beyond the two data paths")
    output = args.output_dir.resolve()
    outside_evidence(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory must be empty; this entry point does not resume")
    manifest = json.loads((EXPERIMENT / "provenance/reproducibility-manifest.json").read_text())
    for name, digest in manifest["tracked_source_sha256"].items():
        verify_hash(EXPERIMENT / "meanflow" / name, digest)
    for name, digest in manifest["bundle_sha256"].items():
        verify_hash(EXPERIMENT / "meanflow_table1e_runbook" / name, digest)
    if sys.version_info[:3] != (3, 12, 3):
        raise ValueError("Training requires the recorded Python 3.12.3")
    runtime = json.loads((EXPERIMENT / "runs/p_1.0/runtime.json").read_text())
    packages = {**runtime["packages"], "jax-cuda13-pjrt": "0.11.2",
                "jax-cuda13-plugin": "0.11.2"}
    for name, version in packages.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Training requires {name}=={version}")
    for name, digest in manifest["vae_file_sha256"].items():
        verify_hash(args.vae_dir.resolve() / name, digest)
    verify_hash(Path(config["fid"]["cache_ref"]), manifest["fid_reference_sha256"])
    verify_hash(args.inception_dir.resolve() / "inception_v3_weights_fid.pickle",
                manifest["inception_weights_sha256"])
    verify_hash(args.latent_manifest.resolve(), manifest["latent_manifest_sha256"])
    latent_root = Path(config["dataset"]["root"]) / "train"
    filenames = args.latent_manifest.read_text().splitlines()
    if {p.name for p in latent_root.glob("*.pt")} != set(filenames):
        raise ValueError("Latent filenames differ from the recorded manifest")
    if args.check:
        print("Source, config, external checksums and latent filenames verified; no training")
        return
    uuid = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise ValueError("Select one GPU")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="meanflow-") as temporary:
        # The authors' downloader reads TMPDIR/jax_fid; each run gets its own link.
        (Path(temporary) / "jax_fid").symlink_to(args.inception_dir.resolve(), target_is_directory=True)
        environment = {
            "CUDA_VISIBLE_DEVICES": uuid, "MEANFLOW_SINGLE_PROCESS": "1",
            "MEANFLOW_VAE_PATH": str(args.vae_dir.resolve()),
            "MEANFLOW_LATENT_MANIFEST": str(args.latent_manifest.resolve()),
            "HF_HOME": str(output.parent / "hf_cache"), "TMPDIR": temporary,
            "JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MF_RUN_DIR": str(output),
        }
        os.environ.update(environment)
        source = EXPERIMENT / "meanflow"
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location(
            "recorded_entrypoint", EXPERIMENT / "meanflow_table1e_runbook/audit_run.py")
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)
        recorder.ROOT = EXPERIMENT
        recorder.REPO = source
        sys.argv = [str(spec.origin), f"--config={Path(__file__).resolve()}:{config_path}",
                    f"--workdir={output}"]
        (output / "config.yml").write_bytes(config_path.read_bytes())
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        (output / "start.txt").write_text(started + "\n")
        (output / "command.json").write_text(json.dumps({
            "started_utc": started, "working_directory": str(source),
            "argv": [sys.executable, *sys.argv], "environment": environment,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "entrypoint": "scripts/run.py calls meanflow_table1e_runbook/audit_run.py",
        }, indent=2) + "\n")
        code = 1
        try:
            recorder.main()
            code = 0
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
            raise
        finally:
            (output / "end.txt").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
            (output / "exit-status.txt").write_text(str(code) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    configs = commands.add_parser("prepare-configs")
    configs.add_argument("--config-dir", type=Path, required=True)
    configs.add_argument("--latents", type=Path, required=True)
    configs.add_argument("--fid-reference", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=(42, 43), required=True)
    train.add_argument("--p", type=float, choices=POWERS, required=True)
    train.add_argument("--gpu", required=True, help="Physical GPU index or UUID; one B200")
    train.add_argument("--vae-dir", type=Path, required=True)
    train.add_argument("--inception-dir", type=Path, required=True)
    train.add_argument("--latent-manifest", type=Path,
                       default=EXPERIMENT / "provenance/latent-file-order.txt")
    train.add_argument("--check", action="store_true", help="Validate files without importing training code")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare-configs" else launch(args)


if __name__ == "__main__":
    main()
