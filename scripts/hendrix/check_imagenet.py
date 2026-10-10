#!/usr/bin/env python3
"""Check the full ImageNet training tree or the authors' cached latent format."""

import argparse
import os
from pathlib import Path
import re


IMAGE_COUNT = 1_281_167
CLASS_COUNT = 1_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def check_raw(root):
    train = root / "train"
    classes = sorted(path for path in train.iterdir() if path.is_dir())
    if len(classes) != CLASS_COUNT or any(not re.fullmatch(r"n\d{8}", path.name) for path in classes):
        raise ValueError(f"Expected {CLASS_COUNT} ImageNet class folders in {train}")
    count = 0
    for folder in classes:
        with os.scandir(folder) as entries:
            count += sum(entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES for entry in entries)
    if count != IMAGE_COUNT:
        raise ValueError(f"Expected {IMAGE_COUNT} training images, found {count}")
    print(f"Raw ImageNet verified: {count} images in {len(classes)} classes", flush=True)


def check_latents(root):
    train = root / "train"
    count = 0
    with os.scandir(train) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue
            if not re.fullmatch(r"\d{8}\.pt", entry.name) or int(entry.name[:-3]) >= IMAGE_COUNT:
                raise ValueError(f"Unexpected latent filename: {entry.name}")
            count += 1
    if count != IMAGE_COUNT:
        raise ValueError(f"Expected {IMAGE_COUNT} latent records, found {count}; the cache is incomplete")
    import torch

    for index in (0, IMAGE_COUNT // 2, IMAGE_COUNT - 1):
        record = torch.load(train / f"{index:08d}.pt", map_location="cpu", weights_only=True)
        image = record["image"]
        if tuple(image.shape) != (8, 32, 32) or not torch.isfinite(image).all():
            raise ValueError(f"Invalid latent tensor at index {index}")
        if not 0 <= int(record["label"]) < CLASS_COUNT:
            raise ValueError(f"Invalid ImageNet label at index {index}")
    print(f"Latent cache verified: {count} records", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("raw", "latents"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    {"raw": check_raw, "latents": check_latents}[args.kind](args.root.expanduser().resolve())


if __name__ == "__main__":
    main()
