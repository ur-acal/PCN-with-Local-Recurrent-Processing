#!/usr/bin/env python3
"""Export deterministic PNG previews of native img_type=rggb training samples."""

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (
    ToPackedRGGB,
    linear_to_srgb,
    simple_demosaic_rggb,
    unpack_rggb,
)


DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def packed_rggb_preview(packed: torch.Tensor) -> Image.Image:
    if tuple(packed.shape) != (4, 16, 16):
        raise ValueError(
            f"Expected model input shape (4, 16, 16), got {tuple(packed.shape)}"
        )

    raw_mosaic = unpack_rggb(packed)
    rgb_linear = simple_demosaic_rggb(raw_mosaic)
    rgb_srgb = linear_to_srgb(rgb_linear.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    rgb8 = (
        rgb_srgb.mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(rgb8).resize(
        (256, 256), Image.Resampling.NEAREST
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    to_tensor = ToTensor()
    to_packed_rggb = ToPackedRGGB(return_orig=False)
    manifest_rows = []

    for task, dataset_cls in DATASETS.items():
        dataset = dataset_cls(
            root=args.data_root,
            train=True,
            download=False,
            transform=None,
        )
        destination = args.output_root / task
        destination.mkdir()

        for index in range(args.samples_per_dataset):
            source_image, label = dataset[index]
            packed = to_packed_rggb(to_tensor(source_image))
            class_name = dataset.classes[label]
            filename = (
                f"sample_{index:02d}_source_index_{index:05d}_"
                f"label_{label:03d}_{safe_name(class_name)}.png"
            )
            packed_rggb_preview(packed).save(destination / filename)
            manifest_rows.append(
                {
                    "dataset": task,
                    "split": "train",
                    "source_index": index,
                    "label": label,
                    "class_name": class_name,
                    "model_tensor_shape": "4x16x16",
                    "png": f"{task}/{filename}",
                }
            )

    with (args.output_root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    (args.output_root / "README.txt").write_text(
        "Native img_type=rggb training previews\n"
        "\n"
        "Each sample comes from the torchvision CIFAR training split and is passed "
        "through the repository's ToTensor -> ToPackedRGGB path. The resulting "
        "model input is a 4x16x16 linear-RGB tensor ordered [R, G1, G2, B]. "
        "For human viewing only, each tensor is unpacked, simply demosaiced, "
        "converted back to sRGB, and nearest-neighbor enlarged to 256x256. "
        "No stochastic crop, flip, erasing, or sensor noise is applied.\n"
    )


if __name__ == "__main__":
    main()
