#!/usr/bin/env python3
"""Export deterministic, displayable PNG previews from packed-RGGB HDF5 data."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SOURCES = {
    ("cifair", "cifar10"): "ciFAIR10.hdf5",
    ("cifair", "cifar100"): "ciFAIR100.hdf5",
    ("scangfi", "cifar10"): "cifar10_raw.h5",
    ("scangfi", "cifar100"): "cifar100_raw.h5",
}


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def packed_rggb_preview(packed: np.ndarray) -> Image.Image:
    if packed.shape != (4, 16, 16):
        raise ValueError(f"Expected packed RGGB shape (4, 16, 16), got {packed.shape}")
    rgb_linear = np.stack(
        (packed[0], 0.5 * (packed[1] + packed[2]), packed[3]), axis=-1
    )
    rgb8 = np.rint(255.0 * linear_to_srgb(rgb_linear)).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB").resize((256, 256), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for (img_type, task), filename in SOURCES.items():
        source = args.data_root / filename
        destination = args.output_root / img_type / task
        destination.mkdir(parents=True, exist_ok=True)
        with h5py.File(source, "r") as handle:
            train_indices = np.flatnonzero(handle["train"][:])[:5]
            for ordinal, index in enumerate(train_indices):
                packed = handle["images"][index]
                label = int(handle["labels"][index])
                output = destination / (
                    f"sample_{ordinal:02d}_source_index_{index:05d}_label_{label:03d}.png"
                )
                packed_rggb_preview(packed).save(output)


if __name__ == "__main__":
    main()
