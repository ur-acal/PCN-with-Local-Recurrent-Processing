#!/usr/bin/env python
"""
Single script to pre-generate all CIFAR-10 clean RAW data and save to HDF5.
This demonstrates the complete workflow in one place.

Usage:
    python pregenerate_cifar_raw.py
"""

from itertools import islice
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from scangen.device import get_device
from scangen.data.paths import data_directory
from scangen.pipeline.generator import RAWGenerator

LOGGER = logging.getLogger(__name__)

# Paths
CIFAR_ROOT = Path("./data")
MODEL_PATH = Path("data/weights/rgb2raw.pth")  # Adjust to your model path
OUTPUT_HDF5 = Path("./data/cifar10_raw.h5")

# Processing parameters
BATCH_SIZE = 32  # Process images in batches
TARGET_SIZE = (256, 256)  # Final RAW image size
DEVICE = get_device(None)  # Auto-detect best device

# Dataset parameters
DATASET_NAME = "cifar10"

# HDF5 parameters
COMPRESSION = "gzip"  # Use gzip compression for storage efficiency
COMPRESSION_LEVEL = 4  # Balance between compression and speed (1-9)

def load_cifar_dataset(name: str, cifar_root: Path, train: bool):
    """Load CIFAR-10 dataset with basic preprocessing."""
    LOGGER.debug(f"Loading CIFAR-10 dataset from {cifar_root} train {train}")
    if name not in ("cifar10", "cifar100"):
        raise ValueError(f"Expected cifar dataset name cifar10 or cifar100, got: {name}")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL to tensor [0,1]
    ])

    getter = CIFAR10 if name == "cifar10" else CIFAR100

    dataset = getter(
        root=str(cifar_root),
        train=train,
        download=False,
        transform=transform
    )

    LOGGER.debug(f"Loaded {len(dataset)} images")
    return dataset


def batches(datasets: list, batch_size: int, batch_limit: Optional[int] = None):
    """Iterates over datasets yielding batches with images, labels, and train flag."""
    for (ds, train) in datasets:
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # Keep original order
            num_workers=0,  # Single-threaded for GPU compatibility
            pin_memory=torch.cuda.is_available()
        )
        # The batch_limit is for debugging to reduce total time.
        for (image, labels) in islice(dataloader, batch_limit):
            yield image, labels, train


def preprocess_rgb_batch(batch_iter, device: torch.device):
    """Preprocess RGB images to match RAWGenerator input requirements."""
    for (rgb_images, labels, train) in batch_iter:
        rgb_images = rgb_images.to(device)

        # Ensure CHW format for entire batch
        if rgb_images.shape[1] != 3:  # Assuming batch is first dimension
            rgb_images = rgb_images.permute(0, 3, 1, 2)  # BHWC -> BCHW

        # Resize entire batch to 256x256 (CycleISP input requirement)
        if rgb_images.shape[2:] != (256, 256):
            rgb_images = F.interpolate(
                rgb_images,
                size=(256, 256),
                mode="bicubic",
                align_corners=True,
                antialias=True
            )

        yield torch.clamp(rgb_images, 0.0, 1.0), labels, train


def clean_raw(batch_iter, model_path: Path, device: torch.device, target_size: Tuple[int, int]):
    """Initialize the RAWGenerator with dummy noise config."""
    LOGGER.info(f"Initializing RAWGenerator on device: {DEVICE}")

    generator = RAWGenerator(
        model_path=model_path,
        device=device,
        noise=None,  # We won't use the noisy output.
        output_dim=target_size,
    )

    for (rgb_images, labels, train) in batch_iter:
        dummy_labels = ["cat" for label in labels]
        clean_raw, _, _ = generator.generate_batch(
            rgb_images=rgb_images,
            labels=labels,
            label_names=dummy_labels
        )
        yield clean_raw.cpu().numpy(), labels.numpy(), train


def tohdf5(
    rgb_images,
    output_path: Path,
    image_cnt: int,
    image_shape: Tuple[int, int, int] = (4, 256, 256),  # (channels, height, width)
    chunk_size: int = 10,
    compression: str = "gzip",
    compression_level: int = 4,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[int, Path]:
    """
    Initialize an HDF5 file with pre-allocated datasets for storing images.

    Args:
        rgb_images: Iterator over input batches to save.
        output_path: Path where the HDF5 file will be created
        image_cnt: Total number of images that will be stored
        image_shape: Shape of each image (C, H, W), default (4, 256, 256) for RGGB RAW
        chunk_size: Size of chunks to write (becomes the chunk size)
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level (1-9 for gzip)
        metadata: Optional dictionary of metadata to store

    Returns:
        Tuple of (chunk_size, output_path)
        - chunk_size: Recommended chunk size for writing
        - output_path: Path to the created HDF5 file

    Example:
        >>> chunk_size, hdf5_path = initialize_hdf5_file(
        ...     Path("cifar10_raw.h5"),
        ...     image_cnt=60000,
        ...     image_shape=(4, 256, 256),
        ...     batch_size=32
        ... )
        >>> print(f"Write images in chunks of {chunk_size} to {hdf5_path}")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate chunk shape for HDF5
    # Align chunk size with batch size for optimal write performance
    chunk_shape = (chunk_size, *image_shape)

    LOGGER.info(f"Initialized HDF5 file: {output_path}")
    LOGGER.debug(f"  Total capacity: {image_cnt} images")
    LOGGER.debug(f"  Image shape: {image_shape}")
    LOGGER.debug(f"  Chunk size: {chunk_size}")
    LOGGER.debug(f"  Compression: {compression} (level {compression_level})")

    with h5py.File(output_path, 'w') as f:
        # Main image dataset - pre-allocated with full size
        images_dataset = f.create_dataset(
            'images',
            shape=(image_cnt, *image_shape),
            dtype='float32',
            chunks=chunk_shape,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None
        )
        images_dataset.attrs['description'] = 'Image data'
        images_dataset.attrs['shape_format'] = '(N, C, H, W)'

        # Labels dataset - assuming integer labels
        labels_dataset = f.create_dataset(
            'labels',
            shape=(image_cnt,),
            dtype='uint8',
            chunks=(min(image_cnt, chunk_size * 10),),  # Larger chunks for small data
            compression=compression
        )
        labels_dataset.attrs['description'] = 'Integer class labels'

        # Metadata group
        meta_group = f.create_group('metadata')
        meta_group.attrs['total_images'] = image_cnt
        meta_group.attrs['image_shape'] = image_shape
        meta_group.attrs['chunk_size'] = chunk_size
        meta_group.attrs['creation_time'] = datetime.now().isoformat()
        meta_group.attrs['file_version'] = '1.0'

        # Add any custom metadata
        if metadata:
            for key, value in metadata.items():
                meta_group.attrs[key] = value

        # NOTE: The `train` flag is written as a scalar per chunk (broadcasted
        # to all entries). If you need per‑image train/test flags, write an
        # array matching the chunk size instead.
        train_dataset = f.create_dataset(
            'train',
            shape=(image_cnt,),
            dtype='bool',
            chunks=(min(image_cnt, chunk_size * 10),),
            fillvalue=False
        )
        train_dataset.attrs['description'] = 'Train if true else test'

        # Track writing progress (initially all false)
        progress_dataset = f.create_dataset(
            'write_progress',
            shape=(image_cnt,),
            dtype='bool',
            chunks=(min(image_cnt, chunk_size * 10),),
            fillvalue=False
        )
        progress_dataset.attrs['description'] = 'Track which images have been written'

        total_written = 0
        for (img_batch, label_batch, train) in rgb_images:
            images_chunk = img_batch.astype(np.float32)
            labels_chunk = label_batch.astype(np.uint8)
            batch_len = images_chunk.shape[0]
            if batch_len > chunk_size:
                raise ValueError(
                    "Expect image batches to match HDF5 chunk size. "
                    f"Instead see img_cnt {batch_len} and chunk_size {chunk_size}"
                )

            start_idx = total_written
            end_idx = start_idx + batch_len
            total_written = end_idx

            images_dataset[start_idx:end_idx] = images_chunk
            labels_dataset[start_idx:end_idx] = labels_chunk
            # Write a per‑image train flag (broadcast scalar to array)
            train_array = np.full(batch_len, train, dtype=bool)
            train_dataset[start_idx:end_idx] = train_array
            progress_dataset[start_idx:end_idx] = True
            f.flush()
            # Record final write metadata
            meta_group.attrs['last_write_time'] = datetime.now().isoformat()
            meta_group.attrs['last_write_index'] = total_written


def check_write_progress(hdf5_path: Path) -> Dict[str, Any]:
    """
    Check the write progress of an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Dictionary with progress information
    """
    with h5py.File(hdf5_path, 'r') as f:
        progress = f['write_progress'][:]
        metadata = f['metadata']

        total_images = metadata.attrs['total_images']
        written_count = np.sum(progress)

        return {
            'total_images': total_images,
            'written_count': written_count,
            'remaining_count': total_images - written_count,
            'percent_complete': (written_count / total_images) * 100,
            'is_complete': written_count == total_images,
            'last_write_time': metadata.attrs.get('last_write_time', 'N/A'),
            'last_write_index': metadata.attrs.get('last_write_index', 0)
        }


def count_written_images(hdf5_path: Path) -> Tuple[int, int]:
    """
    Count the number of train and test images written to an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        Tuple of (train_count, test_count) representing number of written images
    """
    with h5py.File(hdf5_path, 'r') as f:
        write_progress = f['write_progress'][:]
        train_flags = f['train'][:]
        written_train_flags = train_flags[write_progress]
        train_count = np.sum(written_train_flags)
        test_count = np.sum(write_progress) - train_count
        return train_count, test_count


def verify_hdf5(filepath):
    """Verify the saved HDF5 file by reading a sample."""
    LOGGER.info("Verifying saved HDF5 file...")

    with h5py.File(filepath, 'r') as f:
        # Print structure
        LOGGER.info("HDF5 Structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                LOGGER.info(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                LOGGER.info(f"  Group: {name}")
                for attr_name, attr_value in obj.attrs.items():
                    LOGGER.info(f"    - {attr_name}: {attr_value}")

        f.visititems(print_structure)

        # Load and check a sample
        clean_raw = f['images']
        labels = f['labels']

        LOGGER.info(f"\nSample check:")
        LOGGER.info(f"  First image shape: {clean_raw[0].shape}")
        LOGGER.info(f"  First label: {labels[0]}")
        LOGGER.info(f"  Value range: [{clean_raw[0].min():.4f}, {clean_raw[0].max():.4f}]")


def main():
    """
    Convert CIFAR-10 data into a RAW HDF5 format.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        ds = "cifar10"
        batch_cnt = 50
        target_size = (16, 16)
        device = get_device()
        cifar_root = data_directory()
        model_path = cifar_root / "weights/rgb2raw.pth"
        output_path = cifar_root / "cifar10_raw.h5"
        train_dataset = load_cifar_dataset(ds, cifar_root, train=True)
        test_dataset = load_cifar_dataset(ds, cifar_root, train=False)
        img_cnt = len(train_dataset) + len(test_dataset)
        load = batches([(train_dataset, True), (test_dataset, False)], batch_cnt)
        # A yield iterator has no len() so set the total.
        progress = tqdm(load, desc="Batches", total=(img_cnt - 1) // batch_cnt + 1)
        resize = preprocess_rgb_batch(progress, device)
        clean = clean_raw(resize, model_path, device, target_size)
        tohdf5(clean, output_path, img_cnt, (4, *target_size), batch_cnt)

        # clean_raw, labels, label_names = generate_all_raw_data(dataset, generator)
        # save_to_hdf5(clean_raw, labels, label_names, OUTPUT_HDF5)
        # NOTE: If the total number of images is not an exact multiple of the
        # chunk size, the final partial batch will be written but the
        # `write_progress` dataset may not reflect the remaining images,
        # leading to the perception that not all images were saved.
        # Ensure `chunk_size` divides `image_cnt` or handle the last partial
        # chunk explicitly.
        verify_hdf5(output_path)
        LOGGER.info(f"HDF5 file location: {OUTPUT_HDF5.absolute()}")

    except Exception as e:
        LOGGER.error(f"Error during pre-generation: {e}")
        raise

if __name__ == "__main__":
    main()
