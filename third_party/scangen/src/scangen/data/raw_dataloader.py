"""Factory functions for creating PyTorch DataLoaders with RAW data generation.

This module provides convenient factory functions to create DataLoaders that generate
RAW sensor data on-the-fly from RGB datasets.
"""

import logging
import os
from pathlib import Path
import random
from typing import Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from scangen.pipeline.noise import dnd_noise
from scangen.data.raw_dataset import RawDataset
from scangen.device import get_device
from scangen.formats.config_format import NoiseFormat

LOGGER = logging.getLogger("scangen.data.raw_dataloader")


def _worker_init_fn(worker_id: int) -> None:
    """Initialize worker process for multi-processing compatibility."""
    # Set different random seeds for each worker to ensure diversity

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    LOGGER.debug(f"Initialized worker {worker_id} with seed {worker_seed}")


def create_raw_dataloader(
    rgb_dataset: Dataset,
    noise_config: Union[dict[str, Any],NoiseFormat,None] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    model_path: str | None = None,
    target_size: tuple[int, int] = (16, 16),
    device: str | None = None,
    pin_memory: bool | None = None,
    **dataloader_kwargs,
) -> DataLoader:
    """Create a PyTorch DataLoader that generates RAW data from RGB datasets.

    This factory function wraps an RGB dataset with RawDataset and creates a
    DataLoader with sensible defaults for RAW data generation.

    Args:
        rgb_dataset: Any PyTorch Dataset that returns (image, label) tuples
        noise_config: Noise configuration, e.g. {'type': 'cycleisp'}
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading.
                    Recommended: 0 for debugging, 2-4 for training
        model_path: Path to RGB2RAW model weights. If None, uses default weights
        target_size: Target size for RAW images as (height, width)
        device: Device for RAW generation ('cpu', 'cuda', 'mps',...). Defaults to None.
        pin_memory: Whether to pin memory. Auto-detected if None
        **dataloader_kwargs: Additional arguments passed to DataLoader

    Returns:
        DataLoader that yields (rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list)
        where RAW data is in 4-channel RGGB format at half resolution

    Examples:
        >>> from torchvision.datasets import CIFAR10
        >>> from scangen.data import create_raw_dataloader
        >>> # Load CIFAR-10
        >>> rgb_dataset = CIFAR10(root="./data", train=True, download=True)
        >>> # Create RAW DataLoader
        >>> dataloader = create_raw_dataloader(
        ...     rgb_dataset=rgb_dataset, noise_config={"type": "pointwise"}, batch_size=32, num_workers=2
        ... )
        >>> # Use in training loop
        >>> for rgb, clean_raw, noisy_raw, metadata in dataloader:
        ...     # rgb: (batch_size, 3, 16, 16)
        ...     # clean_raw: (batch_size, 4, 128, 128)
        ...     # noisy_raw: (batch_size, 4, 128, 128)
        ...     # metadata: list of dicts with noise parameters
        ...     pass
    """
    # Smart num_workers handling based on device
    device = get_device(device)
    if device.type != "cpu":  # GPU devices
        if num_workers > 0:
            LOGGER.warning(
                f"num_workers={num_workers} ignored for GPU device '{device}'. "
                "Using num_workers=0 to avoid multiprocessing issues with CUDA/MPS."
            )
        num_workers = 0  # Force single-threaded for GPU
    else:  # CPU device
        if num_workers == 0:
            # Use half the CPU cores as default for CPU
            cpu_cnt = os.cpu_count() or 1
            default_workers = max(1, cpu_cnt // 2)
            num_workers = default_workers
            LOGGER.info(f"Using default num_workers={num_workers} for CPU device")

    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and device != "cpu"

    if noise_config is None:
        noise_config = dnd_noise()
    noise_model = NoiseFormat.model_validate(noise_config)
    # Log configuration
    LOGGER.info(
        f"Creating RAW DataLoader: batch_size={batch_size}, "
        f"num_workers={num_workers}, target_size={target_size}, "
        f"noise_type={noise_model.type}"
    )

    # Create RAW dataset wrapper
    raw_dataset = RawDataset(
        rgb_dataset=rgb_dataset,
        noise_config=noise_model,
        model_path=Path(model_path),
        target_size=target_size,
        device=device,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset=raw_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        collate_fn=_collate_raw_batch,
        **dataloader_kwargs,
    )

    LOGGER.info(f"Created DataLoader with {len(raw_dataset)} samples")
    return dataloader


def _collate_raw_batch(batch):
    """Custom collate function for RAW dataset batches.

    Args:
        batch: List of (rgb, clean_raw, noisy_raw, metadata) tuples

    Returns:
        Tuple of (rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list)
    """
    rgb_images = []
    clean_raws = []
    noisy_raws = []
    metadata_list = []

    for rgb, clean_raw, noisy_raw, metadata in batch:
        rgb_images.append(rgb)
        clean_raws.append(clean_raw)
        noisy_raws.append(noisy_raw)
        metadata_list.append(metadata)

    # Stack tensors into batches
    rgb_batch = torch.stack(rgb_images)
    clean_raw_batch = torch.stack(clean_raws)
    noisy_raw_batch = torch.stack(noisy_raws)

    return rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list


def create_cifar10_raw_dataloader(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    noise_config: Union[NoiseFormat,dict[str, Any],None] = None,
    **kwargs,
) -> DataLoader:
    """Convenience function to create a CIFAR-10 RAW DataLoader.

    Args:
        root: Root directory for CIFAR-10 data
        train: Whether to use training or test split
        download: Whether to download CIFAR-10 if not present
        noise_config: Noise configuration. Defaults to DND noise
        **kwargs: Additional arguments passed to create_raw_dataloader

    Returns:
        DataLoader for CIFAR-10 with RAW data generation

    Examples:
        >>> from scangen.data import create_cifar10_raw_dataloader
        >>> # Simple usage with defaults
        >>> dataloader = create_cifar10_raw_dataloader()
        >>> # Custom configuration
        >>> dataloader = create_cifar10_raw_dataloader(
        ...     root="./datasets", noise_config={"type": "pointwise"}, batch_size=64, num_workers=4
        ... )
    """
    # Create CIFAR-10 dataset with minimal transforms
    # Note: RawDataset will handle resizing and normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL to tensor and scale to [0,1]
        ]
    )

    cifar_dataset = CIFAR10(root=root, train=train, download=download, transform=transform)

    if noise_config is None:
        noise_config = dnd_noise()
    noise_model = NoiseFormat.model_validate(noise_config)
    # Create RAW DataLoader
    return create_raw_dataloader(rgb_dataset=cifar_dataset, noise_config=noise_model, **kwargs)
