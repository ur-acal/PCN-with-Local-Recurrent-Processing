"""PyTorch Dataset wrapper for reading pre-generated RAW data from HDF5 and adding noise on-the-fly.

This module provides the NoiseDataset class that reads pre-generated clean RAW data
from HDF5 files and applies noise during training, avoiding expensive RGB2RAW conversion.
"""
from collections.abc import Mapping
from difflib import get_close_matches
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from scangen.data.paths import data_directory
from scangen.data.simple_raw2rgb import rggb_to_rgb
from scangen.device import get_device
from scangen.formats.config_format import NoiseFormat
from scangen.pipeline.noise import add_noise, random_noise_levels, set_noise_seed
from scangen.pipeline.noise_d300 import pointwise_noise_levels


LOGGER = logging.getLogger("scangen.data.noise_dataset")


def find_similar_files(filepath, n=5, cutoff=0.6):
    """
    Find files in the same directory with names similar to the given filepath.

    Args:
        filepath: Path to the file you're looking for (str or Path)
        n: Maximum number of matches to return
        cutoff: Similarity threshold (0-1), higher = stricter matching

    Returns:
        List of similar Paths found in the directory
    """
    path = Path(filepath)
    directory = path.parent
    target = path.name

    try:
        files = [f.name for f in directory.iterdir()]
    except OSError:
        return []

    return get_close_matches(target, files, n=n, cutoff=cutoff)


class NoiseDataset(Dataset):
    """PyTorch Dataset that reads pre-generated RAW data from HDF5 and adds noise on-the-fly.

    This dataset reads clean RAW data that was pre-generated and stored in HDF5 format,
    then applies noise during training. This is much faster than generating RAW data
    on-the-fly from RGB images.

    The HDF5 file is expected to have the following structure:
    - /images: (N, 4, H, W) array of clean RAW images in RGGB format
    - /labels: (N,) array of integer class labels
    - /label_names: (N,) array of string class names
    - /metadata: group with attributes about the dataset

    Examples:
        >>> from scangen.data import NoiseDataset
        >>> # Use pre-generated HDF5 file
        >>> dataset = NoiseDataset(
        ...     hdf5_path=Path("cifar10_raw.h5"),
        ...     noise_config={"type": "pointwise"},
        ... )
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for rgb, clean_raw, noisy_raw, metadata in dataloader:
        ...     # Train your model here
        ...     pass
    """

    def __init__(
        self,
        hdf5_path: Path,
        train: bool,
        noise_config: NoiseFormat,
        device: torch.device = None,
        seed: int | None = None,
        make_rgb: bool | None = None,
        augment: bool | None = None,
    ) -> None:
        """Initialize NoiseDataset wrapper.

        Args:
            hdf5_path: Path to HDF5 file containing pre-generated clean RAW data
            train: True for training dataset, False for test.
            noise_config: Noise configuration dict, e.g. {'type': 'pointwise'}
            device: Device for noise generation. If None, uses CPU (recommended for DataLoader)
            cache_size: Number of images to cache in memory (0 = no caching, -1 = cache all)
            seed: Random seed for reproducible noise generation

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If HDF5 file has unexpected structure
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        if isinstance(noise_config, Mapping):
            self.noise_config = NoiseFormat.model_validate(noise_config)
        elif isinstance(noise_config, NoiseFormat):
            self.noise_config = noise_config
        else:
            raise ValueError(f"noise_config should be NoiseFormat or dict, not {type(noise_config)}")

        self.device = get_device(device) if device is not None else get_device()

        # Set random seed for noise generation
        if seed is not None:
            set_noise_seed(seed)
        self._validate_hdf5()
        self.train = train
        if train:
            self.indices = self.train_indices
        else:
            self.indices = self.test_indices

        # Leave the HDF5 file open. HDF5 provides an LRU cache on the blocks.
        self._clean_file = h5py.File(self.hdf5_path, 'r')
        self._make_rgb = make_rgb if make_rgb is not None else False
        self._augment = augment if augment is not None else False
        self._rgb_placeholder = torch.zeros(3, self.target_size[0], self.target_size[1])
        self._augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            # No ColorJitter because it assumes 3 channels (RGB)
        ])

        LOGGER.info(
            f"Created NoiseDataset with {self.dataset_size} samples from {self.hdf5_path.name}, "
            f"noise_type={self.noise_config.type}"
        )

    def __del__(self) -> None:
        if hasattr(self, "_clean_file") and self._clean_file is not None:
            self._clean_file.close()

    def _validate_hdf5(self) -> None:
        """Validate HDF5 file structure and load metadata."""
        with h5py.File(self.hdf5_path, "r") as f:
            # Check required datasets exist
            required_datasets = ["images", "labels", "train"]
            for name in required_datasets:
                if name not in f:
                    raise ValueError(f"HDF5 file missing required dataset: {name}")

            # Get dataset info
            self.dataset_size = len(f['images'])
            self.raw_shape = f['images'].shape[1:]  # (4, H, W)
            self.target_size = (self.raw_shape[1], self.raw_shape[2])  # (H, W)

            # Load metadata if available
            if 'metadata' in f:
                metadata = f['metadata']
                self.dataset_name = metadata.attrs.get('dataset_name', 'unknown')
                self.generation_info = {
                    'model_path': metadata.attrs.get('model_path', 'unknown'),
                    'generation_date': metadata.attrs.get('generation_date', 'unknown'),
                    'target_size': metadata.attrs.get('target_size', self.target_size),
                }
            else:
                self.dataset_name = 'unknown'
                self.generation_info = {}

            train = f["train"][:]
            self.train_indices = np.where(train)[0]
            self.test_indices = np.where(~train)[0]

            LOGGER.debug(
                f"HDF5 validated: {self.dataset_size} samples, "
                f"shape={self.raw_shape}, "
            )

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Get a single sample with noise applied to pre-generated RAW data.

        Args:
            idx: Sample index

        Returns:
            Tuple of (rgb_placeholder, clean_raw, noisy_raw, metadata)
            - rgb_placeholder: Empty tensor or reconstructed RGB (for compatibility)
            - clean_raw: Clean RAW data from HDF5, shape (4, H, W)
            - noisy_raw: Noisy RAW data with noise applied, shape (4, H, W)
            - metadata: Dict with noise parameters and other info
        """
        # Ensure the file is open.
        try:
            self._clean_file.id.valid
        except (ValueError, AttributeError, OSError):
            self._clean_file = h5py.File(self.hdf5_path, 'r')


        clean_raw = self._clean_file["images"][self.indices[idx]]
        label = self._clean_file["labels"][self.indices[idx]]

        if clean_raw.shape[-1] == 4:
            clean_raw = clean_raw.transpose(2, 0, 1)
        # Convert to torch tensor
        clean_raw = torch.from_numpy(clean_raw).float()
        if self._augment:
            clean_raw = self._augment_transform(clean_raw)

        # Generate noise parameters
        if self.noise_config.type == "pointwise":
            fit_noise = pointwise_noise_levels(clean_raw, self.noise_config)
            noisy_raw = clean_raw + fit_noise
        elif self.noise_config.type == "cycleisp":
            shot_noise, read_noise = random_noise_levels(self.noise_config)
            # Apply noise to clean RAW
            noisy_raw = add_noise(
                clean_raw,
                shot_noise=shot_noise,
                read_noise=read_noise,
                device=self.device
            )
        else:
            msg = f"Unknown noise type: {self.noise_config.type}"
            raise ValueError(msg)

        noisy_raw = torch.clamp(noisy_raw, 0.0, 1.0)

        # Simple demosaicing for visualization (not used in training)
        # Just average the channels to get a rough RGB
        if self._make_rgb:
            rgb_ish = rggb_to_rgb(noisy_raw)
        else:
            rgb_ish = self._rgb_placeholder

        # Prepare metadata
        metadata = {
            "noise_type": self.noise_config.type,
            "index": idx,
            "label": int(label),
            "from_hdf5": True,  # Flag to indicate this came from HDF5
        }

        return rgb_ish, clean_raw, noisy_raw, metadata


class NoiseCIFARDataset(NoiseDataset):
    """
    Dataset that reads pre-generated CIFAR RAW data from HDF5 and adds noise.

    This is a convenience class that automatically finds the appropriate HDF5 file
    for CIFAR-10 or CIFAR-100 datasets.

    Args:
        root: Path to the data directory containing HDF5 files
        input_name: The basename of the HDF5 file (without extension),
            e.g., "cifar10" or "cifar100" to load files named
            "cifar10.h5" or "cifar100.h5" respectively.
        train: If True, use training split; if False, use test split (if splits are available)
        noise_config: A `NoiseFormat` to describe noise values
        device: A `torch.device` for computations (default is gpu/mps/cpu).
        seed: an RNG seed for the noise (default None).

    Returns:
        A dataset that is a `torch.utils.data.Dataset`.

    Expected HDF5 file structure in root directory:
    ```
    data/
    ├── cifar10.h5     # All 60,000 CIFAR-10 images
    └── cifar100.h5    # All 60,000 CIFAR-100 images
    ```
    """

    def __init__(
        self,
        root: Path,
        input_name: str,
        train: bool,
        noise_config: NoiseFormat,
        device: torch.device = None,
        seed: int | None = None,
        make_rgb: bool | None = None,
        augment: bool | None = None,
    ):
        data_dir = data_directory()

        root = Path(root)
        if not root.is_absolute():
            root = data_dir / root
        candidates = [input_name, f"{input_name}.h5", f"{input_name}.hdf5"]
        data_found = [root / c for c in candidates if (root / c).exists()]
        if not data_found:
            similars = find_similar_files(root / input_name)
            raise FileNotFoundError(
                f"Could not find pre-generated HDF5 file for {input_name}. "
                f"Expected at: {root} {candidates}. Was it one of {similars}? "
                f"Please run the pre-generation script first."
            )
        hdf5_path = data_found[0]

        device = get_device(device)

        LOGGER.info(
            f"Loading NoiseCIFARDataset: {input_name} from {hdf5_path}, "
            f"train={train}, device={device}"
        )

        super().__init__(
            hdf5_path=hdf5_path,
            train=train,
            noise_config=noise_config,
            device=device,
            seed=seed,
            make_rgb=make_rgb,
            augment=augment,
        )

        self.input_name = input_name


class MyNoiseCIFARDataset(NoiseCIFARDataset):
    def __init__(self, noisy_inp=True, transform=None, **kwargs):
        self.noisy_inp = noisy_inp
        self.transform = transform
        super().__init__(**kwargs)

    def __getitem__(
            self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Accessing from self.indices, avoid mixing train and test dataset.
        """
        real_idx = self.indices[idx]
        # Ensure the file is open.
        try:
            self._clean_file.id.valid
        except (ValueError, AttributeError, OSError):
            self._clean_file = h5py.File(self.hdf5_path, 'r')

        clean_raw = self._clean_file["images"][real_idx]
        label = self._clean_file["labels"][real_idx]

        if clean_raw.shape[-1] == 4:
            clean_raw = clean_raw.transpose(2, 0, 1)
        # Convert to torch tensor
        clean_raw = torch.from_numpy(clean_raw).float()
        if self._augment:
            clean_raw = self._augment_transform(clean_raw)

        # Generate noise parameters
        if self.noise_config.type == "pointwise":
            fit_noise = pointwise_noise_levels(clean_raw, self.noise_config)
            noisy_raw = clean_raw + fit_noise
        elif self.noise_config.type == "cycleisp":
            shot_noise, read_noise = random_noise_levels(self.noise_config)
            # Apply noise to clean RAW
            noisy_raw = add_noise(
                clean_raw,
                shot_noise=shot_noise,
                read_noise=read_noise,
                device=self.device
            )
        else:
            msg = f"Unknown noise type: {self.noise_config.type}"
            raise ValueError(msg)

        noisy_raw = torch.clamp(noisy_raw, 0.0, 1.0)

        # Simple demosaicing for visualization (not used in training)
        # Just average the channels to get a rough RGB
        if self._make_rgb:
            rgb_ish = rggb_to_rgb(noisy_raw)
        else:
            rgb_ish = self._rgb_placeholder

        # Prepare metadata
        metadata = {
            "noise_type": self.noise_config.type,
            "index": real_idx,
            "label": int(label),
            "from_hdf5": True,  # Flag to indicate this came from HDF5
        }

        if self.transform is not None:
            clean_raw = self.transform(clean_raw)
            noisy_raw = self.transform(noisy_raw)
        return noisy_raw if self.noisy_inp else clean_raw, torch.tensor(label).long()