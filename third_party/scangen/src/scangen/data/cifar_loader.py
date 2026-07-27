from collections.abc import Sized
from importlib import resources
import logging
import tomllib
from typing import cast, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LOGGER = logging.getLogger("scangen.data.cifar_loader")


class CIFARDataset(Dataset):
    """Base class for CIFAR dataset handling with preprocessing.

    Provides high-quality upscaling from CIFAR's 32x32 images to configurable
    target resolutions using bicubic interpolation, with optional preprocessing
    and label preservation.
    """
    blur_kernel: Optional[torch.Tensor]

    def __init__(
        self,
        dataset_name: str = "cifar10",
        root: str = "./data",
        train: bool = True,
        target_size: tuple[int, int] = (256, 256),
        download: bool = False,
        apply_blur: bool = True,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ) -> None:
        """Initialize CIFAR dataset loader.

        Args:
            dataset_name: Either "cifar10" or "cifar100"
            root: Root directory for dataset storage
            train: Whether to use training set (True) or test set (False)
            target_size: Target image size (height, width) after upscaling
            download: Whether to download dataset if not found
            apply_blur: Whether to apply Gaussian blur preprocessing
            blur_kernel_size: Kernel size for Gaussian blur
            blur_sigma: Standard deviation for Gaussian blur
        """
        self.dataset_name = dataset_name.lower()
        self.target_size = target_size
        self.apply_blur = apply_blur

        # Create transforms for loading raw data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL to tensor [0,1]
            ]
        )
        LOGGER.debug(f"Looking for {dataset_name} in {root} download {download}")

        # Load appropriate CIFAR dataset
        if self.dataset_name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=download, transform=transform
            )
            self.class_names = get_cifar_labels("cifar10")
        elif self.dataset_name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=download, transform=transform
            )
            self.class_names = get_cifar_labels("cifar100")
        else:
            msg = f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'."
            raise ValueError(msg)

        # Prepare Gaussian blur if needed
        if apply_blur:
            self.blur_kernel = self._create_gaussian_kernel(blur_kernel_size, blur_sigma)
        else:
            self.blur_kernel = None

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Get preprocessed image, label index, and label name.

        Args:
            idx: Dataset index

        Returns:
            Tuple of (processed_image, label_index, label_name)
            - processed_image: Tensor of shape (3, H, W) with values in [0,1]
            - label_index: Integer class index
            - label_name: String class name
        """
        image, label = self.dataset[idx]

        # Upscale from 32x32 to target size using high-quality interpolation
        image = self._upscale_image(image)

        # Apply Gaussian blur if configured
        if self.apply_blur and self.blur_kernel is not None:
            image = self._apply_gaussian_blur(image)

        # Ensure divisible by 16 (common requirement for models)
        image = self._make_divisible_by_16(image)

        label_name = self.class_names[label]

        return image, label, label_name

    def _upscale_image(self, image: torch.Tensor) -> torch.Tensor:
        """Upscale image using high-quality bicubic interpolation.

        Args:
            image: Input tensor of shape (3, 32, 32)

        Returns:
            Upscaled tensor of shape (3, target_height, target_width)
        """
        # Add batch dimension for interpolation
        image = image.unsqueeze(0)  # Shape: (1, 3, 32, 32)

        # Bicubic interpolation for high quality upscaling
        upscaled = F.interpolate(image, size=self.target_size, mode="bicubic", align_corners=False)

        # Remove batch dimension
        return upscaled.squeeze(0)  # Shape: (3, H, W)

    def _apply_gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to reduce noise from upscaling artifacts.

        Args:
            image: Input tensor of shape (3, H, W)

        Returns:
            Blurred tensor of same shape
        """
        assert self.blur_kernel is not None
        
        # Add batch dimension
        image = image.unsqueeze(0)  # Shape: (1, 3, H, W)

        # Apply Gaussian blur (separable convolution for efficiency)
        blurred = F.conv2d(
            image, self.blur_kernel, padding=self.blur_kernel.size(-1) // 2, groups=3
        )

        # Remove batch dimension
        return blurred.squeeze(0)

    def _make_divisible_by_16(self, image: torch.Tensor) -> torch.Tensor:
        """Crop image to be divisible by 16 (neural network requirement).

        Args:
            image: Input tensor of shape (3, H, W)

        Returns:
            Cropped tensor with dimensions divisible by 16
        """
        _, h, w = image.shape

        # Calculate new dimensions
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        # Center crop to new dimensions
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2

        return image[:, start_h : start_h + new_h, start_w : start_w + new_w]

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create separable Gaussian blur kernel.

        Args:
            kernel_size: Size of the kernel (should be odd)
            sigma: Standard deviation for Gaussian distribution

        Returns:
            Gaussian kernel tensor of shape (3, 1, kernel_size, kernel_size)
        """
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2

        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()

        # Create 2D kernel from outer product
        kernel_2d = g[:, None] * g[None, :]

        # Reshape for convolution (3 channels, 1 input channel per group, H, W)
        return kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()


class CIFAR10Dataset(CIFARDataset):
    """CIFAR-10 dataset loader with preprocessing."""

    def __init__(self, root: str = "./data", train: bool = True, **kwargs) -> None:
        """Initialize CIFAR-10 dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to use training set (True) or test set (False)
            **kwargs: Additional arguments passed to CIFARDataset
        """
        super().__init__(dataset_name="cifar10", root=root, train=train, **kwargs)


class CIFAR100Dataset(CIFARDataset):
    """CIFAR-100 dataset loader with preprocessing."""

    def __init__(self, root: str = "./data", train: bool = True, **kwargs) -> None:
        """Initialize CIFAR-100 dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to use training set (True) or test set (False)
            **kwargs: Additional arguments passed to CIFARDataset
        """
        super().__init__(dataset_name="cifar100", root=root, train=train, **kwargs)


def create_cifar_loader(
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: tuple[int, int] = (256, 256),
    **dataset_kwargs,
) -> DataLoader:
    """Create a CIFAR DataLoader with preprocessing.

    Args:
        dataset_name: Either "cifar10" or "cifar100"
        batch_size: Batch size for loading
        train: Whether to use training set
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        target_size: Target image size after upscaling
        **dataset_kwargs: Additional arguments passed to dataset

    Returns:
        DataLoader instance with configured dataset

    Examples:
        >>> loader = create_cifar_loader("cifar10", batch_size=8, target_size=(256, 256))
        >>> for images, labels, names in loader:
        ...     print(f"Batch shape: {images.shape}")
        ...     print(f"Labels: {names}")
        ...     break
    """
    dataset: Union[CIFAR10Dataset, CIFAR100Dataset]
    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10Dataset(train=train, target_size=target_size, **dataset_kwargs)
    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100Dataset(train=train, target_size=target_size, **dataset_kwargs)
    else:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,  # Only use persistent workers when multi-threading
    )


def get_cifar_labels(dataset_name: str = "cifar10") -> list[str]:
    """Get class label names for CIFAR dataset.

    Args:
        dataset_name: Either "cifar10" or "cifar100"

    Returns:
        List of class names

    Examples:
        >>> labels = get_cifar_labels("cifar10")
        >>> len(labels)
        10
        >>> labels[0]
        'airplane'
    """
    name_lower = dataset_name.lower()
    if name_lower in ["cifar10", "cifar100"]:
        ref = resources.files("scangen.data") / f"{name_lower}.toml"
        try:
            with ref.open("rb") as finput:
                data = tomllib.load(finput)
                return data["class_names"].copy()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Cannot find dataset labels in data/*.toml that should be "
                f"installed with the package install. Looking for data/{name_lower}.toml"
                )
    msg = f"Unknown dataset: {dataset_name}"
    raise ValueError(msg)


class BatchSampler:
    """Sampling utility for generating batches with replacement.

    Supports sampling images with replacement for data augmentation
    and stochastic batch generation as specified in the requirements.
    """

    def __init__(self, dataset_size: int, batch_size: int, seed: int | None = None) -> None:
        """Initialize batch sampler.

        Args:
            dataset_size: Total number of samples in dataset
            batch_size: Number of samples per batch
            seed: Random seed for reproducible sampling
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    def sample_batch(self) -> list[int]:
        """Sample a batch of indices with replacement.

        Returns:
            List of dataset indices for the batch
        """
        return self.rng.choice(self.dataset_size, size=self.batch_size, replace=True).tolist()

    def sample_multiple_batches(self, num_batches: int) -> list[list[int]]:
        """Sample multiple batches of indices.

        Args:
            num_batches: Number of batches to generate

        Returns:
            List of batch index lists
        """
        return [self.sample_batch() for _ in range(num_batches)]


def create_reproducible_cifar_loader(
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    num_batches: int = 1,
    seed: int | None = None,
    num_workers: int = 4,
    **loader_kwargs,
) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    """Create reproducible CIFAR batches with memory-efficient sampling.

    Generates a specific number of batches with deterministic sampling
    for reproducible RAW data generation. Only loads individual images
    as needed, avoiding loading the entire dataset into memory.

    Args:
        dataset_name: Either "cifar10" or "cifar100"
        batch_size: Number of images per batch
        num_batches: Number of batches to generate
        seed: Random seed for reproducible sampling
        num_workers: Number of worker threads for data loading
        **loader_kwargs: Additional arguments for create_cifar_loader

    Returns:
        List of batches, each containing (images, labels, label_names)

    Examples:
        >>> batches = create_reproducible_cifar_loader(
        ...     "cifar10", batch_size=4, num_batches=2, seed=42
        ... )
        >>> len(batches)
        2
        >>> images, labels, names = batches[0]
        >>> images.shape
        torch.Size([4, 3, 256, 256])
    """
    # Create a minimal dataset loader to get dataset size without loading all data
    seed = seed or 42
    temp_loader = create_cifar_loader(
        dataset_name, batch_size=1, shuffle=False, num_workers=0, **loader_kwargs
    )
    dataset_size = len(cast(Sized, temp_loader.dataset))
    del temp_loader  # Ensure no reference remains

    # Get label names for the dataset
    get_cifar_labels(dataset_name)

    # Create batch sampler with known dataset size
    sampler = BatchSampler(dataset_size, batch_size, seed=seed)

    # Create a single dataset instance for efficient access
    dataset_loader = create_cifar_loader(
        dataset_name, batch_size=1, shuffle=False, num_workers=num_workers, **loader_kwargs
    )
    dataset = dataset_loader.dataset

    # Generate batches by loading only required images
    batches = []
    for _batch_idx in range(num_batches):
        # Get indices for this batch
        indices = sampler.sample_batch()

        # Load only the specific images we need for this batch
        batch_images = []
        batch_labels = []
        batch_names = []

        # Load each required image individually
        for idx in indices:
            # Get the specific image by index - this only loads one image
            # CIFAR dataset returns (image, label_index, label_name)
            image, label, label_name = dataset[idx]

            batch_images.append(image)
            batch_labels.append(label)
            batch_names.append(label_name)

        # Convert to tensors and stack
        batch_images_tensor = torch.stack(batch_images)
        batch_labels_tensor = torch.tensor(batch_labels)

        batches.append((batch_images_tensor, batch_labels_tensor, batch_names))

        # Clean up batch data references to free memory immediately
        del batch_images, batch_labels, batch_names, batch_images_tensor, batch_labels_tensor

    # Clean up dataset references
    del dataset, dataset_loader

    return batches
