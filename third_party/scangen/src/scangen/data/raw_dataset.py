"""PyTorch Dataset wrapper for real-time RAW data generation from RGB datasets.

This module provides the RawDataset class that wraps any RGB dataset and generates
RAW sensor data on-the-fly using the CycleISP RGB2RAW pipeline with configurable
noise models.
"""
from collections.abc import Iterable, Mapping, Sized
import logging
from pathlib import Path
from typing import cast, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from scangen.data.paths import data_directory
from scangen.device import get_device
from scangen.formats.config_format import NoiseFormat
from scangen.pipeline.generator import RAWGenerator

LOGGER = logging.getLogger("scangen.data.raw_dataset")


class RawDataset(Dataset):
    """PyTorch Dataset that generates RAW data from RGB datasets in real-time.

    Wraps any RGB dataset (CIFAR-10, ImageNet, etc.) and converts RGB images to
    realistic RAW sensor data using the CycleISP RGB2RAW model with configurable
    noise simulation.

    Examples:
        >>> from torchvision.datasets import CIFAR10
        >>> from scangen.data import RawDataset
        >>> # Load base RGB dataset
        >>> rgb_dataset = CIFAR10(root="./data", train=True, download=True)
        >>> # Wrap with RAW generation
        >>> raw_dataset = RawDataset(
        ...     rgb_dataset=rgb_dataset, noise_config={"type": "pointwise"}, target_size=(256, 256)
        ... )
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(raw_dataset, batch_size=32, shuffle=True)
        >>> for rgb, clean_raw, noisy_raw, metadata in dataloader:
        ...     # Train your model here
        ...     pass
    """
    # The input is (batch, 3 channels, 256, 256).
    CYCLE_ISP_INPUT: tuple[int,int] = (256, 256)

    def __init__(
        self,
        rgb_dataset: Dataset,
        noise_config: NoiseFormat,
        model_path: Path,
        target_size: tuple[int, int],
        device: torch.device,
    ) -> None:
        """Initialize RawDataset wrapper.

        Args:
            rgb_dataset: Any PyTorch Dataset that returns (image, label) or (image, label, ...)
            noise_config: Noise configuration dict, e.g. {'type': 'pointwise'}
            model_path: Path to RGB2RAW model weights. If None, uses default weights.
            target_size: Target size for RAW images as (height, width).
                Becomes (batch, 4 channels, height, width).
            device: Device for computation ('cpu', 'cuda', 'mps', 'auto'). If None, auto-detected.

        Raises:
            ValueError: If noise_config is invalid
            FileNotFoundError: If model_path is specified but doesn't exist
        """
        if isinstance(rgb_dataset, Dataset):
            self.rgb_dataset = rgb_dataset
        else:
            raise ValueError(f"rgb_dataset should be a Dataset not {type(rgb_dataset)}")
        if isinstance(target_size, Iterable) and len(target_size) == 2:
            self.target_size = target_size
        else:
            raise ValueError(f"target_size should be (height, width) not {target_size}")
        self.target_size = target_size
        if isinstance(noise_config, Mapping):
            self.noise_config = NoiseFormat.model_validate(noise_config)
        elif isinstance(noise_config, NoiseFormat):
            self.noise_config = noise_config
        else:
            raise ValueError(f"NoiseConfig should be a NoiseFormat or Dict not {type(noise_config)}")
        if isinstance(model_path, str):
            self._model_path = Path(model_path)
        elif isinstance(model_path, Path):
            self._model_path = model_path
        else:
            raise ValueError(f"ModelPath should be a Path or str not {model_path}")
        if not (self._model_path.exists() and self._model_path.is_file()):
            raise FileNotFoundError(f"ModelPath {self._model_path} not found.")

        self._model_path = model_path

        # Initialize RAW generator (lazy loading will happen on first access)
        self._generator = None
        self._device = device  # Use auto-detection if not specified

        data_cnt = len(cast(Sized, rgb_dataset))
        LOGGER.info(
            f"Created RawDataset wrapper for {data_cnt} samples, "
            f"target_size={target_size}, noise_type={self.noise_config.type}"
        )

    def _get_generator(self):
        """Lazy initialization of RAWGenerator."""
        if self._generator is None:
            LOGGER.debug(
                f"Initializing RAWGenerator with device: {self._device} path {self._model_path}"
            )
            self._generator = RAWGenerator(
                model_path=self._model_path,
                device=self._device,
                noise=self.noise_config,
                output_dim=self.target_size,
            )
            LOGGER.debug(f"RAWGenerator initialized on device: {self._generator.device}")
        return self._generator

    def _preprocess_rgb(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB image to RAW model input size.

        Args:
            rgb_image: RGB image tensor, shape (C, H, W) or (H, W, C)

        Returns:
            Preprocessed RGB tensor, shape (C, H, W) with target_size
        """
        # Ensure tensor is float and in [0, 1] range
        if rgb_image.dtype == torch.uint8:
            rgb_image = rgb_image.float() / 255.0

        # Ensure CHW format
        if rgb_image.shape[0] != 3:  # Assume HWC format
            rgb_image = rgb_image.permute(2, 0, 1)

        # Resize to target size using bicubic interpolation
        if rgb_image.shape[1:] != self.CYCLE_ISP_INPUT:
            rgb_image = F.interpolate(
                rgb_image.unsqueeze(0),  # Add batch dimension
                size=self.CYCLE_ISP_INPUT,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)  # Remove batch dimension

        # Clamp to [0, 1] range (bicubic interpolation can produce values outside this range)
        return torch.clamp(rgb_image, 0.0, 1.0)

    def __len__(self) -> int:
        """Return the length of the underlying RGB dataset."""
        return len(cast(Sized, self.rgb_dataset))

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Get a single sample with RAW data generation.

        Args:
            idx: Sample index

        Returns:
            Tuple of (rgb_image, clean_raw, noisy_raw, metadata)
            - rgb_image: Original RGB image, shape (3, H, W)
            - clean_raw: Clean RAW data, shape (4, H//2, W//2) in RGGB format
            - noisy_raw: Noisy RAW data, shape (4, H//2, W//2) in RGGB format
            - metadata: Dict with noise parameters and other info
        """
        # Get RGB sample from underlying dataset
        sample = self.rgb_dataset[idx]

        # Handle different dataset return formats
        if isinstance(sample, tuple | list):
            rgb_image = sample[0]
            # Keep additional data (labels, etc.) in metadata
            labels = sample[1] if len(sample) > 1 else ()
        else:
            rgb_image = sample
            labels = ()

        # Preprocess RGB image
        rgb_processed = self._preprocess_rgb(rgb_image)

        # Generate RAW data using the pipeline
        generator = self._get_generator()

        # Prepare inputs for generator (expects batch format)
        rgb_batch = rgb_processed.unsqueeze(0)  # Shape: (1, 3, H, W)
        labels_batch = torch.tensor([0])  # Dummy label for single sample
        label_names = ["sample"]  # Dummy label name

        # Ensure RGB data is on the correct device for the generator
        rgb_batch = rgb_batch.to(generator.device)

        # Generate clean and noisy RAW
        clean_raw_batch, noisy_raw_batch, batch_metadata = generator.generate_batch(
            rgb_images=rgb_batch,
            labels=labels_batch,
            label_names=label_names,
        )

        # Extract single sample from batch and move to CPU
        clean_raw = clean_raw_batch[0]  # Shape: (4, H//2, W//2)
        noisy_raw = noisy_raw_batch[0]  # Shape: (4, H//2, W//2)

        LOGGER.debug(f"clean_raw {clean_raw.shape}")

        # Prepare metadata
        metadata = {
            "noise_type": batch_metadata["noise_type"],
            "index": idx,
            "label": labels,  # Preserve labels, etc.
        }

        return rgb_processed, clean_raw, noisy_raw, metadata


class RawCIFARDataset(RawDataset):
    """
    Dataset that translates CIFAR into RAW.

    Args:
        root: pathlib.Path to the data directory, relative to package root or CWD.
        input_name: "cifar10" or "cifar100"
        download: True if should try to download the dataset if it isn't found.
        train: True to get dataset for training and validation, false for test.
        noise_config: A `NoiseFormat` to describe noise values.
        model_path: pathlib.Path to the weights for the RGB-to-RAW neural net.
        target_size: Output dimensions (16x16).
        device: A `torch.device` for computations.
    
    Returns:
        A dataset that is a `torch.utils.data.Dataset`.
    
    This is what should be in the `root` directory.
    ```
    data
    ├── cifar-10-batches-py
       ├── batches.meta
       ├── data_batch_1
       ├── data_batch_2
       ├── data_batch_3
       ├── data_batch_4
       ├── data_batch_5
       ├── readme.html
       └── test_batch
    ```
    And a `model_path` might be `data/weights/rgb2raw.pth`.
    """
    
    def __init__(
            self,
            root: Path, # Where dataset sits within SCANGENDIR.
            input_name: str, # "cifar10" or "cifar100"
            download: bool, # whether to try to download it.
            train: bool, # whether this is (train, validate) or test.
            noise_config: NoiseFormat,
            model_path: Path,
            target_size: tuple[int, int],
            device: torch.device,
            ):

        data_dir = data_directory()
        device = get_device(device)

        if input_name not in ("cifar10", "cifar100"):
            raise ValueError(f"RawCIFARDataset input name is cifar10 or cifar100 not {input_name}")
        # Create CIFAR-10 dataset with minimal transforms
        # Note: RawDataset will handle resizing and normalization
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL to tensor and scale to [0,1]
            ]
        )

        root = Path(root)
        if not root.is_absolute():
            root = data_dir / root
        dataget = CIFAR10 if input_name == "cifar10" else CIFAR100
        LOGGER.info(f"Loading {input_name} root={root} download={download} on device={device}")
        cifar_dataset = dataget(root=root, train=train, download=download, transform=transform)
        super().__init__(
            rgb_dataset=cifar_dataset,
            noise_config=noise_config,
            model_path=Path(model_path),
            target_size=target_size,
            device=device,
        )
