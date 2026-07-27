"""RAW image format utilities and data conversion functions.

This module provides functions for converting between different RAW image formats,
saving/loading RAW data, and handling Bayer pattern conversions.

Adapted from CycleISP utilities for scangen project.

"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage import img_as_ubyte  # type: ignore[attr-defined]

LOGGER = logging.getLogger("scangen.raw_formats")


def unpack_raw(packed_raw: torch.Tensor) -> torch.Tensor:
    """Convert packed RGGB channels to full Bayer pattern.

    Takes RAW data in packed RGGB format (4 channels, each H/2 x W/2) and
    reconstructs the full Bayer pattern (single channel, H x W) with alternating
    Red-Green-Green-Blue pixel arrangement.

    Args:
        packed_raw: Packed RAW tensor of shape (B, 4, H//2, W//2)
            where channels are [Red, Green_Red, Green_Blue, Blue]

    Returns:
        torch.Tensor: Unpacked Bayer pattern of shape (B, 1, H, W)

    Examples:
        >>> packed = torch.randn(1, 4, 64, 64)  # RGGB channels
        >>> bayer = unpack_raw(packed)
        >>> bayer.shape
        torch.Size([1, 1, 128, 128])

    Notes:
        The Bayer pattern arrangement is:
        R G R G ...
        G B G B ...
        R G R G ...
        G B G B ...
    """
    batch_size, channels, h, w = packed_raw.shape
    assert channels == 4, f"Expected 4 RGGB channels, got {channels}"

    # Output dimensions
    H, W = h * 2, w * 2

    # Create full-size Bayer pattern tensor
    bayer = torch.zeros((batch_size, H, W), dtype=packed_raw.dtype, device=packed_raw.device)

    # Place RGGB channels in Bayer pattern positions
    bayer[:, 0:H:2, 0:W:2] = packed_raw[:, 0, :, :]  # Red: even rows, even cols
    bayer[:, 0:H:2, 1:W:2] = packed_raw[:, 1, :, :]  # Green_Red: even rows, odd cols
    bayer[:, 1:H:2, 0:W:2] = packed_raw[:, 2, :, :]  # Green_Blue: odd rows, even cols
    bayer[:, 1:H:2, 1:W:2] = packed_raw[:, 3, :, :]  # Blue: odd rows, odd cols

    # Add channel dimension
    return bayer.unsqueeze(1)


def pack_raw(bayer: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert full Bayer pattern to packed RGGB channels.

    Takes a full Bayer pattern image and packs it into 4 separate RGGB channels
    by extracting alternating pixels according to Bayer pattern positions.

    Args:
        bayer: Bayer pattern image of shape (H, W) or (H, W, 1)
               Can be torch.Tensor or numpy.ndarray

    Returns:
        Packed RGGB tensor/array of shape (H//2, W//2, 4) for numpy
        or (4, H//2, W//2) for torch

    Examples:
        >>> bayer_np = np.random.rand(128, 128)
        >>> packed = pack_raw(bayer_np)
        >>> packed.shape
        (64, 64, 4)

        >>> bayer_torch = torch.randn(128, 128)
        >>> packed = pack_raw(bayer_torch)
        >>> packed.shape
        torch.Size([4, 64, 64])
    """
    if isinstance(bayer, torch.Tensor):
        return _pack_raw_torch(bayer)
    return _pack_raw_numpy(bayer)


def _pack_raw_numpy(bayer: np.ndarray) -> np.ndarray:
    """Pack numpy Bayer pattern to RGGB channels."""
    if bayer.ndim == 3:
        bayer = bayer[:, :, 0]  # Remove channel dimension if present

    H, W = bayer.shape

    # Extract RGGB channels from Bayer pattern
    red = bayer[0:H:2, 0:W:2]  # Even rows, even cols
    green_red = bayer[0:H:2, 1:W:2]  # Even rows, odd cols
    green_blue = bayer[1:H:2, 0:W:2]  # Odd rows, even cols
    blue = bayer[1:H:2, 1:W:2]  # Odd rows, odd cols

    # Stack channels in RGGB order
    return np.stack([red, green_red, green_blue, blue], axis=-1)


def _pack_raw_torch(bayer: torch.Tensor) -> torch.Tensor:
    """Pack torch Bayer pattern to RGGB channels."""
    if bayer.ndim == 3:
        bayer = bayer.squeeze(0)  # Remove batch/channel dimension if present
    elif bayer.ndim == 4:
        bayer = bayer.squeeze(0).squeeze(0)  # Remove batch and channel

    H, W = bayer.shape

    # Extract RGGB channels from Bayer pattern
    red = bayer[0:H:2, 0:W:2]  # Even rows, even cols
    green_red = bayer[0:H:2, 1:W:2]  # Even rows, odd cols
    green_blue = bayer[1:H:2, 0:W:2]  # Odd rows, even cols
    blue = bayer[1:H:2, 1:W:2]  # Odd rows, odd cols

    # Stack channels in RGGB order
    return torch.stack([red, green_red, green_blue, blue], dim=0)


def save_raw_dict(data_dict: dict[str, Any], filepath: str | Path) -> None:
    """Save RAW data dictionary to pickle file.

    Saves a dictionary containing RAW image data and metadata to a pickle file.
    Commonly includes 'clean', 'noisy', 'variance', 'shot_noise', 'read_noise'.

    Args:
        data_dict: Dictionary with RAW data and metadata
        filepath: Output file path (should end in .pkl)

    Examples:
        >>> raw_dict = {
        ...     "clean": np.random.rand(4, 64, 64),
        ...     "noisy": np.random.rand(4, 64, 64),
        ... }
        >>> save_raw_dict(raw_dict, "output.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("wb") as f:
        pickle.dump(data_dict, f)


def load_raw_dict(filepath: str | Path) -> dict[str, Any]:
    """Load RAW data dictionary from pickle file.

    Loads a dictionary containing RAW image data and metadata from a pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Dictionary with RAW data and metadata

    Examples:
        >>> raw_dict = load_raw_dict("input.pkl")
        >>> clean_raw = raw_dict["clean"]  # Shape: (4, H//2, W//2)
    """
    filepath = Path(filepath)
    with filepath.open("rb") as f:
        return pickle.load(f)


def save_raw_png(
    img_data: torch.Tensor | np.ndarray,
    filepath: str | Path,
    packed: bool = True,
    normalize: bool = False,
) -> None:
    """Save RAW data as PNG image for visualization.

    Converts RAW data to 8-bit PNG format for inspection and debugging.
    Handles both packed RGGB format and unpacked Bayer pattern.

    Args:
        img_data: RAW image data to save
        filepath: Output PNG file path
        packed: Whether data is in packed RGGB format (True) or Bayer pattern (False)
        normalize: Whether to apply min-max normalization (False matches CycleISP behavior)

    Examples:
        >>> packed_raw = torch.randn(4, 64, 64)  # RGGB format
        >>> save_raw_png(packed_raw, "output.png", packed=True)

        >>> bayer_raw = torch.randn(1, 128, 128)  # Bayer pattern
        >>> save_raw_png(bayer_raw, "output.png", packed=False)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if needed
    if isinstance(img_data, torch.Tensor):
        raw_data = img_data.detach().cpu().numpy()
    else:
        raw_data = img_data
    assert isinstance(raw_data, np.ndarray)

    # Convert to Bayer pattern if packed
    if packed:
        if raw_data.ndim == 4 and raw_data.shape[1] == 4:
            # Handle batch dimension (B, 4, H//2, W//2) - take first sample
            raw_data = raw_data[0]  # Shape: (4, H//2, W//2)
            raw_data = _pack_to_bayer_numpy(raw_data)
        elif raw_data.ndim == 3 and raw_data.shape[0] == 4:
            # Already correct shape (4, H//2, W//2)
            raw_data = _pack_to_bayer_numpy(raw_data)
        else:
            msg = (
                f"Packed format expected shape (4, H//2, W//2) or (B, 4, H//2, W//2), "
                f"got {raw_data.shape}"
            )
            raise ValueError(msg)
    # Already in Bayer pattern, just squeeze extra dimensions
    elif raw_data.ndim == 3:
        raw_data = raw_data.squeeze()
    elif raw_data.ndim == 4:
        raw_data = raw_data.squeeze().squeeze()

    # Apply normalization only if requested
    if normalize:
        # Min-max normalization to [0, 1] range
        raw_min, raw_max = raw_data.min(), raw_data.max()
        if raw_max > raw_min:
            raw_data = (raw_data - raw_min) / (raw_max - raw_min)
        else:
            raw_data = np.zeros_like(raw_data)
    # else: Keep original values (matching CycleISP behavior)

    # Convert to 8-bit
    raw_data = img_as_ubyte(raw_data)

    # Save as PNG
    Image.fromarray(raw_data).save(filepath)


def _pack_to_bayer_numpy(packed_rggb: np.ndarray) -> np.ndarray:
    """Convert packed RGGB to Bayer pattern for numpy arrays."""
    channels, h, w = packed_rggb.shape
    assert channels == 4, f"Expected 4 RGGB channels, got {channels}"

    # Create full Bayer pattern
    H, W = h * 2, w * 2
    bayer = np.zeros((H, W), dtype=packed_rggb.dtype)

    # Place RGGB channels in correct positions
    bayer[0:H:2, 0:W:2] = packed_rggb[0, :, :]  # Red
    bayer[0:H:2, 1:W:2] = packed_rggb[1, :, :]  # Green_Red
    bayer[1:H:2, 0:W:2] = packed_rggb[2, :, :]  # Green_Blue
    bayer[1:H:2, 1:W:2] = packed_rggb[3, :, :]  # Blue

    return bayer


def load_image(filepath: str | Path) -> np.ndarray:
    """Load RGB image and convert to float32 [0, 1] range.

    Loads an image file and converts it to numpy array with values in [0, 1].
    Handles common image formats (PNG, JPG, etc.).

    Args:
        filepath: Path to image file

    Returns:
        Image array of shape (H, W, C) with float32 values in [0, 1]

    Examples:
        >>> img = load_image("input.png")
        >>> img.shape  # (height, width, channels)
        (256, 256, 3)
        >>> img.dtype
        dtype('float32')
        >>> 0 <= img.max() <= 1
        True
    """
    img = Image.open(filepath)
    imgf32 = np.array(img).astype(np.float32)

    # Normalize to [0, 1]
    if imgf32.max() > 1.0:
        imgf32 = imgf32 / 255.0

    return imgf32


def is_raw_file(filename: str | Path) -> bool:
    """Check if filename has RAW file extension.

    Args:
        filename: File name or path to check

    Returns:
        True if file has .pkl extension (RAW data format)
    """
    return Path(filename).suffix.lower() == ".pkl"


def is_image_file(filename: str | Path) -> bool:
    """Check if filename has image file extension.

    Args:
        filename: File name or path to check

    Returns:
        True if file has image extension (.png, .jpg, .jpeg)
    """
    image_extensions = {".png", ".jpg", ".jpeg"}
    return Path(filename).suffix.lower() in image_extensions


def get_bayer_pattern_info(pattern: str = "rggb") -> dict[str, tuple[int, int]]:
    """Get Bayer pattern channel positions.

    Args:
        pattern: Bayer pattern string ('rggb', 'bggr', 'grbg', 'gbrg')

    Returns:
        Dictionary mapping channel names to (row_offset, col_offset) positions

    Examples:
        >>> info = get_bayer_pattern_info("rggb")
        >>> info["red"]
        (0, 0)
        >>> info["blue"]
        (1, 1)
    """
    patterns = {
        "rggb": {"red": (0, 0), "green1": (0, 1), "green2": (1, 0), "blue": (1, 1)},
        "bggr": {"blue": (0, 0), "green1": (0, 1), "green2": (1, 0), "red": (1, 1)},
        "grbg": {"green1": (0, 0), "red": (0, 1), "blue": (1, 0), "green2": (1, 1)},
        "gbrg": {"green1": (0, 0), "blue": (0, 1), "red": (1, 0), "green2": (1, 1)},
    }

    pattern = pattern.lower()
    if pattern not in patterns:
        msg = f"Unknown Bayer pattern '{pattern}'. Supported: {list(patterns.keys())}"
        raise ValueError(msg)

    return patterns[pattern]
