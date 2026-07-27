"""RAW data generation pipeline.

This module provides the main pipeline for generating noisy RAW images from
RGB inputs using the CycleISP RGB2RAW network with configurable noise models.
"""

from filecmp import cmp as filecmp
from itertools import combinations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import torch
from torch import nn
import torch.nn.functional as F

from scangen.formats.config_format import NoiseFormat

from scangen.pipeline.noise import (
    add_noise,
    random_noise_levels,
    set_noise_seed,
)
from scangen.pipeline.noise_d300 import pointwise_noise_levels
from scangen.pipeline.rgb2raw import Rgb2Raw

LOGGER = logging.getLogger("scangen.pipeline.generator")


class RAWGenerator:
    """Main RAW data generation pipeline."""
    model: Union[nn.DataParallel[Rgb2Raw],Rgb2Raw]

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        noise: NoiseFormat | None,
        output_dim: tuple[int,int],
        seed: int | None = None,
    ) -> None:
        """Initialize RAW generator.

        Args:
            model_path: Path to RGB2RAW model weights, such as `rgp2raw.pth`.
            device: A `torch.device`.
            noise: A configuration for the noise model.
            seed: An optional integer seed for the noise model.
        """

        self.device = device
        self.noise_config = noise
        self.output_dim = output_dim
        if seed is not None:
            set_noise_seed(seed)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model weights specified and default weights not found in {model_path}")
        model_path = model_path.resolve()
        self._load_model(model_path)


    def _load_model(self, model_path: Path) -> None:
        """Load model weights."""
        if not isinstance(self.device, torch.device):
            raise RuntimeError("The device in RAWGenerator should be a torch.device not {device}")
        self.model = Rgb2Raw()
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            LOGGER.debug("Removed 'module.' prefix from state dict keys")

        self.model.load_state_dict(state_dict, strict=False)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        # Use DataParallel only for GPU devices (not needed for CPU-only)
        if self.device.type != "cpu":
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    def generate_batch(
        self,
        rgb_images: torch.Tensor,
        labels: torch.Tensor,
        label_names: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Generate noisy RAW data from RGB batch.

        Args:
            rgb_images: RGB images tensor (B, 3, H, W)
            labels: Label indices tensor (B,)
            label_names: List of label names

        Returns:
            Tuple of (clean_raw, noisy_raw, metadata)
        """
        batch_size = rgb_images.shape[0]
        rgb_images = rgb_images.to(self.device)
        noise_config = self.noise_config

        with torch.no_grad():
            # Generate clean RAW data
            clean_raw = self.model(rgb_images)
            clean_raw = torch.clamp(clean_raw, 0, 1)

            # Resize before calculating noise so compression doesn't denoise.
            # Preserving corner alignment can cause artifacts.
            clean_resized = F.interpolate(clean_raw, size=self.output_dim, mode="bilinear", align_corners=False)
            LOGGER.debug(f"clean_raw {clean_raw.shape} resized {clean_resized.shape}")
            # Generate noise parameters based on config
            if noise_config is None:
                noisy_raw = clean_resized
            elif noise_config.type == "pointwise":
                fit_noise = pointwise_noise_levels(clean_resized, self.noise_config)
                noisy_raw = clean_resized + fit_noise
            elif noise_config.type == "cycleisp":
                shot_noise, read_noise = random_noise_levels(noise_config)
                shot_noise = shot_noise.to(self.device)
                read_noise = read_noise.to(self.device)

                # Add noise to clean RAW
                noisy_raw = add_noise(
                    clean_resized, shot_noise=shot_noise, read_noise=read_noise, device=self.device
                )
            else:
                msg = f"Unknown noise type: {noise_config.type}"
                raise ValueError(msg)

            noisy_raw = torch.clamp(noisy_raw, 0, 1)

        # Create batch metadata
        metadata = {
            "batch_size": batch_size,
            "labels": labels.tolist(),
            "label_names": label_names,
            "noise_type": "none" if not noise_config else noise_config.type,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
        }

        return clean_resized, noisy_raw, metadata
