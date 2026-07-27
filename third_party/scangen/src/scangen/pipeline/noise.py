"""Noise generation utilities for RAW image synthesis."""

import logging

import torch
import torch.distributions as dist
from scangen.formats.config_format import NoiseFormat

LOGGER = logging.getLogger("scangen.pipeline.noise")


def dnd_noise():
    return NoiseFormat.model_validate(
        {
            "type": "cycleisp",
            "min_shot_noise": 0.0001,
            "max_shot_noise": 0.012,
            "read_noise_offset_width": 0.26,
            "read_noise_offset": 1.20,
            "read_noise_slope": 2.18
        }
    )

def sidd_noise():
    return NoiseFormat.model_validate(
        {
            "type": "cycleisp",
            "min_shot_noise": 0.00068674,
            "max_shot_noise": 0.02194856,
            "read_noise_offset_width": 0.20,
            "read_noise_offset": 0.30,
            "read_noise_slope": 1.85
        }
    )

def random_noise_levels(config) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random noise levels following DND dataset characteristics.

    Creates shot and read noise parameters using the log-log linear distribution
    matching the DND (Darmstadt Noise Dataset) real-world noise characteristics.

    Look at "Unprocessing images for learned raw denoising" by Brooks et al,
    2019, for why this per-batch sampling of noise parameters happens.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Shot noise and read noise levels
            - shot_noise: Proportional to image brightness (Poisson-like)
            - read_noise: Independent additive noise (Gaussian-like)

    Examples:
        >>> shot, read = random_noise_levels(noise_config)
        >>> shot.shape, read.shape
        (torch.Size([1]), torch.Size([1]))
        >>> 0.0001 <= shot.item() <= 0.012
        True
    """
    log_min_shot_noise = torch.log10(torch.tensor([config.min_shot_noise]))
    log_max_shot_noise = torch.log10(torch.tensor([config.max_shot_noise]))

    # Sample shot noise from uniform distribution in log space
    shot_dist = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = shot_dist.sample()
    shot_noise = torch.pow(10, log_shot_noise)

    # Read noise follows empirical relationship with shot noise
    read_dist = dist.normal.Normal(torch.tensor([0.0]), torch.tensor([config.read_noise_offset_width]))
    read_noise_offset = read_dist.sample()

    # Linear relationship: log(read_noise) = 2.18 * log(shot_noise) + 1.20 + offset
    def line(x):
        return config.read_noise_slope * x + config.read_noise_offset

    log_read_noise = line(log_shot_noise) + read_noise_offset
    read_noise = torch.pow(10, log_read_noise)

    return shot_noise, read_noise

def random_noise_levels_dnd() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate noise with DND parameters."""
    return random_noise_levels(dnd_noise())


def random_noise_levels_sidd() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate noise with SIDD parameters."""
    return random_noise_levels(sidd_noise())


def add_noise(
    image: torch.Tensor,
    shot_noise: float | torch.Tensor = 0.01,
    read_noise: float | torch.Tensor = 0.0005,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Add realistic camera noise to RAW image data.

    Applies shot noise (Poisson-distributed, proportional to signal) and
    read noise (Gaussian-distributed, independent of signal) to simulate
    real camera sensor noise characteristics.

    They don't use the Poisson distribution but add it to the Gaussian
    variance, so that is replicated here.

    Args:
        image: Input RAW image tensor of any shape
        shot_noise: Shot noise coefficient (signal-dependent noise)
        read_noise: Read noise coefficient (signal-independent noise)
        device: Target device for computation (if None, uses image's device)

    Returns:
        torch.Tensor: Noisy image with same shape as input

    Notes:
        The noise model follows: noisy = image + N(0, sqrt(shot*max(image,0) + read))
        where N(μ, σ) represents a normal distribution.
        Negative image values are clamped to zero for variance calculation.
        They don't use the Poisson distribution but add it to the Gaussian
        variance, so that is replicated here.

    Examples:
        >>> clean = torch.rand(1, 4, 128, 128)  # RGGB RAW data
        >>> shot_noise, read_noise = random_noise_levels(noise_config)
        >>> noisy = add_noise(clean, shot_noise, read_noise)
        >>> noisy.shape == clean.shape
        True
        >>> torch.all(noisy != clean)  # Should have noise added
        True
    """
    # Determine target device
    if device is not None:
        target_device = device
    else:
        # Default: use same device as image
        target_device = image.device

    # Ensure noise parameters are tensors and move to target device
    if not isinstance(shot_noise, torch.Tensor):
        shot_noise = torch.tensor(shot_noise, device=target_device)
    else:
        shot_noise = shot_noise.to(target_device)

    if not isinstance(read_noise, torch.Tensor):
        read_noise = torch.tensor(read_noise, device=target_device)
    else:
        read_noise = read_noise.to(target_device)

    # Move image to target device
    image = image.to(target_device)

    # Calculate noise variance: shot component + read component
    # Clamp image to non-negative for shot noise calculation
    image_positive = torch.clamp(image, min=0.0)
    variance = image_positive * shot_noise + read_noise

    # Variance is per-pixel.
    variance = torch.clamp(variance, min=1e-8)

    # Generate noise from normal distribution N(0, sqrt(variance))
    mean = torch.zeros_like(variance)

    # The normally-distributed noise is per-pixel.
    distribution = dist.normal.Normal(mean, torch.sqrt(variance))
    noise = distribution.sample()

    # Add noise to original image
    return image + noise

def set_noise_seed(seed: int) -> None:
    """Set random seed for reproducible noise generation.

    Args:
        seed: Random seed value for PyTorch random number generator

    Examples:
        >>> set_noise_seed(42)
        >>> shot1, read1 = random_noise_levels_dnd()
        >>> set_noise_seed(42)
        >>> shot2, read2 = random_noise_levels_dnd()
        >>> torch.allclose(shot1, shot2) and torch.allclose(read1, read2)
        True
    """
    torch.manual_seed(seed)
