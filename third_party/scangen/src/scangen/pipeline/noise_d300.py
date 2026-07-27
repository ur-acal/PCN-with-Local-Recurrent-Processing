from functools import cache
import logging

import numpy as np
import torch

LOGGER = logging.getLogger("scangen.noise_d300")
NOISE_BITS = 12

# Signal and Noise data
# These come from measurements on a D300 camera.
D300_SIGNAL, D300_NOISE = np.array([
    (0.5, 1.50),
    (1.0, 1.67),
    (1.5, 1.85),
    (2.0, 2.02),
    (2.5, 2.18),
    (3.0, 2.34),
    (3.5, 2.48),
    (4.0, 2.61),
    (4.5, 2.76),
    (5.0, 2.91),
    (5.5, 3.06),
    (6.0, 3.25),
    (6.5, 3.42),
    (7.0, 3.62),
    (7.5, 3.85),
    (8.0, 4.06),
    (8.5, 4.31),
    (9.0, 4.52),
    (9.5, 4.76),
    (10.0, 5.01),
    (10.5, 5.25),
    (11.0, 5.50),
    (11.5, 5.76),
    (12.0, 6.01),
    (12.5, 6.29),
    (13.0, 6.50),
    (13.5, 6.83),
    (14.0, 7.07),
]).T


@cache
def d300_noise_coefficients():
    return np.polyfit(D300_SIGNAL, D300_NOISE, 3)


def d300_noise(input_tensor, max_value):
    # zeros will cause Inf and then NaN values.
    clipped = torch.clamp(input_tensor, min=1e-10, max=max_value)
    logsignal = NOISE_BITS + torch.log2(clipped / max_value)
    # Spell out coefficients so we are working with Torch operators.
    c0, c1, c2, c3 = d300_noise_coefficients()
    lognoise = ((c0 * logsignal + c1) * logsignal + c2) * logsignal + c3
    # 14 bits total size. Shot noise parameter is in log2-space.
    return torch.randn_like(clipped) * max_value * 2.0**(lognoise - NOISE_BITS)


def pointwise_noise_levels(input_tensor, _config) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a pointwise noise that comes from a D300 sensor.

    Args:
        input_tensor: A Torch.tensor of image data.
        config: A NoiseConfig.
    
    Returns:
        Noise to add to the input tensor.
    """
    return d300_noise(input_tensor, input_tensor.max())
