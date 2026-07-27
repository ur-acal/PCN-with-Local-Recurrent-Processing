import logging
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("scangen.simple_raw2rgb")


def rggb_to_rgb(rggb: torch.Tensor) -> torch.Tensor:
    """
    Poor man's conversion from RAW to RGB to see how it looks.

    Args:
        rggb: Tensor of shape (4, H, W) where channels are (R, G_r, G_b, B).
    
    Returns:
        RGB tensor of shape (3, 2*H, 2*W) with values in [0, 1]
    """
    raw_cnt, h, w = rggb.shape
    assert raw_cnt == 4
    r, g_r, g_b, b = rggb[0], rggb[1], rggb[2], rggb[3]
    
    # upsample
    r_up = torch.zeros(2 * h, 2 * w, dtype=rggb.dtype)
    g_up = torch.zeros(2 * h, 2 * w, dtype=rggb.dtype)
    b_up = torch.zeros(2 * h, 2 * w, dtype=rggb.dtype)

    # This defines it as RGGB not some other layout.
    r_up[0::2, 0::2] = r
    g_up[0::2, 1::2] = g_r
    g_up[1::2, 0::2] = g_b
    b_up[1::2, 1::2] = b

    r_kernel = torch.tensor([[0.25, 0.5, 0.25],
                             [0.5, 1.0, 0.5],
                             [0.25, 0.5, 0.25]], dtype=rggb.dtype).view(1, 1, 3, 3)
    g_kernel = torch.tensor([[0.0, 0.25, 0.0],
                             [0.25, 1.0, 0.25],
                             [0.0, 0.25, 0.0]], dtype=rggb.dtype).view(1, 1, 3, 3)
    
    # The padding means the edges and corners are suspect.
    rgb_r = F.conv2d(r_up.unsqueeze(0).unsqueeze(0), r_kernel, padding=1).squeeze()
    rgb_g = F.conv2d(g_up.unsqueeze(0).unsqueeze(0), g_kernel, padding=1).squeeze()
    # blue and red use same kernel.
    rgb_b = F.conv2d(b_up.unsqueeze(0).unsqueeze(0), r_kernel, padding=1).squeeze()
    return torch.stack([rgb_r, rgb_g, rgb_b])


def write_rgb(rgb: torch.Tensor, filename: Path):
    rgb_max = rgb.max()
    rgb_uint8 = (rgb * 255).clamp(0, 255).byte()
    img = Image.fromarray(rgb_uint8.permute(1, 2, 0).numpy())
    LOGGER.debug(f"Writing image to {filename}. Image had max {rgb_max} in [0, 1].")
    img.save(filename)
