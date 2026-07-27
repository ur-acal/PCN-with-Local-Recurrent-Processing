from scangen.pipeline.noise_d300 import d300_noise

import pytest
import torch


def test_d300_sigma_zero():
    """Zero doesn't blow up."""
    raw = torch.zeros(4, 4)
    sigma = d300_noise(raw, 1.0)
    assert torch.isfinite(sigma).all()


@pytest.mark.skip
def test_d300_sigma_levels():
    raw = 0.9 * (torch.rand(3, 3) * 2 - 1)
    sigma = d300_noise(raw, 1.0)
    print((sigma / raw).max())
    assert ((sigma / raw) < 0.04).all()
