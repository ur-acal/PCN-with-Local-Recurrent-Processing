import torch
from scangen.data import rggb_to_rgb, write_rgb

def test_rggb_to_rgb(tmp_path):
    rgb = rggb_to_rgb(torch.rand(4, 8, 8) * 2 - 1)
    assert rgb.shape == (3, 16, 16)
    fn = tmp_path / "example.png"
    write_rgb(rgb, fn)
    assert fn.exists()
