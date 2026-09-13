"""Independent multi-batch/channel RHS audit (no production changes)."""
import unittest
from copy import deepcopy
import torch
from torch import nn
from switch import ODEXInitFFFBPixelSwitchStrang as Strang


def make(channels, mid, m, transpose=False):
    block = Strang.__new__(Strang)
    nn.Module.__init__(block)
    block.block_size = m
    block.scale_RHS = False
    cls = nn.ConvTranspose2d if transpose else nn.Conv2d
    block.FBconv = cls(channels, mid, 3, padding=1, bias=True).double()
    block.FFconv = nn.Conv2d(mid, channels, 3, padding=1, bias=True).double()
    block.fb_kh = block.fb_kw = block.ff_kh = block.ff_kw = 3
    block.fb_pad_h = block.fb_pad_w = block.ff_pad_h = block.ff_pad_w = 1
    block.FBconv_copies = [deepcopy(block.FBconv) for _ in range(9)]
    block.block_FBconv_copies = [deepcopy(block.FBconv) for _ in range((m+2)**2)]
    block.act_fn = nn.Hardtanh(-.17, .21)
    return block


class CoupledBatchAudit(unittest.TestCase):
    @torch.no_grad()
    def test_multibatch_channels_expansion_and_edge_rhs(self):
        torch.set_num_threads(1)
        torch.manual_seed(773)
        max_error = 0.
        # Include expansion geometry, unequal channel dimensions, 128 samples,
        # and non-divisible edges. Bias/non-zero f(0) stress padding handling.
        for transpose in (False, True):
            for channels, mid in ((4, 24), (24, 48), (48, 96)):
                y = torch.randn(128, channels, 7, 9, dtype=torch.float64)
                for m in (1, 2, 3, 4):
                    b = make(channels, mid, m, transpose)
                    full = b.FFconv(b.act_fn(b.FBconv(y)))
                    tiles = b.iter_spatial_blocks(7, 9)
                    for tile in (tiles[0], tiles[len(tiles)//2], tiles[-1]):
                        i,j,k,l = tile
                        local = b._make_coupled_block_ode_fn(tile)(0., y)
                        expected = torch.zeros_like(y)
                        expected[:,:,i:k,j:l] = full[:,:,i:k,j:l]
                        err = float((local-expected).abs().max())
                        max_error = max(max_error, err)
                        torch.testing.assert_close(local, expected, atol=1e-12, rtol=1e-12)
                        # Output sample 0 cannot depend on sample 127.
                        changed = y.clone()
                        changed[-1].add_(7.)
                        modified = b._make_coupled_block_ode_fn(tile)(0., changed)
                        self.assertTrue(torch.equal(local[:-1], modified[:-1]))
        print('B128 multi-channel RHS maximum absolute error:', max_error)


if __name__ == '__main__':
    unittest.main()
