from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from inference_utils import replace_transpose_conv
from ode_pc import TogglePulseODEXInitFFFB, ToggleWrapper1State
from pc_conv import PCConvReLU6
from validation import MVMConv, Validator


class ToyPulseModel(nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        torch.manual_seed(23)
        pc = PCConvReLU6(inp_chan=2, out_chan=2, bypass=False, layer_idx=0)
        # Represent an already quantized best checkpoint, as in the launcher.
        with torch.no_grad():
            for conv in (pc.FFconv, pc.FBconv):
                conv.weight.copy_(torch.randint(-15, 16, conv.weight.shape) / 15.)
                conv.weight.flatten()[0] = 1.
        replace_transpose_conv(pc)
        block = TogglePulseODEXInitFFFB(
            pc_conv=pc, noise_level=0., t_end=1.75, toggle_n_cycles=5,
            odexinit_scaling_mode='direct', toggle_timing_mode='fixed',
            toggle_y_time=1e-8, z_over_y_time=1,
            toggle_timing_R=5e4, toggle_timing_C=5e-13)
        source = Path(__file__).resolve().parents[1] / 'hardware_data/mc_45_corners/coupler_full_range/fs_25_2.csv'
        self.wrappers = [ToggleWrapper1State(
            ode_block=block, R=5e4, R_max=None, C=5e-13, v_dd=.5, one_over_q=5,
            w_bits=5, enob=None, weight_quant_factor_bits=1,
            nonlinear_R=nonlinear, nonlinear_R_table=str(source),
            nonlinear_R_mc_quantity='conductance',
            nonlinear_R_curve_sharing='per_coupler',
            nonlinear_R_curve_sampling='empirical_with_replacement',
            nonlinear_R_curve_seed=4096)]
        self.PcConvs = nn.ModuleList([block])

    def forward(self, x):
        return self.PcConvs[0](x)


def expand(model, directory):
    x = torch.full((2, 2, 3, 3), .05)
    Validator(model, str(directory), 'cpu', [(x, torch.zeros(2, dtype=torch.long))],
              str(directory), wrapper=model.wrappers)
    return x


def test_both_pulse_matrices_expand_and_preserve_ideal_output(tmp_path):
    model = ToyPulseModel().eval()
    x = torch.full((2, 2, 3, 3), .05)
    with torch.no_grad():
        before = model(x)
        expand(model, tmp_path)
        block = model.PcConvs[0]
        assert isinstance(block.FFconv, MVMConv)
        assert isinstance(block.FBconv, MVMConv)
        assert not hasattr(block, '_pulse_unrolling')
        torch.testing.assert_close(model(x), before, atol=1e-6, rtol=1e-5)


def test_incomplete_cache_recovers_and_complete_cache_reuses(tmp_path):
    expand(ToyPulseModel(), tmp_path)
    cache, = tmp_path.glob('cache_*/expanded_weights_0.pth')
    data = torch.load(cache, weights_only=False)
    assert set(data) == {'FFconv', 'FBconv'}
    del data['FFconv']  # Historical buggy cache.
    torch.save(data, cache)
    model = ToyPulseModel()
    expand(model, tmp_path)
    assert set(torch.load(cache, weights_only=False)) == {'FFconv', 'FBconv'}
    with patch('validation.conv2d_to_matrix_fixed_padding', side_effect=AssertionError('must reuse cache')):
        expand(ToyPulseModel(), tmp_path)


def test_both_sides_receive_per_coupler_curves(tmp_path):
    model = ToyPulseModel(nonlinear=True)
    expand(model, tmp_path)
    for conv in (model.PcConvs[0].FFconv, model.PcConvs[0].FBconv):
        assert isinstance(conv, MVMConv)
        assert conv.csv_enabled
        assert conv.nonlinear_R_curve_sharing == 'per_coupler'
        assert conv.nonlinear_R_curve_assignment.numel() == conv.mat.values().numel()
        assert conv.nonlinear_R_curve_assignment.unique().numel() > 1
