"""Separate grouped arithmetic correctness from full-model FP32 sensitivity."""
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ode_pc import ODEBlockPC
from tc_curve_groups import group_convolution, sample_groups
from test_tc_curve_groups import package
from test_tc_unified_backend import original_off


def fixture(empirical, top_n, transpose=False):
    original = package(empirical=empirical)
    values = {key: value.double() for key, value in vars(original).items()
              if torch.is_tensor(value) and value.is_floating_point()}
    if original.empirical_bank is not None:
        values['empirical_bank'] = {
            key: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in original.empirical_bank.items()}
    curves = replace(original, **values)
    cls = nn.ConvTranspose2d if transpose else nn.Conv2d
    module = cls(2, 2, 3, padding=1, bias=False).double()
    with torch.no_grad():
        module.weight.copy_(torch.arange(module.weight.numel()).reshape_as(module.weight).remainder(16)/15)
    owner = SimpleNamespace(v_dd=.1, q_hi=15, weight_scale=1.,
        _tc_curve_package=curves, _tc_grouped_convolution=True,
        _tc_fused_correction=False)
    for name in ('_values_to_level_idx', '_get_quant_magnitude_levels'):
        setattr(owner, name, MethodType(getattr(ODEBlockPC, name), owner))
    codes = owner._values_to_level_idx(module.weight.detach())
    draw = sample_groups(curves, codes, top_n, generator=torch.Generator().manual_seed(19))
    # Away from clamp/grid knots: a classical derivative exists here.
    inputs = torch.linspace(.011, .041, 18, dtype=torch.float64).reshape(1, 2, 3, 3)
    return owner, module, draw, inputs


@pytest.mark.parametrize('empirical', [False, True])
@pytest.mark.parametrize('top_n', [1, 3, 15])
@pytest.mark.parametrize('transpose', [False, True])
def test_float64_grouped_matches_independent_serial_expression(empirical, top_n, transpose):
    torch.set_num_threads(1)
    owner, module, draw, data = fixture(empirical, top_n, transpose)
    answers = []
    for backend in (original_off, group_convolution):
        inputs = data.clone().requires_grad_()
        output = backend(owner, module, inputs, draw)
        gradients = torch.autograd.grad(output.square().sum(), (inputs, module.weight))
        answers.append((output, *gradients))
    for actual, expected in zip(answers[1], answers[0]):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('empirical', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
def test_grouped_backward_matches_finite_differences(empirical, transpose):
    torch.set_num_threads(1)
    owner, module, draw, inputs = fixture(empirical, 3, transpose)
    class Corrected(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = module
        def forward(self, x):
            return group_convolution(owner, self.conv, x, draw)
    model = Corrected()
    def corrected(x, weight):
        # Keep the graph and fixed solve-local masks; check both derivatives.
        return torch.func.functional_call(model, {'conv.weight': weight}, (x,))
    assert torch.autograd.gradcheck(corrected, (inputs.requires_grad_(), module.weight),
                                    eps=1e-6, atol=1e-7, rtol=1e-5, fast_mode=True)


@pytest.mark.parametrize('empirical', [False, True])
def test_grouped_checkpoint_replay_preserves_values_and_gradients(empirical):
    torch.set_num_threads(1)
    owner, module, draw, data = fixture(empirical, 15)
    answers = []
    for replay in (False, True):
        inputs = data.clone().requires_grad_()
        def rhs(x):
            return group_convolution(owner, module, x, draw)
        output = checkpoint(rhs, inputs, use_reentrant=False) if replay else rhs(inputs)
        answers.append((output, *torch.autograd.grad(output.square().sum(), (inputs, module.weight))))
    for actual, expected in zip(answers[1], answers[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
