"""Negative roundoff must not turn physically nonnegative noise variances into NaNs."""
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import torch
from torch import nn

from ode_pc import ODEBlockPC, ToggleAveragedPhysicalFFFB
from physical_feedforward_tc import TCPhysicalBasicBlock


def check_tc_nominal_sum_guards_roundoff(kind):
    source = torch.ones(1, 1, 1, 3)
    raw = torch.tensor([[[[-1.49e-8, 0., 2.5]]]])
    if kind == 'sparse':
        module = SimpleNamespace(
            mat=torch.eye(3).to_sparse_csr(),
            meta=dict(padding=0, ker_h=1, ker_w=1, stride=1, out_chan=1))
        target, result = 'torch.sparse.mm', raw.reshape(3, 1)
    else:
        cls = nn.Conv2d if kind == 'conv' else nn.ConvTranspose2d
        module = cls(1, 1, 1, bias=False)
        target = 'ode_pc.F.conv2d' if kind == 'conv' else 'ode_pc.F.conv_transpose2d'
        result = raw
    with patch(target, return_value=result):
        actual = ODEBlockPC._tc_nominal_sum(None, module, source)
    assert torch.equal(actual, torch.tensor([[[[0., 0., 2.5]]]]))
    assert torch.isfinite(actual.sqrt()).all()
    assert TCPhysicalBasicBlock._tc_nominal_sum is ODEBlockPC._tc_nominal_sum


def check_toggle_level2_guards_roundoff_without_changing_valid_noise():
    state = torch.zeros(1, 1, 1, 3)
    levels = torch.tensor([[[[-1.49e-8, 0., 2.5]]]])
    seed = 37
    block = SimpleNamespace(
        q_hi=15, coupler_noise_p=.6e-12,
        _stage_capacitance=lambda stage: 500e-15,
        _coupler_noise_generator=lambda state: torch.Generator().manual_seed(seed))
    actual = ToggleAveragedPhysicalFFFB._averaged_coupler_brownian_increment(
        block, state, 5e-9, 'z', levels)
    normal = torch.randn(state.shape, generator=torch.Generator().manual_seed(seed))
    expected = (.6e-12 / 500e-15) * (
        torch.as_tensor(5e-9) * levels.clamp_min(0.) / 15).sqrt() * normal
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[..., :2], torch.zeros_like(actual[..., :2]))


class ConductanceRoundoffTests(unittest.TestCase):
    def test_tc_paths(self):
        for kind in ('conv', 'transpose', 'sparse'):
            with self.subTest(kind=kind):
                check_tc_nominal_sum_guards_roundoff(kind)

    def test_toggle_level2(self):
        check_toggle_level2_guards_roundoff_without_changing_valid_noise()
