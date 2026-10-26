"""Explicit legacy NPZ couplers retain independently configured TC pooling."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from measured_pooling import MeasuredAvgPool2d
from tc_empirical import is_npz_bank, prepare_curves
from test_tc_cli import resolved
from test_tc_inference import expanded


@pytest.fixture
def bank(tmp_path):
    grid = torch.tensor([-.1, 0., .1], dtype=torch.float64)
    nominal = 1e4/(torch.arange(1, 16, dtype=torch.float64)/15)
    curves = nominal[:, None, None]*(1+.01*torch.arange(4)[None, :, None]+.1*grid[None, None, :])
    path = tmp_path/'legacy.npz'
    np.savez(path, v_grid=grid.numpy(), programmed_resistances=nominal.numpy(), curves=curves.numpy())
    return path


def options(bank):
    root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(nonlinear_R_curve_sampling='empirical_with_replacement',
        tc_empirical_curve_bank=str(bank), R=1e4, measured_pooling_nominal_R=1e4,
        measured_pooling_curve_path=str(root/'hardware_data/res_vs_vin_10k_150k.csv'),
        nonlinear_R_table='/unused-for-couplers', nonlinear_R_curve_seed=12,
        tc_covariance_table=str(root/'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'),
        resistance_floor_ohms=None)


def load_package(bank):
    # Couplers must not quietly fit Gaussian curves in this mode.
    with patch('tc_nonidealities.prepare_tc_resistance_curves', side_effect=AssertionError):
        return prepare_curves('/ignored-mean', '/ignored-covariance',
            mode='empirical_with_replacement', empirical_path=str(bank),
            levels=torch.arange(16, dtype=torch.float64)/15, R=1e4, R_max=150000.,
            dtype=torch.float64, device='cpu')


def test_documented_launcher_explicitly_selects_npz(bank, monkeypatch):
    monkeypatch.delenv('NONLINEAR_R_CURVE_SAMPLING', raising=False)
    args = resolved('eval', NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                    TC_EMPIRICAL_CURVE_BANK=str(bank))
    assert args.nonlinear_R_curve_sampling == 'empirical_with_replacement'
    assert args.tc_empirical_curve_bank == str(bank)
    assert load_package(bank).empirical_bank['draws_per_code'] == 4
    # Keep the agreed explicit-selector contract: a path alone changes nothing.
    default = resolved('eval', TC_EMPIRICAL_CURVE_BANK=str(bank))
    assert default.nonlinear_R_curve_sampling == 'multivariate_gaussian'
    document = (Path(__file__).resolve().parents[1]/'docs/tc_empirical_bank.md').read_text()
    assert 'NONLINEAR_R_CURVE_SAMPLING=empirical_with_replacement' in document
    assert 'sampling automatically' not in document


def test_pcn_npz_expanded_convolution_and_measured_pooling_match_legacy(bank):
    from tc_cli import pooling_options
    args = options(bank)
    wrappers = [SimpleNamespace(ode_block=SimpleNamespace(FFconv=nn.Conv2d(1, 1, 1).double()))]
    selected = pooling_options(args, wrappers)
    control = copy.copy(args)
    control.nonlinear_R_curve_sampling = 'multivariate_gaussian'
    original = pooling_options(control, wrappers)
    assert selected['curve_path'] is None
    for name in ('v_grid', 'mean', 'factor'):
        torch.testing.assert_close(selected['curve_gaussian'][name], original['curve_gaussian'][name],
                                   rtol=0, atol=0)
    conv = expanded(load_package(bank)).eval()
    assert conv.nonlinear_R_curve_sampling == 'empirical_with_replacement'
    assignment = conv.nonlinear_R_curve_assignment.clone()
    actual_pool = MeasuredAvgPool2d(kernel_size=2, seed=19, **selected).eval()
    control_pool = MeasuredAvgPool2d(kernel_size=2, seed=19, **original).eval()
    x = torch.linspace(.01, .08, 18, dtype=torch.float64).reshape(2, 1, 3, 3)
    with torch.no_grad():
        output = conv(x)
        actual = actual_pool(output).flatten(1)
        expected = control_pool(output).flatten(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert torch.equal(assignment, conv.nonlinear_R_curve_assignment)


def test_cnn_npz_pooling_keeps_gaussian_mean_and_covariance(bank):
    from tc_feedforward_cli import configure_pooling
    args = options(bank)
    assert load_package(bank).empirical_bank is not None
    def model():
        net = nn.Module()
        net.weight = nn.Parameter(torch.ones(1, dtype=torch.float64))
        net.global_pool = nn.AdaptiveAvgPool2d(1)
        return net
    actual, control = model(), model()
    configure_pooling(actual, args)
    baseline = copy.copy(args)
    baseline.nonlinear_R_curve_sampling = 'multivariate_gaussian'
    configure_pooling(control, baseline)
    assert actual.global_pool.curve_gaussian
    x = torch.linspace(.01, .08, 32, dtype=torch.float64).reshape(2, 1, 4, 4)
    torch.testing.assert_close(actual.global_pool.eval()(x), control.global_pool.eval()(x), rtol=0, atol=0)


def test_npz_detection_does_not_misclassify_directory(tmp_path):
    directory = tmp_path/'measured.npz'
    directory.mkdir()
    assert not is_npz_bank(directory)
    assert not is_npz_bank(None)
