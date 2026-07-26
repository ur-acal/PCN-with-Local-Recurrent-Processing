from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from measured_activation import (
    MeasuredPiecewiseLinearReLU6Activation,
    PiecewiseLinearActivation,
    configure_measured_activation_corner_mode,
)


ALL_CURVES = (Path(__file__).resolve().parents[1] / "hardware_data" /
              "relu_current_0p2uA_all.csv")
RAW_TT_COLUMN = (
    "/IPRB_OUT/PLUS (modelFiles=cln22ull_1d8_elk_v1d3_2p1_"
    "shrink0d855_embedded_usage_config.scs:top_tt,Vbn_value=0,"
    "VDD_VALUE=1,temperature=25) Y")

def _write_curve(tmp_path):
    curve_path = tmp_path / "curve.csv"
    curve_path.write_text(
        "Vin,Vout_TT,Vout_SS\n"
        "-0.3,0.0,-0.01\n"
        "0.0,0.03,0.02\n"
        "0.3,0.21,0.18\n")
    return curve_path


def test_piecewise_linear_matches_scaled_table_and_midpoints(tmp_path):
    activation = PiecewiseLinearActivation(
        _write_curve(tmp_path), v_dd=0.1, corner="TT")
    x = torch.tensor([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])

    expected = torch.tensor([0.0, 0.0, 0.005, 0.01, 0.04, 0.07, 0.07])
    assert torch.allclose(activation(x), expected, atol=1e-7, rtol=1e-6)
    assert activation.uniform_grid


def test_piecewise_linear_gradient_is_local_segment_slope(tmp_path):
    activation = PiecewiseLinearActivation(
        _write_curve(tmp_path), v_dd=0.1, corner="TT")
    x = torch.tensor([-0.2, -0.05, 0.05, 0.2], requires_grad=True)

    activation(x).sum().backward()

    assert torch.allclose(
        x.grad, torch.tensor([0.0, 0.1, 0.6, 0.0]),
        atol=1e-6, rtol=1e-6)


def test_endpoint_normalization_scales_entire_curve(tmp_path):
    curve_path = _write_curve(tmp_path)
    original = PiecewiseLinearActivation(
        curve_path, v_dd=0.1, corner="TT",
        normalize_positive_endpoint=False)
    normalized = PiecewiseLinearActivation(
        curve_path, v_dd=0.1, corner="TT",
        normalize_positive_endpoint=True)
    x = torch.tensor([0.0, 0.1, 0.2])
    expected_factor = 0.3 / 0.21

    assert torch.allclose(
        normalized(x), original(x) * expected_factor,
        atol=1e-7, rtol=1e-6)
    assert torch.allclose(normalized(torch.tensor(0.1)), torch.tensor(0.1))


def test_coordinate_pullback_matches_phi_beta_x_over_beta(tmp_path):
    curve_path = _write_curve(tmp_path)
    reference = PiecewiseLinearActivation(curve_path, v_dd=0.1, corner="TT")
    pullback = PiecewiseLinearActivation(curve_path, v_dd=0.1, corner="TT")
    beta_c = torch.tensor(0.4)
    x = torch.tensor([-0.15, 0.05, 0.2], requires_grad=True)

    expected = reference(beta_c * x) / beta_c
    pullback.set_coordinate_pullback_scale(beta_c)
    actual = pullback(x)
    actual.sum().backward()

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_curves_are_fixed_and_checkpoint_neutral(tmp_path):
    activation = PiecewiseLinearActivation(
        _write_curve(tmp_path), v_dd=0.1, corner="TT")

    assert list(activation.parameters()) == []
    assert activation.state_dict() == {}
    assert activation.corner_names == ("TT", "SS")
    activation.select_corner("SS")
    assert activation.active_corner == "SS"


def test_raw_all_corner_table_and_tt_alias():
    activation = PiecewiseLinearActivation(ALL_CURVES, v_dd=0.1, corner="TT")

    assert len(activation.corner_names) == 45
    assert activation.active_corner == "TT_VDD1_T25"
    assert activation.corner_columns[activation.active_corner] == RAW_TT_COLUMN
    activation.select_corner(RAW_TT_COLUMN)
    assert activation.active_corner == "TT_VDD1_T25"


def test_random_corner_is_shared_once_per_top_level_training_forward():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = PiecewiseLinearActivation(ALL_CURVES, 0.1, "TT")
            self.second = PiecewiseLinearActivation(ALL_CURVES, 0.1, "TT")

        def forward(self, x):
            first = self.first(x)
            assert self.first.active_corner == self.second.active_corner
            return self.second(first)

    model = Toy()
    assert configure_measured_activation_corner_mode(
        model, "random_per_forward") == 2
    model.train()
    with patch("torch.randint", return_value=torch.tensor([3])) as randint:
        model(torch.tensor([0.0]))
    assert randint.call_count == 1
    assert model.first.active_corner == model.second.active_corner
    assert model._last_measured_activation_corner == model.first.active_corner

    model.eval()
    model(torch.tensor([0.0]))
    assert model._last_measured_activation_corner is None
    assert model.first.active_corner == "TT_VDD1_T25"
    assert model.second.active_corner == "TT_VDD1_T25"


def test_piecewise_linear_relu6_uses_same_scaling_modes(tmp_path):
    curve_path = _write_curve(tmp_path)
    original = MeasuredPiecewiseLinearReLU6Activation(
        curve_path, corner="TT", normalize_positive_endpoint=False)
    normalized = MeasuredPiecewiseLinearReLU6Activation(
        curve_path, corner="TT", normalize_positive_endpoint=True)

    assert torch.allclose(original(torch.tensor(6.0)), torch.tensor(4.2))
    assert torch.allclose(normalized(torch.tensor(6.0)), torch.tensor(6.0))
    assert torch.allclose(normalized(torch.tensor(9.0)), torch.tensor(6.0))
