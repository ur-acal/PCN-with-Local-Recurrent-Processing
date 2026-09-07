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


def _write_mc_curve(directory):
    directory.mkdir()
    curve_path = directory / "tt_25_1.csv"
    curve_path.write_text(
        '"/Vout (VDD=1.2,temperature=25,mcparamset=1) X",'
        '"/Vout (VDD=1.2,temperature=25,mcparamset=1) Y"\n'
        "-0.6,0.6\n"
        "0.0,0.63\n"
        "0.6,1.1\n")
    return curve_path


def test_piecewise_linear_matches_scaled_table_and_midpoints(tmp_path):
    activation = PiecewiseLinearActivation(
        _write_curve(tmp_path), v_dd=0.1, corner="TT")
    x = torch.tensor([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])

    expected = torch.tensor([0.0, 0.0, 0.005, 0.01, 0.04, 0.07, 0.07])
    assert torch.allclose(activation(x), expected, atol=1e-7, rtol=1e-6)
    assert activation.uniform_grid


def test_fixed_output_reference_is_scoped_to_0906_relu_bank(tmp_path):
    new_curve = _write_mc_curve(tmp_path / "0906_RELU_Voltage")
    other_curve = _write_mc_curve(tmp_path / "another_absolute_vout_bank")

    new_activation = PiecewiseLinearActivation(
        new_curve, v_dd=0.5, corner="MC1")
    other_activation = PiecewiseLinearActivation(
        other_curve, v_dd=0.5, corner="MC1")

    assert torch.allclose(
        new_activation.curves["MC1"], torch.tensor([0.0, 0.03, 0.5]))
    assert torch.allclose(
        other_activation.curves["MC1"], torch.tensor([0.6, 0.63, 1.1]))


def test_0906_directory_loads_curves_from_every_file(tmp_path):
    curve_dir = tmp_path / "0906_RELU_Voltage"
    first_curve = _write_mc_curve(curve_dir)
    second_curve = curve_dir / "ff_-20_0.csv"
    second_curve.write_text(first_curve.read_text())

    activation = PiecewiseLinearActivation(
        curve_dir, v_dd=0.5, corner="TT_25_1_MC1")

    assert activation.corner_names == ("FF_-20_0_MC1", "TT_25_1_MC1")
    assert torch.allclose(
        activation.curves["FF_-20_0_MC1"], torch.tensor([0.0, 0.03, 0.5]))


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

    expected = (reference(beta_c * x) / beta_c).clamp(-0.1, 0.1)
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
        model, "random_per_forward", sharing="per_model") == 2
    model.train()
    with patch(
            "torch.randint",
            side_effect=(torch.tensor([3]), torch.tensor([4]),
                         torch.tensor([5]))) as randint:
        model(torch.tensor([0.0]))
        assert model.first.active_corner == model.second.active_corner
        assert model._last_measured_activation_corner == model.first.active_corner

        model.eval()
        model(torch.tensor([0.0]))
        eval_corner = model.first.corner_names[4]
        assert model.first.active_corner == eval_corner
        assert model.second.active_corner == eval_corner
        assert model._last_measured_activation_corner == eval_corner

        model(torch.tensor([0.0]))
        next_eval_corner = model.first.corner_names[5]
        assert model.first.active_corner == next_eval_corner
        assert model.second.active_corner == next_eval_corner
        assert next_eval_corner != eval_corner
    assert randint.call_count == 3


def test_random_per_forward_defaults_to_one_curve_per_layer():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = PiecewiseLinearActivation(ALL_CURVES, 0.1, "TT")
            self.second = PiecewiseLinearActivation(ALL_CURVES, 0.1, "TT")

        def forward(self, x):
            return self.second(self.first(x))

    model = Toy().train()
    configure_measured_activation_corner_mode(model, "random_per_forward")
    with patch("torch.randint", return_value=torch.tensor([1, 2])):
        model(torch.tensor([0.0]))

    assert model.first.active_corner == model.first.corner_names[1]
    assert model.second.active_corner == model.second.corner_names[2]
    assert model._last_measured_activation_corner == (
        model.first.corner_names[1], model.second.corner_names[2])


def test_random_per_forward_supports_per_spin_training():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = PiecewiseLinearActivation(
                ALL_CURVES, 0.1, "TT")

        def forward(self, x):
            return self.activation(x)

    model = Toy().train()
    configure_measured_activation_corner_mode(
        model, "random_per_forward", sharing="per_spin")
    samples = iter((
        torch.zeros((1, 3, 4, 4), dtype=torch.long),
        torch.ones((1, 3, 4, 4), dtype=torch.long),
        torch.full((1, 3, 4, 4), 2, dtype=torch.long),
    ))
    with patch("torch.randint", side_effect=lambda *args, **kwargs: next(samples)):
        model(torch.zeros(2, 3, 4, 4))

        assert model.activation.curve_sharing == "per_spin"
        assert model.activation._sampled_curve_indices.shape == (1, 3, 4, 4)

        model.eval()
        model(torch.zeros(2, 3, 4, 4))
        eval_indices = model.activation._sampled_curve_indices.clone()
        assert model.activation.curve_sharing == "per_spin"
        model(torch.zeros(2, 3, 4, 4))
        assert not torch.equal(
            model.activation._sampled_curve_indices, eval_indices)


def test_piecewise_linear_relu6_uses_same_scaling_modes(tmp_path):
    curve_path = _write_curve(tmp_path)
    original = MeasuredPiecewiseLinearReLU6Activation(
        curve_path, corner="TT", normalize_positive_endpoint=False)
    normalized = MeasuredPiecewiseLinearReLU6Activation(
        curve_path, corner="TT", normalize_positive_endpoint=True)

    assert torch.allclose(original(torch.tensor(6.0)), torch.tensor(4.2))
    assert torch.allclose(normalized(torch.tensor(6.0)), torch.tensor(6.0))
    assert torch.allclose(normalized(torch.tensor(9.0)), torch.tensor(6.0))
