from pathlib import Path

import torch

from measured_activation import CubicBSplineActivation


ROOT = Path(__file__).resolve().parents[1]
CURRENT_CURVE = ROOT / "hardware_data" / "relu_current_0p2uA_finer.csv"
VOLTAGE_CURVE = ROOT / "hardware_data" / "relu_0p3mV.csv"


def test_auto_constrains_an_all_nonnegative_corner():
    unconstrained = CubicBSplineActivation(
        CURRENT_CURVE, v_dd=0.1, corner="TT", fit_constraint="none")
    automatic = CubicBSplineActivation(
        CURRENT_CURVE, v_dd=0.1, corner="TT", fit_constraint="auto")
    grid = torch.linspace(-0.1, 0.1, 20001)

    assert unconstrained.corner_fit_constraints["TT"] == "none"
    assert automatic.corner_fit_constraints["TT"] == "nonnegative"
    assert unconstrained(grid).min() < 0
    assert automatic(grid).min() >= 0
    assert torch.all(automatic.coefficients["TT"] >= 0)


def test_auto_leaves_an_entire_table_unconstrained_if_any_output_is_signed():
    activation = CubicBSplineActivation(
        VOLTAGE_CURVE, v_dd=0.1, corner="TT", fit_constraint="auto")

    assert set(activation.corner_fit_constraints.values()) == {"none"}


def test_explicit_modes_override_the_automatic_decision():
    unconstrained = CubicBSplineActivation(
        VOLTAGE_CURVE, v_dd=0.1, corner="TT", fit_constraint="none")
    constrained = CubicBSplineActivation(
        VOLTAGE_CURVE, v_dd=0.1, corner="TT", fit_constraint="nonnegative")

    assert set(unconstrained.corner_fit_constraints.values()) == {"none"}
    assert set(constrained.corner_fit_constraints.values()) == {"nonnegative"}
