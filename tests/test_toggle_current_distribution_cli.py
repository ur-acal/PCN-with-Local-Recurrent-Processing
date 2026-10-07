"""CLI metadata regressions for the toggle current-distribution diagnostic."""
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "diagnostic_scripts"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from diagnostic_config import MODEL_NAMES, MODEL_ROOTS  # noqa: E402
from diagnostic_runtime import _model_parameters, reference_corner_command  # noqa: E402
from plot_toggle_current_distributions import (  # noqa: E402
    dataset_description,
    default_output,
    parse_bound_percentiles,
    resolve_model_arguments,
)


def test_summary_percentiles_reject_nonfinite_values():
    import pytest
    with pytest.raises(ValueError):
        parse_bound_percentiles("nan")


def test_default_model_arguments_are_materialized_before_path_use():
    args = SimpleNamespace(
        model_name=None, model_dir=None, dataset_split="train", n_trials=1)
    resolve_model_arguments(args)
    assert args.model_name == MODEL_NAMES["cifar100"]
    assert Path(args.model_dir) == MODEL_ROOTS["cifar100"]
    assert default_output(args, ["FS_V2_T1"]).name.endswith(
        "_FS_V2_T1_1trials")


def test_dataset_description_uses_resolved_data_and_split():
    args = SimpleNamespace(
        model_name=MODEL_NAMES["cifar10"], model_dir=None,
        dataset_split="test", n_trials=1)
    resolve_model_arguments(args)
    assert dataset_description(args) == (
        "CiFAIR-10 test split with evaluation preprocessing")

    args.img_type = "scanGFI"
    assert dataset_description(args) == (
        "scanGFI CIFAR-10 test split with evaluation preprocessing")


def test_runtime_preserves_pinned_fixed_timing_parameters():
    args, _, _ = reference_corner_command(
        model_name=MODEL_NAMES["cifar100"],
        model_root=MODEL_ROOTS["cifar100"],
        corner="FS_V2_T1", batch_size=128, trial_index=0)
    ode_params, wrapper_params, _ = _model_parameters(args, trial_index=0)

    assert ode_params["toggle_timing_mode"] == "fixed"
    assert ode_params["toggle_y_time"] == 10e-9
    assert ode_params["z_over_y_time"] == 1.0
    assert ode_params["toggle_timing_R"] == 50e3
    assert ode_params["toggle_timing_C"] == 500e-15
    assert wrapper_params["adapt_relu_offset"] is True
