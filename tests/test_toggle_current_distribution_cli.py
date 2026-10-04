"""CLI metadata regressions for the toggle current-distribution diagnostic."""
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "diagnostic_scripts"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from diagnostic_config import MODEL_NAMES, MODEL_ROOTS  # noqa: E402
from plot_toggle_current_distributions import (  # noqa: E402
    dataset_description,
    default_output,
    resolve_model_arguments,
)


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
