"""Shared defaults for the repository's standalone diagnostic scripts."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MC45_ROOT = REPO_ROOT / "hardware_data" / "mc_45_corners"

MODEL_NAMES = {
    "cifar10": (
        "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
        "ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_"
        "128BS_0.01LR_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_"
        "srrlDistill_a0p3_t2p0_CiFAIR_1REP"
    ),
    "cifar100": (
        "TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
        "ToggleODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_"
        "128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_"
        "srrlDistill_a0p3_t2p0_CiFAIR_1REP"
    ),
}

MODEL_ROOTS = {
    "cifar10": REPO_ROOT / "saved_ckpt_runs" / (
        "coupler_v2_CiFAIR10_qf1_noENOB_matchdistill_srrl_noRE_"
        "ftlr0p002_e200_toggle_odexinit"),
    "cifar100": REPO_ROOT / "saved_ckpt_runs" / (
        "coupler_v2_CiFAIR100_qf1_noENOB_matchdistill_srrl_noRE_"
        "toggle_odexinit"),
}

# Up-to-date 45-corner reference configuration shared by the diagnostics.
R_OHM = 50e3
C_FARAD = 500e-15
T_END = 1.75
N_CYCLES = 5
W_BITS = 5
WEIGHT_QUANT_FACTOR_BITS = 1
V_DD = 0.1
COUPLER_SOURCE = MC45_ROOT / "coupler_monte_v2"
RELU_SOURCE = MC45_ROOT / "relu_monteCarlo"


def normalize_task(task):
    key = str(task).strip().lower().replace("-", "")
    aliases = {
        "cifar10": "cifar10",
        "c10": "cifar10",
        "cifar100": "cifar100",
        "c100": "cifar100",
    }
    if key not in aliases:
        raise ValueError("task must be cifar10 or cifar100")
    return aliases[key]


def infer_model_data(model_name):
    """Infer the task and input representation encoded in a model name."""
    model_name = str(model_name)
    lowered = model_name.lower()
    task = "cifar100" if "C100" in model_name else "cifar10"
    if "cifair" in lowered:
        img_type = "CiFAIR"
    elif "scangfi" in lowered:
        img_type = "scanGFI"
    else:
        raise ValueError(
            "model_name must contain a CiFAIR or scanGFI data marker")
    return task, img_type


def model_paths(model_name=None, model_root=None, task=None):
    """Resolve one checkpoint, inferring its dataset from the model name."""
    if model_name is None:
        task = normalize_task(task or "cifar100")
        model_name = MODEL_NAMES[task]
    inferred_task, img_type = infer_model_data(model_name)
    if task is not None and normalize_task(task) != inferred_task:
        raise ValueError(
            "task does not agree with the dataset encoded in model_name")
    root = Path(model_root) if model_root else MODEL_ROOTS[inferred_task]
    if not root.is_absolute():
        root = REPO_ROOT / root
    model_dir = root / model_name
    return {
        "task": inferred_task,
        "img_type": img_type,
        "model_name": model_name,
        "model_root": root,
        "model_dir": model_dir,
        "quantized": model_dir / (model_name + "_best_ckpt.pth"),
        "full_param": model_dir / (model_name + "_full_param_best_ckpt.pth"),
    }


def add_model_arguments(parser):
    parser.add_argument(
        "--model_name", default=None,
        help="Model name; task and input representation are inferred from it.")
    parser.add_argument(
        "--model_dir", default=None,
        help="Parent model directory containing MODEL_NAME/ (optional).")
    return parser


def load_checkpoint(path):
    import torch

    path = Path(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_state(model_name=None, model_root=None, full_param=False,
                     task=None):
    paths = model_paths(
        model_name=model_name, model_root=model_root, task=task)
    path = paths["full_param" if full_param else "quantized"]
    checkpoint = load_checkpoint(path)
    return checkpoint["net"], paths


def layer_indices(state_dict):
    indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("PcConvs.") and key.endswith(".FFconv.weight")
    }
    return sorted(indices)

