"""Validate real queued launcher commands without running training/evaluation.

Uses production argument parsers, deserializes FT input checkpoints on CPU,
and resolves/validates every corner's actual child inference command. Future
FT output checkpoints are checked for consistent paths, not fabricated.
"""
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baseline")]


def parse(script, arguments, function="parse_args"):
    namespace = runpy.run_path(str(ROOT / script), run_name="launch_preflight")
    with patch.object(sys, "argv", [script, *arguments]):
        return namespace, namespace[function]()


def readable_checkpoint(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("net", state.get("state_dict", state.get("model_state_dict", state)))
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Empty checkpoint: {path}")
    tensors = [v for v in weights.values() if isinstance(v, torch.Tensor)]
    if not tensors or any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError(f"Invalid checkpoint tensors: {path}")
    return str(path)


def check_training(script, namespace, args):
    import numpy as np
    # Confirm the singleton is an actual paired-column conductance bank.
    bank = np.loadtxt(args.nonlinear_R_table, delimiter=",", skiprows=1)
    assert bank.shape[1] in (2, 4), bank.shape
    assert np.isfinite(bank).all() and (bank[:, 1::2] > 0).all()
    if bank.shape[1] == 4:
        assert np.array_equal(bank[:, :2], bank[:, 2:])
    assert args.fuse_measured_activation and args.enable_measured_activation
    epochs = (args.num_epochs if script == "train_ode_cifar.py" else
              namespace["parse_kv_overrides"](args.override)["num_epochs"])
    assert args.enable_measured_pooling and epochs == 50
    assert Path(args.activation_curve_path).is_file()
    assert Path(args.teacher_ckpt).is_file()
    assert shutil.which("scangen"), "scangen missing from queued environment PATH"
    from measured_activation import PiecewiseLinearActivation
    activation = PiecewiseLinearActivation(args.activation_curve_path, args.v_dd,
                                          corner=args.activation_corner)
    import torch
    assert torch.isfinite(activation(torch.zeros(1))).all()
    if script == "train_ode_cifar.py":
        raw = os.environ["MODEL_NAME_OVERRIDE"]
        source = Path(args.save_path) / raw / (raw + "_best_ckpt.pth")
        assert args.toggle_timing_mode == "derived" and args.one_over_q == 1
    else:
        source = args.resume_checkpoint
        assert args.toggle_timing_mode == "fixed"
        assert abs(args.toggle_y_time - 15e-9) < 1e-18 and args.one_over_q == 5
    return readable_checkpoint(source)


def check_evaluation(script, namespace, args):
    from data_utils import MC45CornerData
    catalog = MC45CornerData(
        args.mc_45_corner_dir,
        spin_variation_source=args.mc_spin_variation_source,
        dtc_pulse_width_variation_source=args.mc_dtc_pulse_width_variation_source,
        relu_monte_carlo_source=args.mc_relu_monte_carlo_source,
        coupler_nonlinear_variation_source=args.mc_coupler_nonlinear_variation_source,
        coupler_nonlinear_variation_quantity=args.mc_coupler_nonlinear_variation_quantity,
        coupler_nominal_R=args.mc_coupler_nominal_R)
    assert len(catalog.corners) == 45 and args.n_trials == 1
    assert not args.corner_ids and args.fuse_measured_activation
    captured = []
    if script == "scripts/run_toggle_nonideality_ablation.py":
        for i, corner in enumerate(catalog.corners):
            command, _ = namespace["build_corner_command"](args, corner, i)
            captured.append(command)
        child = "ode_inference.py"
    else:
        class Captured(Exception):
            pass

        def capture(command, **kwargs):
            captured.append(command)
            raise Captured

        def log_open(path, mode="r", *a, **kw):
            if mode == "w":
                return io.StringIO()
            return original_open(path, mode, *a, **kw)

        original_open = Path.open
        for corner in catalog.corners:
            with patch("subprocess.Popen", capture), patch.object(Path, "open", log_open):
                try:
                    namespace["run_corner"](args, corner, [None])
                except Captured:
                    pass
        child = "baseline/evaluate_physical_feedforward_cifar.py"
    child_namespace = runpy.run_path(str(ROOT / child), run_name="launch_preflight")
    for corner, command in zip(catalog.corners, captured):
        start = command.index(child)
        with patch.object(sys, "argv", command[start:]):
            resolved = child_namespace["parse_args"]()
        assert resolved.fuse_measured_activation
        assert Path(resolved.nonlinear_R_table).is_file()
        assert Path(resolved.activation_curve_path).is_file()
        assert len(corner["coupler"]["curve_indices"]) == 100
    assert len(captured) == 45
    return len(captured)


def main():
    arguments = sys.argv[1:]
    if arguments[0] == "--check-handoff":
        records = [json.loads(line.split(" ", 1)[1])
                   for line in Path(arguments[1]).read_text().splitlines()
                   if line.startswith("PREFLIGHT_PASS ")]
        assert len(records) == 2
        assert records[0]["expected_checkpoint"] == records[1]["checkpoint"]
        print("PREFLIGHT_HANDOFF_PASS " + records[1]["checkpoint"])
        return
    if arguments[0] == "-u":
        arguments.pop(0)
    script = arguments.pop(0)
    function = "get_args" if script == "train_ode_cifar.py" else "parse_args"
    namespace, args = parse(script, arguments, function)
    if script in ("train_ode_cifar.py", "baseline/train_baseline_cifar.py"):
        source = check_training(script, namespace, args)
        if script == "train_ode_cifar.py":
            name = namespace["_constr_model_name"](args, rep=1)
            expected = Path(args.output_save_path) / name / (name + "_best_ckpt.pth")
        else:
            from input_preprocessing import append_preprocessing_suffix, resolve_preprocessing
            bits, center = resolve_preprocessing(args.resume_checkpoint,
                                                 args.input_quant_bits, args.center_student_input)
            output = append_preprocessing_suffix(args.output_dir, bits, center)
            name = f"{args.case}_{args.dataset}_{args.model_name}"
            expected = Path(output) / args.dataset / args.case / args.model_name / name / (name + "_full_param_best_ckpt.pth")
        print("PREFLIGHT_PASS " + json.dumps({"script": script, "source": source,
              "singleton": args.nonlinear_R_table, "epochs": 50,
              "expected_checkpoint": str(expected)}))
    else:
        corners = check_evaluation(script, namespace, args)
        checkpoint = (str(Path(args.model_dir) / args.model_name / (args.model_name + "_best_ckpt.pth"))
                      if script == "scripts/run_toggle_nonideality_ablation.py" else args.checkpoint)
        print("PREFLIGHT_PASS " + json.dumps({"script": script, "corners": corners,
              "trials": args.n_trials, "family": args.mc_coupler_nonlinear_variation_source,
              "checkpoint": checkpoint}))


if __name__ == "__main__":
    main()
