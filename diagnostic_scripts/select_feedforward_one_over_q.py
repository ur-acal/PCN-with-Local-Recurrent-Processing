#!/usr/bin/env python3
"""Training-subset sanity check and deterministic, family-specific q selection.

Uses the production physical evaluator with unexpanded Level-2 operations and
frozen BN. Every candidate reloads the original unitless pretrained checkpoint.
No test images, BN recalibration or optimizer updates are used for selection.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluate_physical_feedforward_cifar as evaluator


def args_from_ft_command(argv):
    """Reuse the production FT parser rather than maintain another recipe."""
    from baseline import train_baseline_cifar as training
    original = sys.argv
    try:
        sys.argv = ['train', *argv]
        ft = training.parse_args()
        sys.argv = ['eval', '--model_name', ft.model_name,
                    '--checkpoint', ft.resume_checkpoint]
        args = evaluator.parse_args()
    finally:
        sys.argv = original
    for name in vars(args):
        if hasattr(ft, name):
            setattr(args, name, getattr(ft, name))
    args.checkpoint = ft.resume_checkpoint
    args.checkpoint_kind = 'unitless_physical'
    args.mismatch_type = ft.noise_type
    args.physical = True
    args.physical_level = 2
    args.use_expanded_weights = False
    args.bn_recalibrate = False
    args.evaluation_split = 'train'
    args.selection_samples = int(os.environ.get('SELECTION_SAMPLES', '1280'))
    args.max_eval_batches = int(os.environ.get('SELECTION_BATCHES', '10'))
    args.batch_size = int(os.environ.get('SELECTION_BATCH_SIZE', '128'))
    args.selection_training_hardware = True
    args.require_unitless_source = True
    args.model_config_override = training.parse_kv_overrides(ft.override)
    args.enable_nonlinear_R = ft.nonlinear_R_train_mode != 'none'
    if ft.activation_corner_mode == 'random_per_forward':
        args.activation_curve_sharing = (
            ft.activation_random_curve_sharing or ft.activation_curve_sharing)
    for name in ('data_seed', 'spin_variation_seed', 'summing_noise_seed',
                 'coupler_noise_seed', 'nonlinear_R_curve_seed',
                 'activation_curve_seed', 'dtc_timing_seed'):
        if getattr(args, name, None) is None:
            setattr(args, name, ft.seed)
    if ft.tc_feedforward or not ft.physical_feedforward or ft.physical_level != 2:
        raise ValueError('Automatic q selection requires toggle Level-2 FT')
    if args.selection_samples <= 0 or args.max_eval_batches <= 0 or args.batch_size <= 0:
        raise ValueError('Selection workload sizes must be positive')
    return args


def select(args, candidates, output, *, sanity_tolerance=5.0):
    if not candidates or any(not math.isfinite(q) or q <= 0 for q in candidates):
        raise ValueError("Supply positive finite q candidates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_one_over_q.txt").unlink(missing_ok=True)
    if getattr(args, 'require_unitless_source', False):
        import torch
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if (evaluator.classify_checkpoint_state_dict(checkpoint.get('net', checkpoint)) != 'physical_flat'
                or checkpoint.get('checkpoint_weight_format') in ('flattened_quantized', 'full_param')):
            raise ValueError('Automatic q selection needs an unquantized physical-pretraining checkpoint, not digital/QAT/FT weights')
    document = {
        "source": str(Path(args.checkpoint).resolve()),
        "source_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "configuration": vars(copy.copy(args)),
        "tie_break": "highest training-subset top1, then smallest one_over_q",
        "candidates": [],
    }
    document["curve_sha256"] = {
        name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
        for name in ("nonlinear_R_table", "activation_curve_path")
        if Path(getattr(args, name)).is_file()
    }

    def run(current):
        accuracy = float(evaluator.evaluate_once(current))
        if not math.isfinite(accuracy):
            raise RuntimeError("Non-finite selection accuracy")
        import torch
        torch.cuda.empty_cache()
        return accuracy

    def save():
        temporary = output / "selection.json.tmp"
        temporary.write_text(json.dumps(document, indent=2, default=str) + "\n")
        temporary.replace(output / "selection.json")

    ideal = copy.copy(args)
    for key in ("enable_nonlinear_R", "enable_measured_activation",
                "enable_measured_pooling", "enable_spin_variation",
                "enable_summing_current_noise", "enable_coupler_noise",
                "enable_dtc_nonideality", "enable_slow_summing_current",
                "enable_slow_coupler_noise"):
        setattr(ideal, key, False)
    ideal.w_bits = 8
    ideal.weight_quant_factor_bits = -1
    ideal.nonlinear_R_train_mode = "none"
    ideal.ft_corner_coupled_sampling = False
    ideal.activation_corner_mode = 'fixed'
    ideal.selection_training_hardware = False
    ideal.one_over_q = 128
    ideal.physical = False
    reference = run(ideal)
    ideal.physical = True
    physical = run(ideal)
    document["sanity"] = dict(reference_percent=reference,
                              physical_8bit_percent=physical,
                              tolerance_pp=sanity_tolerance,
                              configuration=vars(ideal))
    save()
    if abs(reference - physical) > sanity_tolerance:
        raise RuntimeError("Ideal physical accuracy did not recover unitless reference; see selection.json")
    print(f"SELECTION_SANITY_PASS unitless={reference:.4f} physical={physical:.4f}", flush=True)
    for q in sorted(set(candidates)):
        current = copy.copy(args)
        current.one_over_q = q
        current.result_path = str(output / f"qinv_{q:g}")
        accuracy = run(current)
        document["candidates"].append(dict(one_over_q=q, accuracy_percent=accuracy))
        save()
        print(f"SELECTION_CANDIDATE qinv={q:g} top1={accuracy:.4f}", flush=True)
    winner = max(document["candidates"], key=lambda item: (item["accuracy_percent"], -item["one_over_q"]))
    document["selected_one_over_q"] = winner["one_over_q"]
    save()
    (output / "selected_one_over_q.txt").write_text(f"{winner['one_over_q']:g}\n")
    print("SELECTION_COMPLETE", winner, flush=True)
    return winner["one_over_q"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sanity_tolerance_pp", type=float, default=5.0)
    parser.add_argument('--from_ft_command', action='store_true')
    own, remaining = parser.parse_known_args()
    if own.from_ft_command:
        args = args_from_ft_command(remaining)
    else:
        sys.argv = [sys.argv[0], *remaining]
        args = evaluator.parse_args()
    if args.evaluation_split != "train" or args.use_expanded_weights or args.bn_recalibrate:
        raise ValueError("Selection requires training data, unexpanded weights, and frozen BN")
    if args.physical_level != 2 or args.checkpoint_kind != "unitless_physical":
        raise ValueError("Selection requires Level 2 and a unitless physical pretrained checkpoint")
    select(args, own.candidates, own.output, sanity_tolerance=own.sanity_tolerance_pp)


if __name__ == "__main__":
    main()
