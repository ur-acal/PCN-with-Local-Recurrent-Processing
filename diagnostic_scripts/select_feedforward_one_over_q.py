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
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluate_physical_feedforward_cifar as evaluator


def select(args, candidates, output, *, sanity_tolerance=5.0):
    if not candidates or any(not math.isfinite(q) or q <= 0 for q in candidates):
        raise ValueError("Supply positive finite q candidates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_one_over_q.txt").unlink(missing_ok=True)
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
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = evaluator.parse_args()
    if args.evaluation_split != "train" or args.use_expanded_weights or args.bn_recalibrate:
        raise ValueError("Selection requires training data, unexpanded weights, and frozen BN")
    if args.physical_level != 2 or args.checkpoint_kind != "unitless_physical":
        raise ValueError("Selection requires Level 2 and a unitless physical pretrained checkpoint")
    select(args, own.candidates, own.output, sanity_tolerance=own.sanity_tolerance_pp)


if __name__ == "__main__":
    main()
