"""Compare legacy endpoint interpolation with an adaptively accepted exact landing.

This script patches AdaptiveGridSolver only in memory. It reconstructs the exact
20 model inputs stored in the hardware-validation pickle, runs both endpoint
policies through the same expanded model, and writes numerical differences to
JSON. Production solver code and the input pickle are not modified.
"""

import argparse
import inspect
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ode_inference
import ode_pc
from TorchDiffEqPack.odesolver import adaptive_grid_solver as ag
from TorchDiffEqPack.odesolver.base import ODESolver


MODEL_NAME = (
    "QAT5bNT0p1mulQAT5bNT0p1mulPCNetNoBatchNorm_PCConvReLU6_0.0eps_"
    "ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_"
    "0.01LR_3K1S4C_0.25Dropout_8Layers_2Pool_scanGFI_1REP"
)
DEFAULT_PKL = (
    ROOT / "hw_validation_data" / MODEL_NAME / "5b" /
    "20samples_dopri5.pkl"
)
DEFAULT_OUTPUT = ROOT / "results" / "dopri5_endpoint_landing_comparison.json"


def build_capped_integrator():
    """Return an in-memory variant that caps h before the normal error test."""
    original = ag.AdaptiveGridSolver.integrate_search_grids
    source = textwrap.dedent(inspect.getsource(original))
    old_step = "h_current = h_new  # .clone().detach()"
    new_step = (
        "final_t = self.t_eval[-1] if self.t_eval is not None else self.t1\n"
        "                remaining = abs(final_t - t_current)\n"
        "                if h_new > remaining:\n"
        "                    # adapt_stepsize mutates tensor inputs in place; keep\n"
        "                    # the accepted capped step separate from its proposal.\n"
        "                    h_new = float(remaining)\n"
        "                h_current = h_new  # exact-landing diagnostic"
    )
    old_crossing = "> torch.abs(self.t_end - self.t0)"
    new_crossing = ">= torch.abs(self.t_end - self.t0)"
    if source.count(old_step) != 1 or source.count(old_crossing) != 1:
        raise RuntimeError("Adaptive solver source no longer matches diagnostic patch")
    source = source.replace(old_step, new_step, 1)
    source = source.replace(old_crossing, new_crossing, 1)
    scope = dict(vars(ag))
    exec(source, scope)
    return scope["integrate_search_grids"]


def model_args():
    argv = [
        "ode_inference.py",
        "--model_name", MODEL_NAME,
        "--ckpt", "best",
        "--task", "cifar10",
        "--model_dir", str(ROOT / "saved_ckpt"),
        "--method", "dopri5",
        "--tol", "1e-6",
        "--n_steps", "1000",
        "--ts_scale", "1",
        "--d_start", "0",
        "--d_end", "1",
        "--n_sweep_left", "0",
        "--n_sweep_right", "1",
        "--C", "49e-15",
        "--R", "10e3",
        "--R_max", "150e3",
        "--tie_cap", "false",
        "--one_over_q", "1",
        "--v_dd", "0.1",
        "--w_bits", "5",
        "--pc_conv", "PCConvReLU6Noisy",
        "--ode_block", "ODEXInitFFFB",
        "--ode_wrapper", "QATTester1State",
        "--img_type", "scanGFI",
        "--thermal_noise", "false",
        "--conv_only", "true",
        "--hw_validate", "true",
        "--test_expanded", "false",
        "--rec_full_traj", "true",
        "--t_end_sf", "5",
        "--pvt_to_origin", "false",
        "--hw_val_path", str(ROOT / "hw_validation_data"),
        "--expanded_w_dir", str(ROOT / "expanded_weights"),
        "--valid_samples", "20",
        "--valid_select_layer", "6",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = argv
        return ode_inference.parse_args()
    finally:
        sys.argv = saved_argv


def load_pickle(path):
    import pickle

    with path.open("rb") as fp:
        return pickle.load(fp)


def first_state(traj):
    return traj[0] if isinstance(traj, (tuple, list)) else traj


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def endpoint_metrics(current, exact):
    delta = exact.astype(np.float64) - current.astype(np.float64)
    current64 = current.astype(np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta.ravel()) /
            max(np.linalg.norm(current64.ravel()), np.finfo(np.float64).eps)
        ),
        "fraction_gt_1e-7": float(np.mean(np.abs(delta) > 1e-7)),
        "fraction_gt_1e-5": float(np.mean(np.abs(delta) > 1e-5)),
    }


def elementwise_relative_error_metrics(current, exact):
    current64 = current.astype(np.float64)
    exact64 = exact.astype(np.float64)
    absolute_error = np.abs(current64 - exact64)
    exact_magnitude = np.abs(exact64)
    nonzero = exact_magnitude > 0
    stabilized = absolute_error / np.maximum(
        exact_magnitude, np.finfo(exact.dtype).eps)
    nonzero_values = absolute_error[nonzero] / exact_magnitude[nonzero]
    per_sample = stabilized.reshape(stabilized.shape[0], -1).mean(axis=1)
    return {
        "formula": "mean(abs(current - exact) / abs(exact))",
        "epsilon": float(np.finfo(exact.dtype).eps),
        "epsilon_stabilized_mean": float(stabilized.mean()),
        "nonzero_exact_mean": float(nonzero_values.mean()),
        "exact_zero_count": int(np.count_nonzero(~nonzero)),
        "element_count": int(exact_magnitude.size),
        "per_sample_epsilon_stabilized_mean": per_sample.tolist(),
    }


def per_sample_endpoint_metrics(current, exact):
    current64 = current.astype(np.float64).reshape(current.shape[0], -1)
    delta = exact.astype(np.float64).reshape(exact.shape[0], -1) - current64
    denom = np.linalg.norm(current64, axis=1)
    denom = np.maximum(denom, np.finfo(np.float64).eps)
    return [{
        "sample": int(sample),
        "mean_abs": float(np.mean(np.abs(delta[sample]))),
        "max_abs": float(np.max(np.abs(delta[sample]))),
        "relative_l2": float(np.linalg.norm(delta[sample]) / denom[sample]),
    } for sample in range(current.shape[0])]


def change_ratio(endpoint, start):
    denom = np.maximum(np.abs(start), np.finfo(start.dtype).eps)
    return np.mean(np.abs(endpoint - start) / denom, axis=1)


class EndpointComparisonValidator(ode_inference.Validator):
    input_pickle = None
    output_path = None

    @torch.no_grad()
    def gen_validate_data(self, wrappers, solver, n_samples=20,
                          sample_inp=None, select_layer=6):
        del solver, sample_inp, select_layer
        saved = load_pickle(self.input_pickle)
        physical_input = torch.from_numpy(saved["layer_0"]["inp"]).to(self.device)
        flat_per_sample = physical_input.shape[1]
        input_channels = self.model.ics[0]
        side = int(round(math.sqrt(flat_per_sample / input_channels)))
        if input_channels * side * side != flat_per_sample:
            raise ValueError("Cannot infer image shape from saved layer-0 input")
        model_input = (
            physical_input.reshape(n_samples, input_channels, side, side) /
            wrappers[0].inp_scale
        )

        original_integrator = ag.AdaptiveGridSolver.integrate_search_grids
        capped_integrator = build_capped_integrator()
        original_solve = ode_pc.aca_ode_solve
        original_interpolate = ODESolver.interpolate

        def run_case(integrator, direct_endpoint=False):
            final_intervals = []
            active_layer = [-1]
            solve_count = [0]

            def tracked_solve(func, y0, options, *args, **kwargs):
                layer = solve_count[0]
                solve_count[0] += 1
                active_layer[0] = layer
                try:
                    return original_solve(func, y0, options, *args, **kwargs)
                finally:
                    active_layer[0] = -1

            def tracked_interpolate(solver_obj, t_old, t_new, t_eval,
                                    y0, y1, *args, **kwargs):
                t_old_f = float(t_old)
                t_new_f = float(t_new)
                t_eval_f = float(t_eval)
                t1_f = float(solver_obj.t1)
                if math.isclose(t_eval_f, t1_f, rel_tol=1e-6, abs_tol=1e-15):
                    final_intervals.append({
                        "layer": active_layer[0],
                        "t_old": t_old_f,
                        "t_new": t_new_f,
                        "t_eval": t_eval_f,
                        "solver_t1": t1_f,
                        "step": t_new_f - t_old_f,
                        "overshoot": t_new_f - t_eval_f,
                    })
                if direct_endpoint and t_new_f == t_eval_f:
                    return y1
                return original_interpolate(
                    solver_obj, t_old, t_new, t_eval, y0, y1,
                    *args, **kwargs)

            ag.AdaptiveGridSolver.integrate_search_grids = integrator
            ode_pc.aca_ode_solve = tracked_solve
            ODESolver.interpolate = tracked_interpolate
            layer_outputs = []
            handlers = []
            for layer in self.model.PcConvs:
                handlers.append(layer.register_forward_hook(
                    lambda _module, _inputs, output: layer_outputs.append(
                        to_numpy(output))))
            try:
                logits = self.model(model_input)
                trajectories = []
                steps = []
                for layer in self.model.PcConvs:
                    trajectories.append(
                        np.array(first_state(layer._last_full_traj), copy=True))
                    steps.append(np.array(layer._last_full_steps, copy=True))
                return {
                    "logits": to_numpy(logits),
                    "trajectories": trajectories,
                    "steps": steps,
                    "layer_outputs": layer_outputs,
                    "final_intervals": final_intervals,
                }
            finally:
                for handler in handlers:
                    handler.remove()
                ag.AdaptiveGridSolver.integrate_search_grids = original_integrator
                ode_pc.aca_ode_solve = original_solve
                ODESolver.interpolate = original_interpolate

        try:
            current = run_case(original_integrator)
            exact = run_case(capped_integrator, direct_endpoint=True)
            current_repeat = run_case(original_integrator)
            exact_repeat = run_case(capped_integrator, direct_endpoint=True)
        finally:
            ag.AdaptiveGridSolver.integrate_search_grids = original_integrator
            ode_pc.aca_ode_solve = original_solve
            ODESolver.interpolate = original_interpolate

        report = {
            "model_name": MODEL_NAME,
            "input_pickle": str(self.input_pickle),
            "sample_count": int(n_samples),
            "comparison": (
                "current cubic interpolation after overshoot vs final step "
                "capped to remaining interval before normal adaptive acceptance, "
                "with the accepted endpoint returned directly"
            ),
            "logits": endpoint_metrics(current["logits"], exact["logits"]),
            "predictions": {
                "current": np.argmax(current["logits"], axis=1).tolist(),
                "exact": np.argmax(exact["logits"], axis=1).tolist(),
                "disagreements": int(np.count_nonzero(
                    np.argmax(current["logits"], axis=1) !=
                    np.argmax(exact["logits"], axis=1))),
            },
            "repeatability": {
                "current_logits": endpoint_metrics(
                    current["logits"], current_repeat["logits"]),
                "exact_logits": endpoint_metrics(
                    exact["logits"], exact_repeat["logits"]),
            },
            "layers": [],
        }
        current_intervals = {
            row["layer"]: row for row in current["final_intervals"]
        }
        exact_intervals = {
            row["layer"]: row for row in exact["final_intervals"]
        }
        for layer_idx, (current_traj, exact_traj) in enumerate(zip(
                current["trajectories"], exact["trajectories"])):
            current_endpoint = current_traj[-1]
            exact_endpoint = exact_traj[-1]
            current_steps = current["steps"][layer_idx]
            exact_steps = exact["steps"][layer_idx]
            current_target = current_steps[-1] / self.t_end_sf
            exact_target = exact_steps[-1] / self.t_end_sf
            current_return_idx = int(np.argmin(
                np.abs(current_steps - current_target)))
            exact_return_idx = int(np.argmin(
                np.abs(exact_steps - exact_target)))
            current_return_state = current_traj[current_return_idx]
            exact_return_state = exact_traj[exact_return_idx]
            saved_endpoint = first_state(saved[f"layer_{layer_idx}"]["traj"])[-1]
            current_ratio = change_ratio(current_endpoint, current_traj[0])
            exact_ratio = change_ratio(exact_endpoint, exact_traj[0])
            report["layers"].append({
                "layer": layer_idx,
                "current_points": int(current_traj.shape[0]),
                "exact_points": int(exact_traj.shape[0]),
                "current_final_interval": current_intervals.get(layer_idx),
                "exact_final_interval": exact_intervals.get(layer_idx),
                "endpoint_difference": endpoint_metrics(
                    current_endpoint, exact_endpoint),
                "elementwise_relative_error": (
                    elementwise_relative_error_metrics(
                        current_endpoint, exact_endpoint)),
                "per_sample_endpoint_difference": per_sample_endpoint_metrics(
                    current_endpoint, exact_endpoint),
                "network_output_difference": endpoint_metrics(
                    current["layer_outputs"][layer_idx],
                    exact["layer_outputs"][layer_idx]),
                "returned_state_difference": endpoint_metrics(
                    current_return_state, exact_return_state),
                "current_return_selection": {
                    "index": current_return_idx,
                    "time": float(current_steps[current_return_idx]),
                    "target": float(current_target),
                    "neighbor_times": current_steps[
                        max(0, current_return_idx - 2):
                        current_return_idx + 3].tolist(),
                    "output_vs_selected_max_abs": float(np.max(np.abs(
                        current["layer_outputs"][layer_idx].reshape(
                            current_return_state.shape) - current_return_state))),
                },
                "exact_return_selection": {
                    "index": exact_return_idx,
                    "time": float(exact_steps[exact_return_idx]),
                    "target": float(exact_target),
                    "neighbor_times": exact_steps[
                        max(0, exact_return_idx - 2):
                        exact_return_idx + 3].tolist(),
                    "output_vs_selected_max_abs": float(np.max(np.abs(
                        exact["layer_outputs"][layer_idx].reshape(
                            exact_return_state.shape) - exact_return_state))),
                },
                "current_repeat_endpoint_difference": endpoint_metrics(
                    current_endpoint,
                    current_repeat["trajectories"][layer_idx][-1]),
                "exact_repeat_endpoint_difference": endpoint_metrics(
                    exact_endpoint,
                    exact_repeat["trajectories"][layer_idx][-1]),
                "saved_replay_max_abs": float(np.max(
                    np.abs(current_endpoint - saved_endpoint))),
                "current_ratio_mean": float(current_ratio.mean()),
                "current_ratio_max": float(current_ratio.max()),
                "exact_ratio_mean": float(exact_ratio.mean()),
                "exact_ratio_max": float(exact_ratio.max()),
                "per_sample_current_ratio": current_ratio.tolist(),
                "per_sample_exact_ratio": exact_ratio.tolist(),
                "per_sample_ratio_delta": (
                    exact_ratio - current_ratio).tolist(),
            })

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(report, indent=2) + "\n")

        print("mode comparison complete")
        print("logit max abs difference: {:.9g}".format(
            report["logits"]["max_abs"]))
        print("current repeat logit max abs difference: {:.9g}".format(
            report["repeatability"]["current_logits"]["max_abs"]))
        print("exact repeat logit max abs difference: {:.9g}".format(
            report["repeatability"]["exact_logits"]["max_abs"]))
        print(
            "layer  current_overshoot  mean_abs_diff  max_abs_diff  "
            "relative_l2  current_ratio  exact_ratio"
        )
        for row in report["layers"]:
            interval = row["current_final_interval"] or {"overshoot": float("nan")}
            diff = row["endpoint_difference"]
            print(
                "{:>5d}  {:>17.9g}  {:>13.9g}  {:>12.9g}  "
                "{:>11.9g}  {:>13.7g}  {:>11.7g}".format(
                    row["layer"], interval["overshoot"],
                    diff["mean_abs"], diff["max_abs"],
                    diff["relative_l2"], row["current_ratio_mean"],
                    row["exact_ratio_mean"],
                )
            )
        print("report: {}".format(self.output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pickle", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.input_pickle = args.input_pickle.resolve()
    args.output = args.output.resolve()
    if not args.input_pickle.is_file():
        raise FileNotFoundError(args.input_pickle)

    saved = load_pickle(args.input_pickle)
    layer0 = saved["layer_0"]["inp"]
    checkpoint = (
        ROOT / "saved_ckpt" / MODEL_NAME /
        (MODEL_NAME + "_best_ckpt.pth")
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    input_channels = checkpoint_data["init_args"]["model_args"]["inp_channels"][0]
    side = int(round(math.sqrt(layer0.shape[1] / input_channels)))
    dummy_inputs = torch.zeros(2, input_channels, side, side)
    dummy_targets = torch.zeros(2, dtype=torch.long)
    dummy_loader = DataLoader(
        TensorDataset(dummy_inputs, dummy_targets), batch_size=2, shuffle=False)

    parsed = model_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = (
        Path(parsed.model_dir) / parsed.model_name /
        (parsed.model_name + "_{}_ckpt.pth".format(parsed.ckpt))
    )
    pc_conv = ode_inference.PC_CONV_CLASS.get(
        parsed.pc_conv, ode_inference.PCConvNoisy)

    EndpointComparisonValidator.input_pickle = args.input_pickle
    EndpointComparisonValidator.output_path = args.output
    original_validator = ode_inference.Validator
    ode_inference.Validator = EndpointComparisonValidator
    try:
        with torch.no_grad():
            ode_inference.run_validation_data_gen(
                parsed, dummy_loader, str(checkpoint_path), pc_conv, device)
    finally:
        ode_inference.Validator = original_validator


if __name__ == "__main__":
    main()
