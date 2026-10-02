"""Small-batch, CPU-only full-checkpoint FT/validation coordinate audit.

Reconstructs the old /q placement as the reference; never updates checkpoint
files. Uses synthetic 32x32 inputs/teacher features to check algebra/gradients,
not to estimate dataset accuracy. Run Level-3 data checks separately.
"""
import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Never compete with ongoing GPU training.

import torch
from torch import nn
from torch.nn import functional as F
from distillation.srrl import SRRLLoss
from inference_utils import load_and_prepare_model
from measured_pooling import configure_measured_pooling
from ode_pc import ToggleODEXInitFFFB, ToggleQATWrapper1State
from pc_conv import PCConvReLU6
from pc_model import PCNetNoBatchNorm, logits_for_loss


def build(path, legacy):
    torch.manual_seed(4096)
    wrappers = {}
    model = load_and_prepare_model(
        str(path), "cpu", model_struct=PCNetNoBatchNorm, pc_conv_layer=PCConvReLU6,
        fuse_bn=False, noise_level=0, wrappers=wrappers,
        ode_params=dict(ode_block=ToggleODEXInitFFFB, method="dopri5", t_end=1.75,
                        n_steps=5, tol=1e-6, odexinit_scaling_mode="direct",
                        toggle_timing_mode="fixed", toggle_y_time=10e-9,
                        z_over_y_time=1, toggle_timing_R=50e3, toggle_timing_C=500e-15,
                        enable_spin_variation=True, sigma_spin=.1,
                        spin_variation_seed=4096, enable_summing_current_noise=True,
                        summing_current_p=.6e-12, summing_noise_seed=4096,
                        enable_coupler_noise=True, coupler_noise_p=.6e-12,
                        coupler_noise_seed=4096),
        ode_wrapper_params=dict(
            ode_wrapper=ToggleQATWrapper1State, R=50e3, C=500e-15, v_dd=.5,
            one_over_q=5, w_bits=5, weight_quant_factor_bits=1, thermal_noise=False,
            nonlinear_R=True, nonlinear_R_table=str(ROOT / "hardware_data/mc_45_corners/coupler_full_range"),
            nonlinear_R_mc_quantity="conductance", nonlinear_R_train_mode="exact_curve",
            nonlinear_R_curve_sharing="shared", nonlinear_R_curve_seed=4096,
            enable_measured_activation=True,
            activation_curve_path=str(ROOT / "hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv"),
            activation_corner="MC18", activation_interpolation="piecewise_linear",
            activation_normalize_positive_endpoint=False, fuse_measured_activation=False))
    if legacy:
        wrappers["wrappers"][-1].physical_head_output = False
        wrappers["wrappers"][-1].out_scale = wrappers["wrappers"][-1].q
        model.states_are_physical = False
        model.linear.physical_bias_scale = 1.0
    configure_measured_pooling(model, wrappers["wrappers"], enable_nonideality=True,
        curve_path=str(ROOT / "hardware_data/mc_45_corners/coupler_full_range"),
        nominal_R=50e3, seed=4096, training_curve_mode="exact_curve")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    logging.disable(logging.WARNING)
    torch.set_num_threads(args.threads)
    old, new = build(Path(args.checkpoint), True), build(Path(args.checkpoint), False)
    initial_weights = copy.deepcopy(old.state_dict())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=args.checkpoint, device="cpu", batch_size=args.batch_size,
                  synthetic_inputs=True, actual_hardware_curves=True, rows=[])
    torch.manual_seed(51)
    aux_old = SRRLLoss(old.ocs[-1], 8)
    aux_new = copy.deepcopy(aux_old)
    teacher_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, old.linear.out_features))
    teacher_head.requires_grad_(False)
    opts = [torch.optim.SGD(list(m.parameters()) + list(a.parameters()),
                           lr=.001, momentum=.9, weight_decay=.001)
            for m, a in ((old, aux_old), (new, aux_new))]
    for i in range(args.batches):
        # Compare each gradient/update from identical checkpoint weights, not
        # two increasingly diverging floating-point training trajectories.
        old.load_state_dict(initial_weights)
        new.load_state_dict(initial_weights)
        aux_new.load_state_dict(aux_old.state_dict())
        opts[1].load_state_dict(copy.deepcopy(opts[0].state_dict()))
        torch.manual_seed(100 + i)
        x = torch.rand(args.batch_size, old.ics[0], 32, 32)
        labels = torch.arange(args.batch_size) % old.linear.out_features
        teacher_features = torch.rand(args.batch_size, 8, 4, 4)
        teacher_logits = teacher_head(teacher_features)
        use_srrl = i % 2 == 1
        rows = []
        for model, aux, opt in ((old, aux_old, opts[0]), (new, aux_new, opts[1])):
            model.train(); opt.zero_grad()
            torch.manual_seed(500 + i)
            features, raw = model(x, is_feat=True)
            logits = logits_for_loss(raw, model)
            loss = F.cross_entropy(logits, labels)
            if use_srrl:
                loss = loss + .3 * aux(features[-1], teacher_features, teacher_logits, teacher_head)
            loss.backward()
            rows.append((features[-1].detach(), logits.detach(), loss.detach(), raw.detach()))
        row = dict(batch=i, srrl=use_srrl)
        for j, key in enumerate(("feature", "restored_logit", "loss")):
            row[key + "_max_abs_diff"] = float((rows[0][j]-rows[1][j]).abs().max())
            torch.testing.assert_close(rows[0][j], rows[1][j], rtol=2e-4, atol=2e-5)
        row["raw_logit_scale_error"] = float((rows[1][3]-new.state_q*rows[0][3]).abs().max())
        row["gradient_max_abs_diff"] = 0.
        row["gradient_max_relative_l2_diff"] = 0.
        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                error = float((a.grad-b.grad).abs().max())
                if error > row["gradient_max_abs_diff"]:
                    row["gradient_max_abs_diff"], row["worst_gradient"] = error, name
                relative = float((a.grad-b.grad).norm() / a.grad.norm().clamp_min(1e-8))
                row["gradient_max_relative_l2_diff"] = max(row["gradient_max_relative_l2_diff"], relative)
                try:
                    # Scale-aware criteria: individual near-zero components
                    # are ill-conditioned for relative-error comparisons.
                    assert relative < 3e-5 or float((a.grad-b.grad).norm()) < 1e-6
                    assert error < 3e-5 * float(a.grad.abs().max()) + 1e-6
                except AssertionError as exc:
                    row.update(failed_gradient=name, detail=str(exc),
                               reference_grad_max=float(a.grad.abs().max()),
                               reference_grad_norm=float(a.grad.norm()),
                               gradient_diff_norm=float((a.grad-b.grad).norm()))
                    report["rows"].append(row)
                    out.write_text(json.dumps(report, indent=2)+"\n")
                    raise
        for a, b in zip(aux_old.parameters(), aux_new.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=1e-4)
        for opt in opts:
            opt.step()
        row["parameter_max_abs_diff"] = max(float((a-b).detach().abs().max())
                                             for a,b in zip(old.parameters(), new.parameters()))
        old.eval(); new.eval()
        with torch.no_grad():
            torch.manual_seed(600+i); ref = old(x)
            # Compare validation from identical parameters too.
            # Optimizer-update differences are recorded separately above.
            new.load_state_dict(old.state_dict())
            torch.manual_seed(600+i); raw = new(x)
            torch.testing.assert_close(raw/new.state_q, ref, rtol=2e-4, atol=2e-5)
            row["validation_prediction_matches"] = int((raw.argmax(1)==ref.argmax(1)).sum())
            assert row["validation_prediction_matches"] == args.batch_size
        report["rows"].append(row)
        out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
