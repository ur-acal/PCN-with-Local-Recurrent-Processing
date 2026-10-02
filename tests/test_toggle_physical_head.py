"""Coordinate-only regression: old /q head versus raw physical head.

Small CPU batches exercise real toggle QAT, measured activation/pooling, SRRL,
dropout, optimizer momentum/decay, and checkpoint reconstruction.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from distillation.srrl import SRRLLoss
from inference_utils import load_and_prepare_model
from measured_pooling import configure_measured_pooling
from ode_pc import (make_ode_block, wrap_ode_block, ToggleODEXInitFFFB,
                    ToggleQATWrapper1State, ToggleWrapper1State,
                    TogglePulseODEXInitFFFB, TogglePulseQATWrapper1State)
from pc_conv import PCConvReLU6
from pc_model import PCNetNoBatchNorm, logits_for_loss

ROOT = Path(__file__).resolve().parents[1]


def base_model(bias=True):
    torch.manual_seed(71)
    model = PCNetNoBatchNorm(
        inp_channels=[3, 4], out_channels=[4, 4], max_pool=[False, False],
        num_classes=3, pc_conv_layer=PCConvReLU6, kernel_size=1, padding=0,
        first_bn=False, avg_pooling=True, dropout=0.25,
        tie_weights=False, tie_bp=False, bypass=False, linear_bias=bias)
    model.device = torch.device("cpu")
    return model.cpu()


def physical(model, legacy=False, measured=False, pulse=False, qat=True):
    make_ode_block(model, ode_block=TogglePulseODEXInitFFFB if pulse else ToggleODEXInitFFFB,
                   method="euler", t_end=1.75, n_steps=5, tol=1e-4,
                   toggle_timing_mode="fixed", toggle_y_time=10e-9,
                   z_over_y_time=1, toggle_timing_R=50e3, toggle_timing_C=500e-15,
                   odexinit_scaling_mode="direct")
    model, wrappers = wrap_ode_block(
        model, ode_wrapper=(TogglePulseQATWrapper1State if pulse else
                            ToggleQATWrapper1State if qat else ToggleWrapper1State),
        R=50e3, C=500e-15, v_dd=.5, one_over_q=5, w_bits=5,
        weight_quant_factor_bits=1, thermal_noise=False,
        enable_measured_activation=measured,
        activation_curve_path=str(ROOT / "hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv"),
        activation_corner="MC18", activation_interpolation="piecewise_linear",
        activation_normalize_positive_endpoint=False,
        fuse_measured_activation=False)
    if legacy:
        # Reconstruct the pre-change coordinate convention exactly. QAT resets
        # out_scale=q each forward when physical_head_output is absent/false.
        wrappers[-1].physical_head_output = False
        wrappers[-1].out_scale = wrappers[-1].q
        model.states_are_physical = False
        model.linear.physical_bias_scale = 1.0
    configure_measured_pooling(
        model, wrappers, enable_nonideality=measured,
        curve_path=str(ROOT / "hardware_data/mc_45_corners/coupler_full_range/tt_25_1.csv") if measured else None,
        nominal_R=50e3, seed=19, training_curve_mode="exact_curve")
    return model, wrappers


@pytest.mark.parametrize("srrl", [False, True])
@pytest.mark.parametrize("measured", [False, True])
def test_physical_losses_gradients_and_optimizer_match(srrl, measured):
    torch.set_num_threads(1)
    old, _ = physical(base_model(), legacy=True, measured=measured)
    new, wrappers = physical(base_model(), measured=measured)
    aux_old = SRRLLoss(4, 4)
    aux_new = copy.deepcopy(aux_old)
    teacher_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
    for p in teacher_head.parameters():
        p.requires_grad_(False)
    params_old = list(old.parameters()) + (list(aux_old.parameters()) if srrl else [])
    params_new = list(new.parameters()) + (list(aux_new.parameters()) if srrl else [])
    opt_old = torch.optim.SGD(params_old, lr=.001, momentum=.9, weight_decay=.001)
    opt_new = torch.optim.SGD(params_new, lr=.001, momentum=.9, weight_decay=.001)
    for batch in range(8):
        torch.manual_seed(100 + batch)
        x = torch.rand(2, 3, 4, 4)
        labels = torch.tensor([0, 2])
        teacher_features = torch.rand(2, 4, 4, 4)
        teacher_logits = teacher_head(teacher_features)
        results = []
        for model, aux, optim in ((old, aux_old, opt_old), (new, aux_new, opt_new)):
            model.train()
            optim.zero_grad()
            torch.manual_seed(200 + batch)  # same dropout mask
            features, raw = model(x, is_feat=True)
            loss_logits = logits_for_loss(raw, model)
            loss = F.cross_entropy(loss_logits, labels)
            if srrl:
                loss = loss + .3 * aux(features[-1], teacher_features, teacher_logits, teacher_head)
            loss.backward()
            results.append((features[-1].detach(), loss_logits.detach(), loss.detach(), raw.detach()))
        for a, b in zip(results[0][:3], results[1][:3]):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(results[1][3], results[0][3] * new.state_q, rtol=2e-5, atol=2e-6)
        for a, b in zip(params_old, params_new):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
        opt_old.step()
        opt_new.step()
        for a, b in zip(params_old, params_new):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        assert wrappers[-1].out_scale == 1.0  # QAT callback must not undo it
        assert new.global_avg_pool2d.input_scale == 1.0
    # Validation has eval() set, but losses still require conversion.
    old.eval(); new.eval()
    with torch.no_grad():
        old_out, new_out = old(x), new(x)
        torch.testing.assert_close(logits_for_loss(new_out, new), old_out, rtol=3e-5, atol=3e-6)
        assert torch.equal(new_out.argmax(1), old_out.argmax(1))


@pytest.mark.parametrize("pulse,qat", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("bias", [False, True])
def test_wrapper_variants_and_bias(pulse, qat, bias):
    old, _ = physical(base_model(bias), legacy=True, pulse=pulse, qat=qat)
    new, wrappers = physical(base_model(bias), pulse=pulse, qat=qat)
    old.eval(); new.eval()
    with torch.no_grad():
        for _ in range(4):
            x = torch.rand(1, 3, 4, 4)
            torch.testing.assert_close(new(x) / new.state_q, old(x), atol=2e-6, rtol=2e-5)
            assert wrappers[-1].out_scale == 1


def test_legacy_checkpoint_and_new_metadata(tmp_path):
    model = base_model()
    args = copy.deepcopy(model.init_args)
    args["model_args"].pop("final_head_type")
    for version, init_args in (("old", args), ("new", model.init_args)):
        path = tmp_path / (version + ".pth")
        torch.save(dict(net=model.state_dict(), init_args=init_args,
                        net_type="PCNetNoBatchNorm"), path)
        loaded = load_and_prepare_model(str(path), "cpu", model_struct=PCNetNoBatchNorm,
                                        pc_conv_layer=PCConvReLU6, fuse_bn=False, noise_level=0)
        assert loaded.final_head_type == "old_ideal"
        assert not loaded.states_are_physical
        model.eval(); loaded.eval()
        x = torch.rand(2, 3, 4, 4)
        torch.testing.assert_close(loaded(x), model(x), atol=0, rtol=0)
        assert logits_for_loss(loaded(x), loaded) is not None
        for k, v in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[k], v, atol=0, rtol=0)


@pytest.mark.parametrize("head", ["analog", "digital"])
def test_placeholder_fails_explicitly(head):
    with pytest.raises(NotImplementedError):
        PCNetNoBatchNorm(inp_channels=[3], out_channels=[4], max_pool=[False], final_head_type=head)


def test_physical_recovery_preserves_bias_optimizer_and_rng(tmp_path):
    from training_recovery import HISTORY, latest_path, save_latest, restore_latest

    def trainer(legacy):
        model, _ = physical(base_model(), legacy=legacy, measured=True)
        return SimpleNamespace(model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.001),
            device="cpu", save_path=str(tmp_path), model_name="physical_head")

    def step(t):
        t.model.train(); t.optimizer.zero_grad()
        x = torch.rand(2, 3, 4, 4)
        feat, raw = t.model(x, is_feat=True)
        loss = F.cross_entropy(logits_for_loss(raw, t.model), torch.tensor([0, 2]))
        # A feature loss checks the is_feat return alongside CE during recovery.
        loss = loss + feat[-1].square().mean()
        loss.backward(); t.optimizer.step()
        return loss.detach()

    reference = trainer(True)
    step(reference)
    history = dict.fromkeys(HISTORY, None)
    history["val_acc"] = .2
    save_latest(reference, 1, history)
    saved = torch.load(latest_path(reference), weights_only=False)
    # A genuinely old checkpoint has no head-selection metadata.
    saved["init_args"]["model_args"].pop("final_head_type")
    torch.save(saved, latest_path(reference))
    expected_loss = step(reference)
    resumed = trainer(False)
    resumed.recovery_checkpoint = latest_path(reference)
    assert restore_latest(resumed)[0] == 1
    assert resumed.model.states_are_physical
    actual_loss = step(resumed)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
    for (name, a), (_, b) in zip(reference.model.named_parameters(), resumed.model.named_parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, msg=name)
        ma = reference.optimizer.state[a].get("momentum_buffer")
        mb = resumed.optimizer.state[b].get("momentum_buffer")
        if ma is not None:
            torch.testing.assert_close(ma, mb, rtol=3e-4, atol=3e-6)
