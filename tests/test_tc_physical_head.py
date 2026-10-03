"""TC one/two-state head-coordinate regression against the original /q path.

Two independently constructed models, small batches, no real training jobs.
Measured-head references use Phi(q*x)/q, not the old ideal final ReLU.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from distillation.srrl import SRRLLoss
from measured_pooling import configure_measured_pooling
from ode_pc import (ODEXInitFFFB, S2NoisyIYAsXZAs0, make_ode_block,
                    wrap_ode_block, QATWrapper1State, QATWrapper2State,
                    QATWrapper1StateWithX, ODEWrapper1State, ODEWrapper2State,
                    ODEWrapper1StateWithX, QATTester1State, QATTester2State,
                    QATTester1StateWithX)
from pc_model import logits_for_loss
from test_toggle_physical_head import base_model

ROOT = Path(__file__).resolve().parents[1]


def build_tc(state=1, legacy=False, scope="all", bias=True, hardware=False,
             device="cpu", qat=True, fused=False, model=None):
    model = (base_model(bias=bias, measured_activation_scope=scope)
             if model is None else model).to(device)
    model.device = torch.device(device)
    two = state == 2
    make_ode_block(model, ode_block=S2NoisyIYAsXZAs0 if two else ODEXInitFFFB,
                   method="dopri5", t_end=.3, n_steps=5, tol=1e-6)
    cls = (QATWrapper2State if two else QATWrapper1StateWithX if state == "with_x"
           else QATWrapper1State) if qat else (ODEWrapper2State if two else
               ODEWrapper1StateWithX if state == "with_x" else ODEWrapper1State)
    if qat == "tester":
        cls = (QATTester2State if two else QATTester1StateWithX
               if state == "with_x" else QATTester1State)
    model, wrappers = wrap_ode_block(
        model, ode_wrapper=cls, R=1e4, R_max=150e3, C=49e-15, k=1e3,
        v_dd=.1, one_over_q=1, w_bits=5, thermal_noise=False,
        tc_nonidealities=hardware, tc_conv_method="shared",
        nonlinear_R=hardware,
        nonlinear_R_table=str(ROOT / "hardware_data/res_vs_vin_10k_150k.csv"),
        tc_covariance_table=str(ROOT / "hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv"),
        nonlinear_R_curve_seed=19, enable_spin_variation=hardware,
        spin_variation_seed=23, enable_summing_current_noise=hardware,
        summing_noise_seed=29, enable_coupler_noise=hardware, coupler_noise_seed=31,
        enable_measured_activation=True,
        activation_curve_path=str(ROOT / "hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv"),
        activation_corner="MC18", activation_interpolation="piecewise_linear",
        activation_normalize_positive_endpoint=False, fuse_measured_activation=fused)
    if legacy:
        wrappers[-1].physical_head_output = False
        wrappers[-1].out_scale = wrappers[-1].q
        model.states_are_physical = False
        model.linear.physical_bias_scale = 1.
        if scope == "all":
            model.final_activation.set_coordinate_pullback_scale(wrappers[-1].q)
    from tc_cli import pooling_options
    pooling = pooling_options(SimpleNamespace(
        measured_pooling_curve_path=str(ROOT / "hardware_data/res_vs_vin_10k_150k.csv"),
        tc_covariance_table=str(ROOT / "hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv"),
        measured_pooling_nominal_R=1e4), wrappers)
    configure_measured_pooling(model, wrappers, enable_nonideality=True,
                              **pooling, seed=37, training_curve_mode="exact_curve")
    return model, wrappers


def compare_training(state, scope, bias, hardware, device="cpu", batches=4, builder=build_tc):
    old, _ = builder(state, True, scope, bias, hardware, device, fused=device == "cuda")
    new, wrappers = builder(state, False, scope, bias, hardware, device, fused=device == "cuda")
    aux = SRRLLoss(4, 4).to(device)
    aux_new = copy.deepcopy(aux)
    teacher = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).to(device)
    teacher.requires_grad_(False)
    optimizers = [torch.optim.SGD(list(m.parameters()) + list(a.parameters()),
                                lr=.001, momentum=.9, weight_decay=.001)
                  for m, a in ((old, aux), (new, aux_new))]
    for batch in range(batches):
        torch.manual_seed(100 + batch)
        x = torch.rand(2, 3, 4, 4, device=device)
        labels = torch.tensor([0, 2], device=device)
        tf = torch.rand(2, 4, 4, 4, device=device)
        results = []
        for model, a, opt in zip((old, new), (aux, aux_new), optimizers):
            torch.manual_seed(200 + batch)
            model.train(); opt.zero_grad()
            inp = x.detach().clone().requires_grad_()
            features, raw = model(inp, is_feat=True)
            logits = logits_for_loss(raw, model)
            loss = F.cross_entropy(logits, labels)
            if batch % 2:
                loss = loss + .3 * a(features[-1], tf, teacher(tf), teacher)
            loss.backward()
            results.append((features[-1].detach(), logits.detach(), loss.detach(), inp.grad))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
        for a, b in zip(list(old.parameters()) + list(aux.parameters()),
                        list(new.parameters()) + list(aux_new.parameters())):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6)
        for opt in optimizers:
            opt.step()
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        assert wrappers[-1].out_scale == 1
        assert new.global_avg_pool2d.input_scale == 1
    old.eval(); new.eval()
    with torch.no_grad():
        old_raw, new_raw = old(x), new(x)
    torch.testing.assert_close(logits_for_loss(new_raw, new), old_raw, rtol=3e-4, atol=3e-6)
    assert torch.equal(old_raw.argmax(1), new_raw.argmax(1))


@pytest.mark.parametrize("state", [1, 2, "with_x"])
@pytest.mark.parametrize("scope", ["pc_only", "all"])
@pytest.mark.parametrize("bias", [False, True])
def test_tc_head_losses_gradients_updates_and_validation(state, scope, bias):
    torch.set_num_threads(1)
    compare_training(state, scope, bias, False)


@pytest.mark.parametrize("state", [1, 2])
@pytest.mark.parametrize("scope", ["pc_only", "all"])
def test_tc_hardware_curves_noise_and_spin_unchanged(state, scope):
    torch.set_num_threads(1)
    compare_training(state, scope, True, True)


@pytest.mark.parametrize("state", [1, 2, "with_x"])
@pytest.mark.parametrize("scope", ["pc_only", "all"])
def test_non_qat_tc_wrappers(state, scope):
    old, _ = build_tc(state, True, scope, qat=False)
    new, _ = build_tc(state, False, scope, qat=False)
    old.eval(); new.eval()
    x = torch.rand(2, 3, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(logits_for_loss(new(x), new), old(x), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("state", [1, 2, "with_x"])
@pytest.mark.parametrize("scope", ["pc_only", "all"])
def test_tc_checkpoint_loading_and_inference_tester(tmp_path, state, scope):
    from inference_utils import load_and_prepare_model
    from pc_conv import PCConvReLU6
    from pc_model import PCNetNoBatchNorm

    trained, _ = build_tc(state, scope=scope, qat=False)
    payload = dict(net=trained.state_dict(), init_args=copy.deepcopy(trained.init_args),
                   net_type="PCNetNoBatchNorm")
    if scope == "pc_only":
        # Legacy checkpoint without the new metadata, but with stored bias.
        payload['init_args']['model_args'].pop('measured_activation_scope')
        payload['init_args']['model_args'].pop('final_head_type')
    path = tmp_path / 'saved.pth'
    torch.save(payload, path)
    models = []
    for legacy in (True, False):
        loaded = load_and_prepare_model(str(path), 'cpu', model_struct=PCNetNoBatchNorm,
                                       pc_conv_layer=PCConvReLU6, fuse_bn=False, noise_level=0)
        assert loaded.measured_activation_scope == scope
        torch.testing.assert_close(loaded.linear.bias, trained.linear.bias, atol=0, rtol=0)
        model, _ = build_tc(state, legacy=legacy, scope=scope, qat="tester", model=loaded)
        model.eval()
        models.append(model)
    x = torch.rand(2, 3, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(logits_for_loss(models[1](x), models[1]), models[0](x),
                                   rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("state", [1, 2])
def test_tc_old_recovery_checkpoint_preserves_next_update(tmp_path, state):
    from training_recovery import HISTORY, latest_path, save_latest, restore_latest

    def trainer(legacy):
        model, _ = build_tc(state, legacy=legacy, scope="pc_only", hardware=True)
        return SimpleNamespace(model=model, device="cpu", save_path=str(tmp_path),
            model_name="tc_head", optimizer=torch.optim.SGD(
                model.parameters(), lr=.001, momentum=.9, weight_decay=.001))

    def step(t):
        t.model.train(); t.optimizer.zero_grad()
        features, raw = t.model(torch.rand(2, 3, 4, 4), is_feat=True)
        loss = F.cross_entropy(logits_for_loss(raw, t.model), torch.tensor([0, 2]))
        loss = loss + features[-1].square().mean()
        loss.backward(); t.optimizer.step()
        return loss.detach()

    old = trainer(True)
    step(old)
    history = dict.fromkeys(HISTORY, None)
    history["val_acc"] = .2
    save_latest(old, 1, history)
    expected = step(old)
    new = trainer(False)
    new.recovery_checkpoint = latest_path(old)
    assert restore_latest(new)[0] == 1
    torch.testing.assert_close(step(new), expected, rtol=3e-5, atol=3e-6)
    assert new.model.states_are_physical
    for a, b in zip(old.model.parameters(), new.model.parameters()):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        if "momentum_buffer" in old.optimizer.state[a]:
            torch.testing.assert_close(old.optimizer.state[a]["momentum_buffer"],
                new.optimizer.state[b]["momentum_buffer"], rtol=5e-4, atol=5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("state", [1, 2])
def test_cuda_tc_all_relu_head(state):
    torch.cuda.set_per_process_memory_fraction(.04)
    torch.set_num_threads(1)
    compare_training(state, "all", True, True, "cuda", batches=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_toggle_all_relu_head():
    from measured_activation import MEASURED_ACTIVATION_TYPES
    from test_toggle_physical_head import physical

    def builder(state, legacy, scope, bias, hardware, device, fused):
        model = base_model(bias=bias, measured_activation_scope=scope).to(device)
        model.device = torch.device(device)
        model, wrappers = physical(model, legacy=legacy, measured=True)
        if legacy:
            model.final_activation.set_coordinate_pullback_scale(wrappers[-1].q)
        for module in model.modules():
            if isinstance(module, MEASURED_ACTIVATION_TYPES):
                module.fuse_measured_activation = fused
        return model, wrappers

    torch.cuda.set_per_process_memory_fraction(.04)
    torch.set_num_threads(1)
    compare_training(1, "all", True, True, "cuda", batches=8, builder=builder)
