"""Runtime, not just parser, coverage for analog classifier TC controls."""
import copy

import pytest
import torch

from final_linear import select_model_head


def build_head(device, method='shared', stages='both', seed=0):
    from test_toggle_physical_head import base_model
    from test_tc_physical_head import build_tc
    model = base_model()
    select_model_head(model, 'analog')
    model, wrappers = build_tc(1, model=model, hardware=True, device=device)
    template = model.PcConvs[-1]
    template._tc_conv_method = method
    template._tc_noise_cfg['tc_noise_stages'] = stages
    template._nonlinear_R_pkg['nonlinear_R_curve_seed'] = seed
    return model


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method', ['shared', 'loop', 'grouped'])
@pytest.mark.parametrize('stages', ['both', 'ff', 'fb'])
def test_tc_controls_reach_classifier_computation(device, method, stages):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = build_head(device, method, stages)
    head = model.linear
    x = torch.full((2, head.in_features), .01, device=device, requires_grad=True)
    head(x).sum().backward()
    circuit = head._circuit
    assert circuit.tc_conv_method == method
    assert circuit.tc_noise_stages == stages
    assert circuit._tc_curve_seed == 0  # zero is a valid explicit seed
    sample = next(iter(circuit._tc_samples.values()))
    from tc_nonidealities import TCSharedCurve
    assert isinstance(sample, TCSharedCurve) == (method == 'shared')
    state = x.new_zeros((2, head.out_features, 1, 1))
    source = torch.cat((x, x.new_full((2, 1), head.q)), 1)[:, :, None, None]
    _, eps = circuit._noise_context(circuit.conv1, source, state, 'z')
    assert bool((eps > 0).any()) == (stages != 'fb')
    assert torch.isfinite(head.weight.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_loop_grouped_classifier_outputs_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = build_head(device, 'loop', 'fb', seed=37)
    model.PcConvs[-1]._tc_noise_cfg['enable_spin_variation'] = False
    head = model.linear
    x = torch.full((2, head.in_features), .01, device=device, requires_grad=True)
    loop = head(x)
    loop_grads = torch.autograd.grad(loop.sum(), (x, head.weight, head.bias))
    circuit = head._circuit
    circuit.eval()
    head.eval()  # retain identical sampled per-code curves
    circuit.tc_conv_method = circuit._tc_conv_method = 'grouped'
    grouped = head(x)
    grouped_grads = torch.autograd.grad(grouped.sum(), (x, head.weight, head.bias))
    torch.testing.assert_close(loop, grouped, atol=1e-6, rtol=1e-5)
    for a, b in zip(loop_grads, grouped_grads):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_validation_resets_private_head_realizations_and_seed_is_effective():
    from trainer import TrainerCiFar
    model = build_head('cpu', seed=19)
    head = model.linear
    x = torch.full((2, head.in_features), .01)
    head(x)
    circuit = head._circuit
    assert circuit._tc_curve_seed == 19
    previous = copy.deepcopy(circuit._tc_samples)
    trainer = object.__new__(TrainerCiFar)
    trainer.model = model
    trainer.reset_spin_variation_for_inference()
    assert not circuit._tc_samples
    assert circuit._spin_factor_y is None and circuit._spin_factor_z is None
    head.eval()
    head(x)
    assert any(not torch.equal(previous[k].resistance, circuit._tc_samples[k].resistance)
               for k in previous)
    cached = next(iter(circuit._tc_samples.values()))
    head(x)
    assert next(iter(circuit._tc_samples.values())) is cached


def test_energy_measurement_rejects_unmetered_analog_head():
    from tc_energy import enable_tc_coupler_energy
    from test_toggle_physical_head import base_model
    model = base_model().eval()
    select_model_head(model, 'analog')
    with pytest.raises(NotImplementedError, match='analog classifier'):
        enable_tc_coupler_energy(model)


def test_tc_probe_resets_private_spin_and_rng_but_keeps_curves():
    from tc_cli import reset_after_probe
    model = build_head('cpu').eval()
    head = model.linear
    head(torch.full((2, head.in_features), .01))
    circuit = head._circuit
    curves = dict(circuit._tc_samples)
    assert circuit._tc_generators
    assert circuit._spin_factor_z is not None
    reset_after_probe(model)
    assert circuit._spin_factor_z is None
    assert not circuit._tc_generators
    assert not circuit._spin_variation_generators
    assert all(circuit._tc_samples[k] is v for k, v in curves.items())


@pytest.mark.parametrize('lazy', [False, True])
def test_health_check_preserves_private_rng(lazy):
    from trainer import TrainerCiFar
    from types import SimpleNamespace
    model = build_head('cpu')
    head = model.linear
    x = torch.full((2, head.in_features), .01)
    if not lazy:
        head(x)
    circuit = head._circuit
    states = {} if lazy else {k: g.get_state().clone()
                             for k, g in circuit._tc_generators.items()}
    trainer = object.__new__(TrainerCiFar)
    trainer.model = model
    trainer.train_dataloader = SimpleNamespace(generator=None)
    trainer.health_check_seed = 123
    trainer.health_check_batches = 1
    def evaluate(*args, **kwargs):
        head(x)
        head._circuit._tc_generators['new'] = torch.Generator().manual_seed(99)
        return 0., 0., 0., 0.
    trainer.evaluate = evaluate
    trainer._evaluate_training_health()
    if lazy:
        assert head._circuit is None
    else:
        assert set(circuit._tc_generators) == set(states)
        assert all(torch.equal(g.get_state(), states[k])
                   for k, g in circuit._tc_generators.items())


def test_validation_resets_dtc_in_backbone_and_private_head():
    from trainer import TrainerCiFar
    from test_final_linear import template
    from final_linear import AnalogLinear
    from physical_feedforward import PulsePhysicalBasicBlock
    block = PulsePhysicalBasicBlock(torch.nn.Conv2d(2, 2, 1))
    head = AnalogLinear(4, 3)
    ref = template()
    head.configure(physical=True, q=.1, v_dd=.5, template=ref,
                   R=ref.R, C=ref.C)
    head.begin_evaluation_trial()
    head.eval()
    head(torch.full((2, 4), .01))
    block._dtc_fixed_variation['y'] = torch.ones(1)
    head._circuit._dtc_fixed_variation['z'] = torch.ones(1)
    trainer = object.__new__(TrainerCiFar)
    trainer.model = torch.nn.ModuleList([block, head])
    trainer.reset_spin_variation_for_inference()
    assert not block._dtc_fixed_variation
    assert not head._circuit._dtc_fixed_variation


def test_fast_path_override_and_unsupported_quantizer():
    from test_final_linear import template
    from final_linear import AnalogLinear, configure_model_head
    from types import SimpleNamespace
    block = template()
    block.toggle_fast_path = False
    head = AnalogLinear(4, 3)
    head.configure(physical=True, q=.1, v_dd=.5, template=block,
                   R=block.R, C=block.C)
    head(torch.ones(2, 4) * .01)
    assert head._circuit.toggle_fast_path is False
    with pytest.raises(NotImplementedError, match='qat_cls'):
        configure_model_head(SimpleNamespace(linear=head), physical=True,
                             q=.1, v_dd=.5, template=block, qat_cls=object)


@pytest.mark.parametrize('mode', ['pre_quant_weight', 'post_quant_amplitude'])
def test_nonzero_mismatch_training_is_not_silently_misapplied(mode):
    from trainer import WrappedNoisyPulseModel
    from test_toggle_physical_head import base_model
    model = base_model()
    select_model_head(model, 'analog')
    with pytest.raises(NotImplementedError, match='mismatch-aware training'):
        WrappedNoisyPulseModel(model, [.1], mismatch_mode=mode)
