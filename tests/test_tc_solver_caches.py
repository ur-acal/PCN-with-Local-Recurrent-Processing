"""Inference cache controls preserve arithmetic and current-interval noise."""
import pytest
import torch
from tc_nonidealities import TCNoiseLifecycle
from TorchDiffEqPack.odesolver.ode_solver import odesolve

def context(ref, retain):
    coefficients = [(ref.new_tensor(.01), ref.new_tensor(.02))] * 2
    generators = {(b, s): torch.Generator(device=ref.device).manual_seed(41 + i)
                  for i, b in enumerate((0, 1, 'fb')) for s in ('sum', 'coupler')}
    return TCNoiseLifecycle(coefficients, generators, coefficients[0], ref,
                            replay_cache=retain)

@pytest.mark.parametrize('method', ['euler', 'dopri5'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_outputs_and_bounded_history(method, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    outputs = []
    for retain in (True, False):
        with torch.no_grad():
            ref = torch.zeros(2, 3, device=device)
            ctx = context(ref, retain)
            class RHS(torch.nn.Module):
                def forward(self, t, y):
                    return (.1 - y[0] + ctx.fb_current(), y[0] - y[1])
            rhs = RHS()
            rhs.tc_context = ctx
            options = dict(t0=0., t1=1., t_eval=[0., 1.], h=.01,
                           method=method, rtol=1e-4, atol=1e-6,
                           eps=(.02236, .02236), noise_type='addi',
                           dense_output=retain, proj_fn=lambda y: y.clamp(-.1, .1))
            solver = odesolve(rhs, (ref, ref.clone()), options, return_solver=True)
            result = solver.integrate((ref, ref.clone()), 0., t_eval=[0., 1.])
            outputs.append(result)
            if retain:
                assert ctx.tape
                assert solver.dense_states['y_end']
            else:
                assert not ctx.tape
                assert not getattr(solver, 'dense_states', {}).get('y_end', [])
    for a, b in zip(*outputs):
        assert torch.equal(a, b)

def test_current_step_reuse_training_fallback_and_replay_guard():
    with torch.no_grad():
        ref = torch.zeros(3)
        ctx = context(ref, False)
        assert torch.equal(ctx.fb_current(), ctx.fb_current())
        ctx.accepted()
        assert not ctx.tape
        with pytest.raises(RuntimeError, match='replay'):
            ctx.restart()
        def rhs(t, y):
            return y
        rhs.tc_context = context(ref, False)
        with pytest.raises(ValueError, match='regenerate_graph'):
            odesolve(rhs, (ref,), dict(method='euler', t0=0., t1=1., h=.1,
                                     regenerate_graph=True), return_solver=True)
    ctx = context(torch.zeros(3), False)
    assert ctx.replay_cache  # Grad-enabled execution retains backward replay.
    ctx.fb_current()
    ctx.accepted()
    assert ctx.tape

@pytest.mark.parametrize('two', [False, True])
def test_pcn_wiring_and_forward(two):
    from test_tc_dense_training import make_block, wrap
    outputs = []
    for retain in (True, False):
        block = make_block(two)
        model = wrap(block, tc_solver_dense_output=retain, tc_noise_replay_cache=retain,
                     enable_summing_current_noise=True, enable_coupler_noise=True,
                     summing_noise_seed=123, coupler_noise_seed=456)
        model.eval()
        assert block.option_aca['dense_output'] == retain
        with torch.no_grad():
            outputs.append(model(torch.full((2, 2, 2, 2), .02)))
    assert torch.equal(*outputs)

def test_cnn_wiring_and_forward():
    from test_tc_feedforward import block
    outputs = []
    for retain in (True, False):
        model = block(tc_method='euler', tc_num_steps=100,
                      tc_solver_dense_output=retain, tc_noise_replay_cache=retain,
                      enable_summing_current_noise=True, enable_coupler_noise=True,
                      summing_noise_seed=123, coupler_noise_seed=456)
        model.eval()
        with torch.no_grad():
            outputs.append(model(torch.full((2, 2, 2, 2), .02, dtype=torch.float64)))
    assert torch.equal(*outputs)
