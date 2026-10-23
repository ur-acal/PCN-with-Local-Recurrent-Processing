"""TC Gaussian corner coupling: real source fits and train/eval lifecycle."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ft_corner_sampling import configure_ft_corner_coupled_sampling
from tc_nonidealities import (prepare_tc_corner_factors, prepare_tc_resistance_curves,
                              tc_training_corner_factor)
from measured_activation import (PiecewiseLinearActivation, MEASURED_ACTIVATION_TYPES,
    configure_measured_activation_corner_mode, configure_feedforward_measured_activation,
    feedforward_measured_activation_factory)
from measured_pooling import MeasuredAvgPool2d
from test_tc_dense_training import make_block, wrap

ROOT = Path(__file__).resolve().parents[1]
MEAN = str(ROOT / 'hardware_data/res_vs_vin_10k_150k.csv')
COV = str(ROOT / 'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
RELU = str(ROOT / 'hardware_data/mc_45_corners/0906_RELU_Voltage')


def make_model(kind, *, coupled=True, checkpoint=False, device='cpu'):
    torch.manual_seed(53)
    if kind == 'cnn':
        from baseline.cifar_resnet import WideResNetCIFAR
        from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_wrappers
        from tc_feedforward_cli import configure_pooling
        model = convert_wide_resnet_to_physical(
            WideResNetCIFAR(depth=10, base_width=2, widen_factor=1,
                num_classes=3, use_batchnorm=False, conv_bias=False,
                avgpool_main_downsample=True, avgpool_downsample_shortcut=True),
            qat=True, physical_level=2, R=1e4, C=49e-15, v_dd=.1,
            tc_options=dict(tc_covariance_table=COV, one_shot_conv=False,
                reuse_accepted_step_training=True, checkpoint_ode_rhs_training=checkpoint))
        wrappers = list(iter_physical_wrappers(model))
        package = wrappers[0].configure_nonlinear_R_training(MEAN, curve_seed=4)
        for wrapper in wrappers[1:]:
            wrapper.install_nonlinear_R_training_package(package)
        configure_feedforward_measured_activation(model,
            feedforward_measured_activation_factory(RELU, .1,
                corner='TT_25_1_MC18', curve_sharing='per_spin'))
        configure_pooling(model, SimpleNamespace(measured_pooling_curve_path=MEAN,
            nonlinear_R_table=MEAN, tc_covariance_table=COV,
            measured_pooling_nominal_R=1e4, nonlinear_R_curve_seed=3))
    else:
        block = make_block(kind == 'state2').to(device)
        block.integration_time = block.integration_time.to(device)
        wrapper = wrap(block, tc_conv_method='shared')
        for options in (block.option_aca, wrapper.orig_option_aca):
            options.update(checkpoint_ode_rhs_training=checkpoint,
                           reuse_accepted_step_training=True)
        package = prepare_tc_resistance_curves(MEAN, COV,
            levels=block._get_quant_magnitude_levels(block.FFconv.weight),
            dtype=torch.float32, device=device)
        block._tc_curve_package = package
        block._tc_curve_generator = torch.Generator(device=device).manual_seed(4)
        block.act_fn = PiecewiseLinearActivation(RELU, .1, corner='TT_25_1_MC18',
                                                 curve_sharing='per_spin')
        pool = MeasuredAvgPool2d(kernel_size=2, nominal_R=1e4, seed=3,
            curve_gaussian=dict(v_grid=package.v_grid, mean=package.means[-1],
                factor=package.factor, value_scale=1., quantity='resistance',
                tc_covariance_path=COV))
        model = nn.Sequential(block, pool, nn.Flatten(), nn.Linear(8, 3))
        # Wrappers are external owners in the production PCN too; retain QAT hooks.
        object.__setattr__(model, '_test_wrapper', wrapper)
    if coupled:
        configure_ft_corner_coupled_sampling(model,
            activation_corner_mode='random_per_forward', include_measured_activation=True)
    configure_measured_activation_corner_mode(model, 'random_per_forward', 'per_spin')
    model.to(device)
    return model


def test_all_real_corner_fits_match_direct_subset_and_share_storage():
    from data_utils import MC45CornerData
    from utils import load_mc_res_curve_gaussian
    package = prepare_tc_resistance_curves(MEAN, COV, levels=torch.arange(16) / 15)
    factors = prepare_tc_corner_factors(COV, package.v_grid, dtype=torch.float64, device='cpu')
    entries = MC45CornerData.coupler_file_corners(COV)
    assert len(factors) == 45
    for (p, v, t), entry in entries.items():
        fit = load_mc_res_curve_gaussian(COV, quantity='resistance',
            curve_indices=entry['curve_indices'], dtype=torch.float64)
        torch.testing.assert_close(factors[f'{p.upper()}_V{v}_T{t}'],
                                   fit['factor'] * fit['value_scale'], rtol=0, atol=0)
    model = make_model('state1')
    banks = [m._tc_corner_factors for m in model.modules() if hasattr(m, '_tc_corner_factors')]
    assert len(banks) == 2 and banks[0] is banks[1]


@pytest.mark.parametrize('kind', ['state1', 'state2', 'cnn'])
def test_coupled_forward_backward_eval_and_checkpoint_replay(kind):
    torch.set_num_threads(1)
    results = []
    for checkpoint in (False, True):
        model = make_model(kind, checkpoint=checkpoint)
        sampler = model._ft_corner_coupled_sampler
        sampler.corners = ('SS_V2_T2',)
        x = torch.rand(2, 3 if kind == 'cnn' else 2,
                       8 if kind == 'cnn' else 4, 8 if kind == 'cnn' else 4) * .02
        output = model(x)
        assert torch.isfinite(output).all()
        for module in model.modules():
            if hasattr(module, '_tc_corner_factors'):
                assert module._ft_corner_coupled_sampler is sampler
                ref = next(iter(module._tc_corner_factors.factors.values()))
                assert tc_training_corner_factor(module, ref) is module._tc_corner_factors.get('SS_V2_T2', ref)
            if isinstance(module, MEASURED_ACTIVATION_TYPES):
                assert set(module._sampled_curve_indices.flatten().tolist()) <= set(
                    module._ft_corner_curve_indices['SS_V2_T2'])
        # A later forward/corner must not change the already-captured RHS graph.
        sampler.active_corner = 'TT_V0_T0'
        output.square().sum().backward()
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        results.append((output.detach(), grads))
        model.eval()
        with torch.no_grad():
            model(x)
        assert sampler.active_corner is None
        for module in model.modules():
            if hasattr(module, '_tc_corner_factors'):
                default = torch.tensor(0.)
                assert tc_training_corner_factor(module, default) is default
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fixed_relu_is_not_coupled_and_off_has_no_corner_bank():
    model = make_model('state1', coupled=False)
    assert not any(hasattr(m, '_tc_corner_factors') for m in model.modules())
    configure_ft_corner_coupled_sampling(model, activation_corner_mode='fixed')
    assert not hasattr(model[0].act_fn, '_ft_corner_coupled_sampler')


def test_local_launcher_forwards_existing_flag():
    import os, shlex, subprocess
    env = dict(os.environ, MODEL_NAME='test', TC_DRY_RUN='true',
               FT_CORNER_COUPLED_SAMPLING='true')
    result = subprocess.run(['bash', 'launch_scripts/run_tc_nonidealities.sh', 'ft'],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout)
    assert args[args.index('--ft_corner_coupled_sampling') + 1] == 'true'


def test_lazy_analog_head_inherits_same_corner_and_bank():
    from final_linear import AnalogLinear
    model = make_model('state1')
    head = AnalogLinear(8, 3)
    head.configure(physical=True, q=.1, v_dd=.1, template=model[0],
                   family='tc', R=1e4, C=49e-15)
    model[3] = head
    output = model(torch.full((2, 2, 4, 4), .01))
    output.square().sum().backward()
    assert head._circuit._tc_corner_factors is model[0]._tc_corner_factors
    assert head._circuit._ft_corner_coupled_sampler is model._ft_corner_coupled_sampler
    assert torch.isfinite(head.weight.grad).all()


def test_bank_migration_is_shared_and_cached():
    model = make_model('state1')
    bank = model[0]._tc_corner_factors
    ref = torch.empty(0, dtype=torch.float64)
    factor = bank.get('SS_V2_T2', ref)
    assert factor.dtype == torch.float64
    assert bank.get('SS_V2_T2', ref) is factor
    assert model[1]._tc_corner_factors.get('SS_V2_T2', ref) is factor


def test_factor_override_covers_shared_and_per_code_sampling():
    package = prepare_tc_resistance_curves(MEAN, COV, levels=torch.arange(16) / 15)
    zero = torch.zeros_like(package.factor)
    samples = package.sample_levels(generator=torch.Generator().manual_seed(1), factor=zero)
    torch.testing.assert_close(samples, package.means, rtol=0, atol=0)
    for method in ('histogram', 'uniform'):
        curve = package.sample_shared(torch.arange(16), sampling=method,
            generator=torch.Generator().manual_seed(1), factor=zero)
        torch.testing.assert_close(curve.resistance, package.means[curve.code - 1], rtol=0, atol=0)
    # The override must not mutate the pooled factor used in evaluation/OFF.
    assert torch.count_nonzero(package.factor) > 0
