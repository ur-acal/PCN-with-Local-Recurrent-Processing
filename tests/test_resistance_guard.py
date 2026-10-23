"""Common Gaussian floor, 50-kOhm mapping, and empirical-path invariance."""
import argparse

import pytest
import torch
from torch import nn

from resistance_guard import add_argument, gaussian_resistance, resistance_floor, saved_floor
from measured_pooling import MeasuredAvgPool2d

FLOOR = 26791.333455489705


@pytest.mark.parametrize('quantity', ['resistance', 'conductance'])
def test_default_matches_previous_sampling_and_explicit_floor_is_in_ohms(quantity):
    values = torch.tensor([-1., 0., 1e-7, 1e-4, 1., 5e4], dtype=torch.float64)
    expected = values.clone().clamp_min_(torch.finfo(values.dtype).eps * 100.)
    if quantity == 'conductance':
        expected = expected.reciprocal()
    assert torch.equal(gaussian_resistance(values.clone(), quantity, 100.), expected)
    assert torch.equal(gaussian_resistance(values.clone(), quantity, 100., FLOOR),
                       expected.clamp_min(FLOOR))


@pytest.mark.parametrize('value', [-1, 0, float('inf'), float('nan')])
def test_invalid_floor(value):
    with pytest.raises(ValueError):
        resistance_floor(value)


def test_common_parser_and_historical_recovery_metadata(monkeypatch):
    monkeypatch.setenv('RESISTANCE_FLOOR_OHMS', str(FLOOR))
    parser = argparse.ArgumentParser()
    add_argument(parser)
    assert parser.parse_args([]).resistance_floor_ohms == FLOOR
    assert parser.parse_args(['--resistance_floor_ohms', 'none']).resistance_floor_ohms is None
    assert '--tc_resistance_floor_ohms' not in parser._option_string_actions
    assert saved_floor(dict(tc_feedforward=True)) == 1e-6
    assert saved_floor(dict(tc_nonidealities=True, tc_resistance_floor_ohms=FLOOR)) == FLOOR
    assert saved_floor(dict(tc_nonidealities=True, resistance_floor_ohms=FLOOR)) == FLOOR
    assert saved_floor(dict(tc_feedforward=False, tc_resistance_floor_ohms=1e-6)) is None
    assert saved_floor(dict(tc_feedforward=False, tc_resistance_floor_ohms=FLOOR)) is None


def test_cnn_fifty_kohm_grid_and_gradients():
    from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
    block = TCPhysicalBasicBlock(nn.Conv2d(2, 2, 1, bias=False), R=50e3,
        C=49e-15, v_dd=.1, one_shot_conv=True,
        tc_covariance_table='hardware_data/tc_stronger_covariance/coupler_full_range_all4500_resistance.csv',
        resistance_floor_ohms=FLOOR)
    wrapper = TCFeedForwardPhysicalWrapper(block, qat=True)
    assert wrapper.R_max == 750e3
    package = wrapper.configure_nonlinear_R_training(
        'hardware_data/tc_stronger_covariance/res_vs_vin_50k_750k.csv')['tc_curve_package']
    assert package.floor_ohms == FLOOR
    assert package.programmed_resistances.numel() == 15
    torch.testing.assert_close(package.programmed_resistances,
                               50e3 / package.levels[1:])
    from tc_nonidealities import prepare_tc_resistance_curves
    reference = prepare_tc_resistance_curves('hardware_data/res_vs_vin_10k_150k.csv',
        block.tc_covariance_table, levels=package.levels, R=10e3, R_max=150e3,
        dtype=package.means.dtype)
    # CSV column labels are measured resistances, slightly different from nominal codes.
    torch.testing.assert_close(package.mean_column_resistances, 5 * reference.mean_column_resistances)
    torch.testing.assert_close(package.means, 5 * reference.means)
    x = torch.full((2, 2, 3, 3), .01, requires_grad=True)
    y = wrapper(x)
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.parameters())


@pytest.mark.parametrize('two', [False, True])
def test_pcn_fifty_kohm_forward_backward(two):
    from test_tc_dense_training import make_block
    from ode_pc import QATWrapper1State, QATWrapper2State
    block = make_block(two)
    cls = QATWrapper2State if two else QATWrapper1State
    model = cls(ode_block=block, tc_nonidealities=True, R=50e3, R_max=750e3,
        C=49e-15, k=1e3, v_dd=.1, state_bound=1., w_bits=5,
        thermal_noise=False, offset_eps=0., is_first=True, is_last=True,
        resistance_floor_ohms=FLOOR, nonlinear_R=True,
        nonlinear_R_train_mode='exact_curve', nonlinear_R_curve_seed=44,
        nonlinear_R_table='hardware_data/tc_stronger_covariance/res_vs_vin_50k_750k.csv',
        tc_covariance_table='hardware_data/tc_stronger_covariance/coupler_full_range_all4500_resistance.csv')
    x = torch.full((2, 2, 2, 2), .02, requires_grad=True)
    y = model(x); y.sum().backward()
    assert block._tc_curve_package.floor_ohms == FLOOR
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fifty_kohm_cnn_expanded_and_offloaded_logits(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_physical_feedforward import _TinyWideResNet
    from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_wrappers
    from feedforward_validation import FeedForwardCNNValidator
    from torch.utils.data import DataLoader, TensorDataset
    outputs = []
    for offload in (False, True):
        torch.manual_seed(93)
        model = convert_wide_resnet_to_physical(_TinyWideResNet().eval(),
            physical_level=2, qat=True, R=50e3, C=49e-15, v_dd=.1,
            resistance_floor_ohms=FLOOR,
            tc_options=dict(tc_covariance_table='hardware_data/tc_stronger_covariance/coupler_full_range_all4500_resistance.csv',
                            one_shot_conv=True)).to(device)
        wrappers = list(iter_physical_wrappers(model))
        package = wrappers[0].configure_nonlinear_R_inference(
            'hardware_data/tc_stronger_covariance/res_vs_vin_50k_750k.csv', curve_seed=44)
        for w in wrappers[1:]:
            w.install_nonlinear_R_inference_package(package)
        x = torch.rand(2, 4, 8, 8, device=device) * .01
        loader = DataLoader(TensorDataset(x, torch.zeros(2, dtype=torch.long)), batch_size=2)
        directory = str(tmp_path / str(offload))
        validator = FeedForwardCNNValidator(model, directory, device, loader, directory,
            expanded_gaussian_curve_cpu_offload=offload)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()
        outputs.append(y)
        validator.close_curve_stager()
    assert torch.equal(*outputs)


@pytest.mark.parametrize('two', [False, True])
def test_pcn_default_and_legacy_floor_outputs_gradients_identical(two):
    from test_tc_dense_training import make_block, wrap
    results = []
    for floor in (None, 1e-6):
        model = wrap(make_block(two), resistance_floor_ohms=floor,
            nonlinear_R=True, nonlinear_R_table='hardware_data/res_vs_vin_10k_150k.csv',
            tc_covariance_table='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
            nonlinear_R_train_mode='exact_curve', nonlinear_R_curve_seed=44)
        x = torch.full((2, 2, 2, 2), .02, requires_grad=True)
        y = model(x)
        y.sum().backward()
        results.append((y.detach(), x.grad, [p.grad for p in model.parameters() if p.grad is not None]))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert all(torch.equal(a, b) for a, b in zip(results[0][2], results[1][2]))


@pytest.mark.parametrize('quantity', ['resistance', 'conductance'])
def test_expanded_gaussian_floor_and_cpu_storage(quantity):
    from validation import MVMConv, conv2d_to_matrix_fixed_padding
    mat, _, _ = conv2d_to_matrix_fixed_padding((1, 3, 3), torch.ones(1, 1, 1, 1), padding=0)
    module = MVMConv(mat, dict(padding=0, stride=1, ker_h=1, ker_w=1, inp_chan=1, out_chan=1))
    grid = torch.tensor([-.1, 0., .1])
    mean = torch.full((3,), -1. if quantity == 'resistance' else 1e-4)
    module.enable_csv(grid, torch.tensor([50e3]), None, None, None, 50e3,
        mul_mismatch_mode='static_mismatch',
        nonlinear_R_curve_sharing='per_coupler', nonlinear_R_curve_sampling='multivariate_gaussian',
        resistance_floor_ohms=FLOOR,
        curve_gaussian=dict(v_grid=grid, mean=mean, factor=torch.zeros(3, 3), value_scale=1., quantity=quantity))
    assert torch.all(module.nonlinear_R_curve_gaussian_R_normalized * 50e3 >= FLOOR - .01)
    expected = module.forward_pulse(torch.full((1, 1, 3, 3), .01), module.mat, nominal_R=50e3)
    module.offload_gaussian_curves_to_cpu()
    assert module._gaussian_curve_cpu is not None
    module.stage_gaussian_curves()
    assert torch.equal(module.forward_pulse(torch.full((1, 1, 3, 3), .01),
                                           module.mat, nominal_R=50e3), expected)


def test_pooling_gaussian_floor_and_empirical_bypass():
    gaussian = dict(v_grid=torch.tensor([-.1, 0., .1]), mean=torch.full((3,), -1.),
                    factor=torch.zeros(3, 3), value_scale=1., quantity='resistance')
    pool = MeasuredAvgPool2d(kernel_size=2, curve_gaussian=gaussian,
                             nominal_R=50e3, resistance_floor_ohms=FLOOR)
    y = pool(torch.full((2, 1, 2, 2), .01))
    assert torch.isfinite(y).all()
    assert torch.all(next(iter(pool._gaussian_curve_samples.values())) >= FLOOR)
    results = []
    for floor in (None, 1e9):
        pool = MeasuredAvgPool2d(kernel_size=2,
            curve_path='hardware_data/mc_45_corners/coupler_full_range',
            nominal_R=50e3, training_curve_mode='exact_curve', seed=42,
            resistance_floor_ohms=floor)
        x = torch.full((2, 1, 2, 2), .01, requires_grad=True)
        y = pool(x); y.sum().backward()
        results.append((y.detach(), x.grad, pool._generator.get_state()))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_empirical_toggle_network_outputs_gradients_assignments_and_rng_unchanged():
    from test_physical_feedforward import _TinyWideResNet
    from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_wrappers
    results = []
    for floor in (None, 1e9):
        torch.manual_seed(317)
        model = convert_wide_resnet_to_physical(_TinyWideResNet(),
            physical_level=2, qat=True, R=50e3, C=500e-15, v_dd=.5,
            one_over_q=20, w_bits=5, weight_quant_factor_bits=1,
            resistance_floor_ohms=floor, enable_spin_variation=True,
            enable_summing_current_noise=True, enable_coupler_noise=True,
            spin_variation_seed=4, summing_noise_seed=5, coupler_noise_seed=6)
        wrappers = list(iter_physical_wrappers(model))
        package = wrappers[0].configure_nonlinear_R_training(
            'hardware_data/mc_45_corners/coupler_full_range', curve_seed=7)
        for wrapper in wrappers[1:]:
            wrapper.install_nonlinear_R_training_package(package)
        x = (torch.rand(2, 4, 8, 8) * .01).requires_grad_()
        y = model(x); y.sum().backward()
        states, assignments = {}, {}
        for name, module in model.named_modules():
            if hasattr(module, '_nonlinear_R_training_curve_indices'):
                assignments[name] = dict(module._nonlinear_R_training_curve_indices)
            for attr in ('_nonlinear_R_training_generators', '_summing_noise_generators',
                         '_coupler_noise_generators', '_spin_variation_generators'):
                for key, gen in getattr(module, attr, {}).items():
                    states[(name, attr, key)] = gen.get_state()
        results.append(dict(logits=y.detach(), input_grad=x.grad,
            parameter_grads=[p.grad for p in model.parameters() if p.grad is not None],
            assignments=assignments, generators=states, global_rng=torch.get_rng_state()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_empirical_toggle_pcn_outputs_gradients_assignments_and_rng_unchanged():
    from pc_conv import PCConvReLU6
    from ode_pc import ToggleODEXInitFFFB, ToggleQATWrapper1State
    results = []
    for floor in (None, 1e9):
        torch.manual_seed(271)
        pc = PCConvReLU6(inp_chan=2, out_chan=2, kernel_size=1, padding=0, cls=2,
                        bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
        block = ToggleODEXInitFFFB(pc_conv=pc, noise_level=0., method='euler',
            t_end=.3, t_step=.1, tol=1e-6, toggle_n_cycles=3, odexinit_scaling_mode='direct',
            enable_spin_variation=True, enable_summing_current_noise=True,
            enable_coupler_noise=True, spin_variation_seed=4, summing_noise_seed=5, coupler_noise_seed=6)
        ToggleQATWrapper1State(ode_block=block, R=50e3, R_max=750e3, C=500e-15,
            k=1e3, v_dd=.5, state_bound=1., w_bits=5, thermal_noise=False,
            offset_eps=0., is_first=True, is_last=True, resistance_floor_ohms=floor,
            nonlinear_R=True, nonlinear_R_train_mode='exact_curve',
            nonlinear_R_curve_seed=7, nonlinear_R_table='hardware_data/mc_45_corners/coupler_full_range')
        x = torch.full((2, 2, 2, 2), .02, requires_grad=True)
        y = block(x); y.sum().backward()
        states = {name: {key: g.get_state() for key, g in getattr(block, name).items()}
                  for name in ('_nonlinear_R_training_generators', '_summing_noise_generators',
                               '_coupler_noise_generators', '_spin_variation_generators')}
        results.append(dict(output=y.detach(), input_grad=x.grad,
            parameter_grads=[p.grad for p in block.parameters() if p.grad is not None],
            assignments=dict(block._nonlinear_R_training_curve_indices),
            generators=states, global_rng=torch.get_rng_state()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
