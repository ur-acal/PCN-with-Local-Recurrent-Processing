"""Opt-in wiring for resistance-coded feedforward CNNs."""
import torch


def add_arguments(parser):
    def boolean(value):
        if value.lower() not in ('true', 'false'):
            raise ValueError('Expected true or false')
        return value.lower() == 'true'
    parser.add_argument('--tc_feedforward', type=boolean, default=False)
    parser.add_argument('--one_shot_conv', type=boolean, default=False)
    parser.add_argument('--tc_method', choices=('dopri5', 'euler'), default='dopri5')
    parser.add_argument('--tc_tol', type=float, default=1e-6)
    parser.add_argument('--tc_step_size', type=float, default=None)
    parser.add_argument('--tc_noise_reference_R', type=float, default=50e3)
    parser.add_argument('--tc_covariance_table', default='./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
    parser.add_argument('--tc_curve_sampling', choices=('histogram', 'uniform'), default='histogram')
    parser.add_argument('--measured_pooling_curve_path', default=None)
    parser.add_argument('--measured_pooling_nominal_R', type=float, default=1e4)


def conversion_options(args):
    if not args.tc_feedforward:
        return {}
    if args.physical_level != 2:
        raise ValueError('TC uses normalized conductances (physical_level=2), never pulse-level weights.')
    return dict(tc_options={key: getattr(args, key) for key in (
        'one_shot_conv', 'tc_method', 'tc_tol', 'tc_step_size',
        'tc_noise_reference_R', 'tc_covariance_table', 'tc_curve_sampling')})


def initialize_args(args):
    if args.tc_feedforward:
        if hasattr(args, 'physical_feedforward') and not args.physical_feedforward:
            raise ValueError('--tc_feedforward requires --physical_feedforward true.')
        if (hasattr(args, 'use_expanded_weights') and not args.use_expanded_weights
                and args.nonlinear_R_train_mode == 'none'):
            raise ValueError('TC physical evaluation requires unrolled convolutions; use exact_curve only for dense FT-protocol diagnostics.')
        for key in ('spin_variation_seed', 'summing_noise_seed', 'coupler_noise_seed',
                    'nonlinear_R_curve_seed', 'activation_curve_seed', 'data_seed'):
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, 4096)
    return args


def configure_pooling(model, args):
    from tc_nonidealities import prepare_tc_resistance_curves
    from measured_pooling import configure_feedforward_measured_pooling
    ref = next(model.parameters())
    package = prepare_tc_resistance_curves(
        args.measured_pooling_curve_path or args.nonlinear_R_table,
        args.tc_covariance_table, levels=torch.tensor([0., 1.]),
        R=args.measured_pooling_nominal_R, R_max=None,
        device=ref.device, dtype=ref.dtype)
    return configure_feedforward_measured_pooling(
        model, enable_nonideality=True, nominal_R=args.measured_pooling_nominal_R,
        seed=args.nonlinear_R_curve_seed, training_curve_mode='exact_curve',
        curve_gaussian=dict(v_grid=package.v_grid, mean=package.means[0],
                            factor=package.factor, value_scale=1., quantity='resistance'))
