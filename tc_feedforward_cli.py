"""Opt-in wiring for resistance-coded feedforward CNNs."""
import os
import torch
from checkpoint_memory_profiler import parse_checkpoint_ode_rhs_portion


def add_arguments(parser):
    parser.add_argument('--tc_empirical_curve_bank', default=None)
    if '--nonlinear_R_curve_sampling' not in parser._option_string_actions:
        parser.add_argument('--nonlinear_R_curve_sampling',
            choices=('multivariate_gaussian', 'empirical_with_replacement'))
    parser.set_defaults(nonlinear_R_curve_sampling=None)
    from resistance_guard import add_argument
    add_argument(parser)
    from compute_time_variation import add_arguments
    add_arguments(parser)
    def boolean(value):
        if value.lower() not in ('true', 'false'):
            raise ValueError('Expected true or false')
        return value.lower() == 'true'
    parser.add_argument('--tc_feedforward', type=boolean, default=False)
    parser.add_argument('--tc_intermediate_activation',
                        choices=('relu', 'relu6'), default='relu6')
    parser.add_argument('--one_shot_conv', type=boolean, default=False)
    parser.add_argument('--tc_method', choices=('dopri5', 'euler'), default='dopri5')
    parser.add_argument('--tc_solver_dense_output', type=boolean, default=True)
    parser.add_argument('--tc_noise_replay_cache', type=boolean, default=True,
                        help='Retain noise history; false releases accepted intervals in no-grad inference only.')
    parser.add_argument('--reuse_accepted_step_training', type=boolean,
                        default=boolean(os.environ.get(
                            'REUSE_ACCEPTED_STEP_TRAINING', 'false')))
    parser.add_argument('--checkpoint_ode_rhs_training', type=boolean,
                        default=boolean(os.environ.get(
                            'CHECKPOINT_ODE_RHS_TRAINING', 'false')))
    parser.add_argument('--checkpoint_ode_rhs_portion',
                        type=parse_checkpoint_ode_rhs_portion,
                        default=parse_checkpoint_ode_rhs_portion(os.environ.get(
                            'CHECKPOINT_ODE_RHS_PORTION', '1.0')))
    parser.add_argument('--tc_tol', type=float, default=1e-6)
    parser.add_argument('--tc_step_size', type=float, default=None)
    parser.add_argument('--tc_num_steps', type=int, default=None,
                        help='Euler steps per convolution; mutually exclusive with tc_step_size.')
    parser.add_argument('--tc_noise_reference_R', type=float, default=50e3)
    parser.add_argument('--tc_covariance_table', default='./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
    parser.add_argument('--tc_curve_sampling', choices=('histogram', 'uniform'), default='histogram')
    parser.add_argument('--tc_curve_top_n', type=int, default=0,
        help='With tc_conv_method=shared: separately represent the top N occupied weight levels (0 through the nonzero level count).')
    parser.add_argument('--tc_conv_method', choices=('shared', 'loop', 'grouped'), default='shared')
    parser.add_argument('--measured_pooling_curve_path', default=None)
    parser.add_argument('--measured_pooling_nominal_R', type=float, default=None)


def conversion_options(args):
    if not args.tc_feedforward:
        return {}
    if args.physical_level != 2:
        raise ValueError('TC uses normalized conductances (physical_level=2), never pulse-level weights.')
    result = dict(tc_options={key: getattr(args, key) for key in (
        'one_shot_conv', 'tc_method', 'tc_tol', 'tc_step_size', 'tc_num_steps',
        'tc_noise_reference_R', 'tc_covariance_table', 'tc_curve_sampling',
        'reuse_accepted_step_training', 'checkpoint_ode_rhs_training')})
    for key in ('tc_solver_dense_output', 'tc_noise_replay_cache'):
        result['tc_options'][key] = getattr(args, key, True)
    result['tc_options']['tc_distribution'] = args.nonlinear_R_curve_sampling
    result['tc_options']['tc_conv_method'] = getattr(args, 'tc_conv_method', 'shared')
    result['tc_options']['tc_curve_top_n'] = getattr(args, 'tc_curve_top_n', 0)
    result['tc_options']['tc_empirical_curve_bank'] = getattr(args, 'tc_empirical_curve_bank', None)
    return result


def initialize_args(args):
    from tc_empirical import sampling_mode
    sampling_mode(args)
    if getattr(args, 'measured_pooling_nominal_R', None) is None:
        args.measured_pooling_nominal_R = (getattr(args, 'R', 13.3e3)
            if args.tc_feedforward and args.nonlinear_R_curve_sampling == 'empirical_with_replacement' else 1e4)
    if args.tc_feedforward:
        from tc_curve_groups import validate_top_n
        validate_top_n(getattr(args, 'tc_curve_top_n', 0), 2**(args.w_bits-1)-1)
        if getattr(args, 'tc_curve_top_n', 0) and args.tc_curve_sampling != 'histogram':
            raise ValueError('tc_curve_top_n > 0 requires histogram sampling.')
        if (getattr(args, 'tc_curve_top_n', 0) and args.tc_conv_method != 'shared'
                and not getattr(args, 'use_expanded_weights', False)):
            raise ValueError('tc_curve_top_n > 0 requires tc_conv_method=shared for dense training.')
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
    from tc_empirical import is_npz_bank
    if (getattr(args, 'nonlinear_R_curve_sampling', 'multivariate_gaussian') == 'empirical_with_replacement'
            and not is_npz_bank(getattr(args, 'tc_empirical_curve_bank', None))):
        from tc_empirical import pooling_options
        from measured_pooling import configure_feedforward_measured_pooling
        return configure_feedforward_measured_pooling(model, enable_nonideality=True,
            seed=args.nonlinear_R_curve_seed, **pooling_options(args, next(model.parameters())))
    # NPZ finite coupler banks retain the original Gaussian pooling configuration.
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
                            factor=package.factor, value_scale=1., quantity='resistance',
                            tc_covariance_path=package.covariance_path),
        resistance_floor_ohms=getattr(args, 'resistance_floor_ohms', None))
