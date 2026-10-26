"""Shared opt-in TC configuration, trial reset and provenance."""
import json
from argparse import ArgumentTypeError
from pathlib import Path

NOISE_KEYS = ('enable_spin_variation','sigma_spin','spin_variation_mean','spin_variation_seed',
    'enable_summing_current_noise','summing_current_p','summing_noise_seed',
    'enable_coupler_noise','coupler_noise_p','coupler_noise_seed',
    'tc_noise_stages','tc_noise_reference_R','tc_asd_reference_p','tc_fb_asd_path')

def add_tc_arguments(parser):
    if '--nonlinear_R_curve_sampling' not in parser._option_string_actions:
        parser.add_argument('--nonlinear_R_curve_sampling', default=None,
                            choices=('multivariate_gaussian', 'empirical_with_replacement'))
    else:
        parser.set_defaults(nonlinear_R_curve_sampling=None)
    from resistance_guard import add_argument
    add_argument(parser)
    from compute_time_variation import add_arguments
    add_arguments(parser)
    def boolean(value):
        if str(value).lower() in ('true','1','yes'):
            return True
        if str(value).lower() in ('false','0','no'):
            return False
        raise ArgumentTypeError('Expected true or false.')
    options=dict(tc_nonidealities=(boolean,False),tc_covariance_table=(str,None),tc_conv_method=(str,'loop'),
        tc_curve_sampling=(str,'histogram'),tc_curve_top_n=(int,0),tc_empirical_curve_bank=(str,None),
        tc_solver_dense_output=(boolean,True),tc_noise_replay_cache=(boolean,True),
        measured_pooling_curve_path=(str,None),measured_pooling_nominal_R=(float,None),
        measure_coupler_energy=(boolean,False),coupler_supply_voltage=(float,1.3),
        tc_coupler_energy_path=(str,None),
        spin_variation_mean=(float,1.0),
        tc_noise_stages=(str,'both'),
        tc_noise_reference_R=(float,50e3),tc_asd_reference_p=(float,.6e-12),
        tc_fb_asd_path=(str,str(Path(__file__).parent/'hardware_data/coupler_asd_vs_freq.csv')),
        tc_metadata_path=(str,None),tc_max_eval_batches=(int,0))
    for key,(kind,default) in options.items():
        if '--'+key not in parser._option_string_actions:
            parser.add_argument('--'+key,type=kind,default=default)
    parser._option_string_actions['--tc_curve_top_n'].help = (
        'With tc_conv_method=shared: give the top N occupied nonzero weight levels '
        'individual curves and histogram-share the remainder (0 through the nonzero level count; default 0). '
        'Expanded per-coupler evaluation is unchanged.')
    if '--expanded_gaussian_curve_cpu_offload' not in parser._option_string_actions:
        parser.add_argument(
            '--expanded_gaussian_curve_cpu_offload',
            '--tc_curve_cpu_offload',
            dest='expanded_gaussian_curve_cpu_offload',
            type=boolean, default=False,
            help='Offload static expanded Gaussian per-coupler curves to CPU.')

def resolve_eval_batch_limit(args):
    """Resolve the legacy TC batch-limit name into the shared option."""
    canonical = getattr(args, 'max_eval_batches', 0)
    legacy = getattr(args, 'tc_max_eval_batches', 0)
    if canonical < 0:
        raise ValueError('max_eval_batches must be nonnegative (0 means full dataset).')
    if legacy < 0:
        raise ValueError('tc_max_eval_batches must be nonnegative (0 means full dataset).')
    if canonical and legacy and canonical != legacy:
        raise ValueError(
            'Conflicting evaluation batch limits: max_eval_batches={} and '
            'tc_max_eval_batches={}.'.format(canonical, legacy))
    args.max_eval_batches = canonical or legacy
    return args.max_eval_batches

def wrapper_options(args, trial=None):
    result={key:getattr(args,key) for key in NOISE_KEYS}
    result.update(tc_nonidealities=True,tc_covariance_table=args.tc_covariance_table,
                  tc_conv_method=args.tc_conv_method,tc_curve_sampling=args.tc_curve_sampling)
    result['tc_empirical_curve_bank'] = getattr(args, 'tc_empirical_curve_bank', None)
    result['tc_curve_top_n'] = getattr(args, 'tc_curve_top_n', 0)
    result['nonlinear_R_curve_sampling'] = getattr(args, 'nonlinear_R_curve_sampling', None) or 'multivariate_gaussian'
    result['resistance_floor_ohms'] = getattr(args, 'resistance_floor_ohms', None)
    for key in ('tc_solver_dense_output', 'tc_noise_replay_cache'):
        result[key] = getattr(args, key, True)
    if trial is not None:
        base=args.hardware_seed if args.hardware_seed is not None else args.data_seed
        for key in ('spin_variation_seed','summing_noise_seed','coupler_noise_seed',
                    'nonlinear_R_curve_seed','activation_curve_seed'):
            seed=getattr(args,key,None)
            result[key]=(base if seed is None else seed)+trial
    return result

def validate_expanded_gaussian_curve_offload(args, inference=False):
    if not getattr(args, 'expanded_gaussian_curve_cpu_offload', False):
        return
    if not inference:
        raise ValueError('Expanded Gaussian curve CPU offload is inference-only.')
    if not args.nonlinear_R or not args.test_expanded:
        raise ValueError(
            'Expanded Gaussian curve CPU offload requires expanded nonlinear-R evaluation.')
    if args.nonlinear_R_curve_sharing != 'per_coupler':
        raise ValueError(
            'Expanded Gaussian curve CPU offload requires per-coupler curves.')
    if getattr(args, 'tc_nonidealities', False):
        if getattr(args, 'nonlinear_R_curve_sampling', None) == 'empirical_with_replacement':
            raise ValueError(
                'Expanded Gaussian curve CPU offload does not support empirical banks.')
    else:
        if args.nonlinear_R_curve_sampling != 'multivariate_gaussian':
            raise ValueError(
                'Expanded Gaussian curve CPU offload requires multivariate-Gaussian sampling.')
        if not str(args.ode_block).startswith('TogglePulse'):
            raise ValueError(
                'Generic PCN curve offload requires a TogglePulse ODE block.')


def validate_tc(args, inference=False):
    from tc_empirical import sampling_mode
    sampling_mode(args)
    validate_expanded_gaussian_curve_offload(args, inference=inference)
    if not args.tc_nonidealities:return
    from tc_curve_groups import validate_top_n
    validate_top_n(getattr(args, 'tc_curve_top_n', 0), 2**(args.w_bits-1)-1)
    if getattr(args, 'tc_curve_top_n', 0) and args.tc_curve_sampling != 'histogram':
        raise ValueError('tc_curve_top_n > 0 requires histogram sampling.')
    if not inference and getattr(args, 'tc_curve_top_n', 0) and args.tc_conv_method != 'shared':
        raise ValueError('tc_curve_top_n > 0 requires tc_conv_method=shared for training.')
    if args.tc_max_eval_batches < 0:
        raise ValueError('tc_max_eval_batches must be nonnegative (0 means full dataset).')
    if args.tc_conv_method not in ('loop','grouped','shared'):
        raise ValueError('tc_conv_method must be loop, grouped or shared.')
    if args.tc_curve_sampling not in ('histogram','uniform'):
        raise ValueError('tc_curve_sampling must be histogram or uniform.')
    if args.tc_noise_stages not in ('both','ff','fb'):
        raise ValueError('tc_noise_stages must be both, ff or fb.')
    if args.measure_coupler_energy and not inference:
        raise ValueError('Coupler-energy measurement is inference-only.')
    if args.measure_coupler_energy and args.ode_block != 'ODEXInitFFFB':
        raise ValueError('Coupler-energy measurement currently supports one-state ODEXInitFFFB.')
    if args.ode_block not in ('ODEXInitFFFB','S2NoisyIYAsXZAs0'):
        raise ValueError('TC requires the ordinary one-/two-state block, not toggle/switched.')
    if args.weight_quant_factor_bits is not None or args.enob is not None:
        raise ValueError('TC requires weight_quant_factor_bits=none and enob=none.')
    if args.w_bits!=5:raise ValueError('The selected TC recipe uses 5-bit weights.')
    if any(getattr(args,key,False) for key in ('enable_slow_summing_current','enable_slow_coupler_noise','enable_dtc_nonideality')):
        raise ValueError('TC has no slow offsets or DTC in this recipe.')
    gaussian = args.nonlinear_R_curve_sampling == 'multivariate_gaussian'
    if gaussian and args.enable_measured_pooling and not args.measured_pooling_curve_path:
        raise ValueError('TC pooling needs its own measured_pooling_curve_path.')
    if gaussian and args.nonlinear_R and not args.tc_covariance_table:
        raise ValueError('TC nonlinear-R needs tc_covariance_table as well as the mean table.')
    if inference:
        if args.noisy_trials < 1:
            raise ValueError('TC evaluation requires at least one trial.')
        state = '1' if args.ode_block == 'ODEXInitFFFB' else '2'
        if args.ode_wrapper not in ('ODEWrapper'+state+'State','QATTester'+state+'State'):
            raise ValueError('TC evaluator wrapper must match the selected state count.')
        if args.analyze_mode is not None or args.sweep_eps:
            raise ValueError('TC uses its own diffusion, not the legacy analysis/epsilon sweep.')
        if not args.test_expanded or args.test_only or args.hw_validate:
            raise ValueError('TC trial evaluation requires test_expanded=true, test_only=false, hw_validate=false.')
        if args.diff_mismatch or any(float(x) for x in args.noise_level_list.split(',')):
            raise ValueError('TC covariance replaces scalar mismatch; use diff_mismatch=false and noise_level_list=0.')
    else:
        state = '1' if args.ode_block == 'ODEXInitFFFB' else '2'
        if args.noise_level not in (None,0) or args.ode_wrapper != 'QATWrapper'+state+'State':
            raise ValueError('TC training requires the matching TC QAT wrapper and noise_level=0.')

def reset_after_probe(model):
    """Keep coupler realizations; discard dry-run spin/dynamic/pooling draws."""
    for block in model.PcConvs:
        if getattr(block,'_tc_current_mode',False):
            block.reset_tc_spin()
            block._tc_generators.clear()
    for module in model.modules():
        reset = getattr(module, 'reset_after_probe', None)
        if callable(reset):
            reset()
        if hasattr(module,'reset_measured_pooling'):
            module.reset_measured_pooling()
            generator=getattr(module,'_generator',None)
            if generator is not None:generator.manual_seed(generator.initial_seed())

        if hasattr(module,'_sampled_curve_indices'):
            module._sampled_curve_indices=None
            generator=getattr(module,'_curve_generator',None)
            if generator is not None:generator.manual_seed(generator.initial_seed())

def pooling_options(args, wrappers):
    """Fit the selected pooling mean with the common TC absolute covariance.

    Pooling has its own nominal resistance, independent of the MVM code grid.
    Reuse the existing nearest-column selection and voltage-grid alignment.
    """
    import torch
    from tc_nonidealities import prepare_tc_resistance_curves
    from tc_empirical import is_npz_bank
    if (getattr(args, 'nonlinear_R_curve_sampling', None) == 'empirical_with_replacement'
            and not is_npz_bank(getattr(args, 'tc_empirical_curve_bank', None))):
        from tc_empirical import pooling_options as empirical_pooling
        return empirical_pooling(args, wrappers[0].ode_block.FFconv.weight)
    # A legacy NPZ replaces only couplers; keep independently configured pooling.
    block = wrappers[0].ode_block
    nominal_R = args.measured_pooling_nominal_R
    if nominal_R is None:
        nominal_R = 10e3
    ref = block.FFconv.weight
    package = prepare_tc_resistance_curves(
        args.measured_pooling_curve_path, args.tc_covariance_table,
        levels=torch.tensor([0., 1.]), R=nominal_R, R_max=None,
        dtype=ref.dtype, device=ref.device)
    return dict(curve_path=None, nominal_R=float(nominal_R),
                curve_gaussian=dict(v_grid=package.v_grid, mean=package.means[0],
                    factor=package.factor, value_scale=1.0, quantity='resistance',
                    tc_covariance_path=package.covariance_path),
                resistance_floor_ohms=getattr(args, 'resistance_floor_ohms', None))

def record_trial(args, model, trial, accuracy, checkpoint):
    from compute_time_variation import records
    path=Path(args.tc_metadata_path or Path(args.hw_val_path)/args.model_name/'tc_trials.jsonl')
    path.parent.mkdir(parents=True,exist_ok=True)
    layers=[]
    for block in model.PcConvs:
        entry={'layer':block.layer_idx,'noise':getattr(block,'_tc_noise_cfg',{}),
               'physical_duration':float(block.integration_time[-1])}
        entry['coupler_seeds']={name:getattr(getattr(block,name),'nonlinear_R_curve_seed',None)
                                for name in ('FFconv','FBconv')}
        layers.append(entry)
    record=dict(trial=trial,accuracy_percent=accuracy,checkpoint=str(Path(checkpoint).resolve()),
                data_seed=args.data_seed+trial,resolved_trial_options=wrapper_options(args,trial),
                pooling_seed=(args.hardware_seed if args.hardware_seed is not None else args.data_seed)+trial,
                working_directory=str(Path.cwd()),options=vars(args),layers=layers,
                compute_time_variation=records(model))
    with path.open('a') as handle:handle.write(json.dumps(record,default=str)+'\n')
