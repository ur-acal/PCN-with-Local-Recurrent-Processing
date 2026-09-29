"""MNIST recipe/CLI only; physical parameters retain the shared TC defaults."""
import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from mnist_train_eval.mnist_data import DEFAULT_DATA_DIR

ROOT = Path(__file__).resolve().parents[1]


def boolean(value):
    if str(value).lower() in ('true', '1', 'yes'):
        return True
    if str(value).lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def parse_args(argv=None, *, evaluation=False):
    p = argparse.ArgumentParser(description='MNIST TC CNN/PCN, no distillation')
    p.add_argument('--family', choices=('cnn', 'pcn'), default='cnn')
    p.add_argument('--variant', choices=('small', 'deep'), default='small')
    p.add_argument('--tc_state', type=int, choices=(1, 2), default=1)
    p.add_argument('--stage', choices=('pretrain', 'ft', 'eval'), default='eval' if evaluation else 'pretrain')
    p.add_argument('--data_dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--output_dir', type=Path, default=ROOT / 'saved_ckpt_runs/mnist_tc')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--expanded_weight_dir', type=Path, default=ROOT / 'expanded_weights/mnist')
    p.add_argument('--epochs', type=int, default=14)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--test_batch_size', type=int, default=1000)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--optimizer', choices=('adadelta', 'adam', 'sgd'), default='adadelta')
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--rho', type=float, default=0.9)
    p.add_argument('--eps', type=float, default=1e-6)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.)
    p.add_argument('--scheduler', choices=('step', 'constant', 'cosine'), default='step')
    p.add_argument('--step_size', type=int, default=1)
    p.add_argument('--gamma', type=float, default=0.7)
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=4096)
    p.add_argument('--health_check_epochs', default='5,10')
    p.add_argument('--health_check_batches', type=int, default=4)
    p.add_argument('--n_trials', type=int, default=10)
    p.add_argument('--t_end', type=float, default=1.75)
    p.add_argument('--tol', type=float, default=None)
    p.add_argument('--reuse_accepted_step_training', type=boolean,
                   default=boolean(os.environ.get('REUSE_ACCEPTED_STEP_TRAINING', 'false')))
    p.add_argument('--checkpoint_ode_rhs_training', type=boolean,
                   default=boolean(os.environ.get('CHECKPOINT_ODE_RHS_TRAINING', 'false')))
    p.add_argument('--one_shot_conv', type=boolean, default=False)
    p.add_argument('--download', type=boolean, default=False)
    p.add_argument('--dry_run', type=boolean, default=False)
    p.add_argument('--limit_train_samples', type=int, default=0, help='Smoke tests only; zero uses all training samples.')
    p.add_argument('--limit_test_samples', type=int, default=0, help='Smoke tests only; zero uses all test samples.')
    p.add_argument('--mem_frac', type=float, default=0.9)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    # Same environment spellings/defaults as tc_hardware_defaults.sh. A launcher
    # sources that file; direct Python invocation has the identical defaults.
    hardware = {
        'R': ('R_VAL', float, 10e3), 'R_max': ('R_MAX', float, 150e3),
        'C': ('C_VAL', float, 49e-15), 'v_dd': ('V_DD', float, .1),
        'one_over_q': ('TOGGLE_ONE_OVER_Q', float, 1.),
        'nonlinear_R_table': ('TC_MEAN_TABLE', str, './hardware_data/res_vs_vin_10k_150k.csv'),
        'tc_covariance_table': ('TC_COVARIANCE_TABLE', str, './hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'),
        'tc_curve_sampling': ('TC_CURVE_SAMPLING', str, 'histogram'),
        'tc_conv_method': ('TC_CONV_METHOD', str, 'shared'),
        'tc_noise_reference_R': ('TC_NOISE_REFERENCE_R', float, 50e3),
        'tc_asd_reference_p': ('TC_ASD_REFERENCE_P', float, .6e-12),
        'tc_fb_asd_path': ('TC_FB_ASD_PATH', str, './hardware_data/coupler_asd_vs_freq.csv'),
        'enable_spin_variation': ('ENABLE_SPIN_VARIATION', boolean, True),
        'sigma_spin': ('SIGMA_SPIN', float, .1),
        'spin_variation_mean': ('SPIN_VARIATION_MEAN', float, 1.),
        'enable_summing_current_noise': ('ENABLE_SUMMING_CURRENT_NOISE', boolean, True),
        'summing_current_p': ('SUMMING_CURRENT_P', float, .6e-12),
        'enable_coupler_noise': ('ENABLE_COUPLER_NOISE', boolean, True),
        'coupler_noise_p': ('COUPLER_NOISE_P', float, .6e-12),
        'enable_nonlinear_R': ('ENABLE_NONLINEAR_R', boolean, True),
        'enable_measured_activation': ('ENABLE_MEASURED_ACTIVATION', boolean, True),
        'fuse_measured_activation': ('FUSE_MEASURED_ACTIVATION', boolean, True),
        'enable_measured_pooling': ('ENABLE_MEASURED_POOLING', boolean, True),
        'measured_pooling_nominal_R': ('MEASURED_POOLING_NOMINAL_R', float, 10e3),
    }
    for name, (env, kind, default) in hardware.items():
        p.add_argument('--' + name, type=kind, default=kind(os.environ.get(env, default)))
    p.add_argument('--activation_curve_path', default=None)
    p.add_argument('--activation_corner', default=None)
    p.add_argument('--measured_pooling_curve_path', default=os.environ.get('MEASURED_POOLING_CURVE_PATH'))
    args = p.parse_args(argv)
    if evaluation and args.stage != 'eval':
        p.error('mnist_evaluate requires --stage eval')
    if not evaluation and args.stage == 'eval':
        p.error('Use mnist_evaluate for evaluation')
    if args.lr is None:
        args.lr = .01 if args.stage == 'ft' else 1.
    if args.tol is None:
        args.tol = 1e-4 if args.stage == 'pretrain' else 1e-6
    if args.stage in ('ft', 'eval') and not args.checkpoint:
        p.error('FT/evaluation requires --checkpoint')
    for name in ('epochs', 'batch_size', 'test_batch_size', 'step_size', 'n_trials'):
        if getattr(args, name) < 1:
            p.error(name + ' must be positive')
    if not 0 < args.mem_frac <= 1 or not 0 <= args.dropout < 1:
        p.error('Invalid mem_frac or dropout')
    if args.tc_conv_method not in ('shared', 'grouped', 'loop') or args.tc_curve_sampling not in ('uniform', 'histogram'):
        p.error('Invalid TC convolution/sampling method')
    if args.family == 'cnn':
        if args.tc_conv_method != 'shared':
            p.error('The existing TC CNN supports the shared-curve training method only.')
        if not math.isclose(args.R_max, 15 * args.R):
            p.error('The existing 5-bit TC CNN maps R_max=15*R; these must agree.')
    bank = './hardware_data/mc_45_corners/0906_RELU_Voltage'
    args.activation_curve_path = args.activation_curve_path or (bank if evaluation else bank + '/tt_25_1.csv')
    args.activation_corner = args.activation_corner or ('TT_25_1_MC18' if evaluation else 'MC18')
    args.measured_pooling_curve_path = args.measured_pooling_curve_path or args.nonlinear_R_table
    for name in ('nonlinear_R_table', 'tc_covariance_table', 'tc_fb_asd_path', 'activation_curve_path', 'measured_pooling_curve_path'):
        path = Path(getattr(args, name))
        setattr(args, name, str(path if path.is_absolute() else ROOT / path))
    return args


def model_name(args):
    if args.family == 'cnn':
        return 'mnist_cnn{}_avgpool'.format(3 if args.variant == 'small' else 5)
    return 'mnist_pcn{}_state{}'.format(2 if args.variant == 'small' else 3, args.tc_state)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(args):
    if args.device == 'cpu':
        # Existing PCN constructors/trainer select CUDA when visible.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    return device
