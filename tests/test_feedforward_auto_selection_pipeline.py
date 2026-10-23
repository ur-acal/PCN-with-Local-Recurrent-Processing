"""Real shell/parser handoffs with isolated artifacts; no Slurm submission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def fake_python(argv):
    argv = [a for a in argv if a != '-u']
    if argv[0] == '-':
        os.execv(sys.executable, [sys.executable, *argv])
    sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
    if 'select_feedforward_one_over_q.py' in argv[0]:
        from diagnostic_scripts.select_feedforward_one_over_q import args_from_ft_command
        args = args_from_ft_command(argv[argv.index('--from_ft_command') + 1:])
        stage = 'selection'
        destination = Path(argv[argv.index('--output') + 1])
    elif 'run_feedforward_mc45_ablation.py' in argv[0]:
        from scripts.run_feedforward_mc45_ablation import parse_args
        sys.argv = argv
        args = parse_args()
        from data_utils import MC45CornerData
        catalog = MC45CornerData(args.mc_45_corner_dir,
            relu_monte_carlo_source=args.mc_relu_monte_carlo_source,
            coupler_nonlinear_variation_source=args.mc_coupler_nonlinear_variation_source,
            coupler_nonlinear_variation_quantity=args.mc_coupler_nonlinear_variation_quantity,
            coupler_nominal_R=args.mc_coupler_nominal_R)
        assert Path(catalog.coupler_source_path).is_dir()
        stage = 'eval'
    else:
        from baseline.train_baseline_cifar import parse_args
        sys.argv = argv
        args = parse_args()
        stage = 'pretrain' if args.physical_pretraining else 'ft'
    with open(os.environ['PIPELINE_TRACE'], 'a') as out:
        out.write(json.dumps({'stage': stage, 'args': vars(args)}, default=str) + '\n')
    if os.environ.get('FAIL_STAGE') == stage:
        return 42
    if stage == 'selection':
        assert Path(args.checkpoint).is_file()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'selected_one_over_q.txt').write_text('8\n')
        return 0
    if stage == 'eval':
        assert Path(args.checkpoint).is_file()
        return 0
    from test_tc_pipeline import fake_python as fake_training
    # Existing fixture generates the exact stage checkpoint names.
    trace = os.environ['PIPELINE_TRACE']
    os.environ['PIPELINE_TRACE'] = str(Path(trace).with_suffix('.fixture'))
    return fake_training(argv)


if __name__ == '__main__' and sys.argv[1:2] == ['--fake']:
    sys.exit(fake_python(sys.argv[2:]))

import pytest


def run_pipeline(directory, **overrides):
    root = Path(directory)
    binary = root / 'bin'
    binary.mkdir()
    python = binary / 'python'
    python.write_text(f'#!/bin/bash\nexec {sys.executable} {Path(__file__).resolve()} --fake "$@"\n')
    python.chmod(0o755)
    (binary / 'activate').write_text(':\n')
    env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
        REPO_ROOT=str(ROOT), mode='pretrain_ft_and_eval', TC_FEEDFORWARD='false',
        MODEL_NAME='wrn_flexible_cifar_avgpool_shortcut', WRN_DEPTH='28',
        WRN_FIRST_STAGE_CHANNELS='27', TASK='cifar100', IMG_TYPE='CiFAIR',
        PRETRAIN_OUTPUT_DIR=str(root / 'pre'), FT_OUTPUT_DIR=str(root / 'ft'),
        SELECTION_OUTPUT_DIR=str(root / 'selection'), RESULT_PATH=str(root / 'eval'),
        EXPANDED_WEIGHT_DIR=str(root / 'expanded'), PIPELINE_TRACE=str(root / 'trace'),
        EXP_PREFIX='isolated_test', FINAL_EVAL_ONLY='true', INPUT_QUANT_BITS='12',
        CENTER_STUDENT_INPUT='false', UNITLESS_MEASURED_PULLBACK_MODE='direct',
        PRETRAIN_ACTIVATION_CORNER_MODE='fixed', ACTIVATION_CORNER_MODE='random_per_forward',
        ACTIVATION_RANDOM_CURVE_SHARING='per_spin', FT_CORNER_COUPLED_SAMPLING='true',
        MC_RELU_MONTE_CARLO_SOURCE='1022_RELU_Voltage',
        NONLINEAR_R_TABLE='coupler_full_range_0p6_150pts', ENABLE_MEASURED_POOLING='true',
        R_VAL='50e3', C_VAL='500e-15', V_DD='0.6', TOGGLE_ONE_OVER_Q='6',
        TOGGLE_TIMING_MODE='fixed', TOGGLE_Y_TIME='10e-9', Z_OVER_Y_TIME='1',
        SCALE_TRAIN_RECIPE='1', WEIGHT_QUANT_FACTOR_BITS='1', ENOB='none',
        FINAL_HEAD_TYPE='digital', FINAL_ADC_NOISE_LSB='0.5', LINEAR_BIAS='false',
        CORNER_IDS='FS_V0_T0 FS_V2_T1', N_TRIALS='5')
    env.update(overrides)
    shell = '''conda() { :; }; export -f conda
sbatch() { bash "${@: -1}"; }
source launch_scripts/slurm_search_feedforward_config.sh
'''
    result = subprocess.run(['bash', '-c', shell], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=90)
    rows = [json.loads(line) for line in (root / 'trace').read_text().splitlines()] if (root / 'trace').exists() else []
    return result, rows


@pytest.mark.parametrize('model,task,head', [
    ('wrn_flexible_cifar_avgpool_shortcut', 'cifar100', 'digital'),
    ('resnet110_cifar', 'cifar10', 'analog'),
    ('wrn_28_2_cifar_nobn_no_bias_avgpool', 'cifar100', 'digital')])
def test_real_slurm_worker_stage_builders_and_parsers(model, task, head):
    with tempfile.TemporaryDirectory() as directory:
        result, rows = run_pipeline(directory, MODEL_NAME=model, TASK=task, FINAL_HEAD_TYPE=head)
        assert result.returncode == 0, result.stdout + result.stderr
        assert [r['stage'] for r in rows] == ['pretrain', 'selection', 'ft', 'eval']
        pre, selection, ft, evaluation = [r['args'] for r in rows]
        assert pre['activation_corner_mode'] == 'fixed'
        assert selection['activation_corner_mode'] == ft['activation_corner_mode'] == 'random_per_forward'
        assert pre['enable_pretrain_measured_activation']
        for args in (pre, selection, ft):
            assert args['activation_curve_path'].endswith('/1022_RELU_Voltage')
            assert args['activation_corner'] == 'TT_25_1_MC18'
        for args in (pre, selection, ft, evaluation):
            assert args['model_name'] == model
            assert args['dataset'] == task
            assert args['v_dd'] == .6
            assert args['toggle_y_time'] == 10e-9
            assert args['final_head_type'] == head
            assert args['final_adc_noise_lsb'] == .5
        assert pre['one_over_q'] == 6
        assert ft['one_over_q'] == evaluation['one_over_q'] == 8
        assert selection['ft_corner_coupled_sampling']
        assert selection['evaluation_split'] == 'train'
        assert selection['selection_training_hardware']
        assert not selection['bn_recalibrate'] and not selection['use_expanded_weights']
        assert selection['checkpoint'].endswith('_last_ckpt.pth')
        assert ft['resume_checkpoint'] == selection['checkpoint']
        assert evaluation['checkpoint'].endswith('_full_param_last_ckpt.pth')
        assert evaluation['corner_ids'] == ['FS_V0_T0', 'FS_V2_T1']
        assert evaluation['n_trials'] == 5
        assert evaluation['mc_relu_monte_carlo_source'] == '1022_RELU_Voltage'
        assert evaluation['mc_coupler_nonlinear_variation_source'].endswith('coupler_full_range_0p6_150pts')
        assert 'lr=0.1,' in pre['override'] and 'weight_decay=0.001' in pre['override']
        assert 'lr=0.005,' in ft['override']


def test_selection_failure_blocks_ft_and_eval():
    with tempfile.TemporaryDirectory() as directory:
        result, rows = run_pipeline(directory, FAIL_STAGE='selection')
        assert result.returncode != 0
        assert [r['stage'] for r in rows] == ['pretrain', 'selection']


def test_conflicting_banks_fail_before_pretrain():
    with tempfile.TemporaryDirectory() as directory:
        result, rows = run_pipeline(directory, MC_COUPLER_NONLINEAR_VARIATION_SOURCE='coupler_monte')
        assert result.returncode != 0 and not rows
        assert 'Conflicting FT/evaluation coupler' in result.stderr


def test_paired_bn_evaluation_changes_only_calibration_and_destination():
    with tempfile.TemporaryDirectory() as directory:
        result, rows = run_pipeline(directory, RUN_BN_RECAL_EVAL='true')
        assert result.returncode == 0, result.stdout + result.stderr
        assert [r['stage'] for r in rows] == ['pretrain', 'selection', 'ft', 'eval', 'eval']
        raw, recal = [r['args'] for r in rows[-2:]]
        assert not raw['bn_recalibrate'] and recal['bn_recalibrate']
        assert recal['bn_calibration_samples'] == 5120
        assert recal['bn_calibration_batch_size'] == 256
        changed = {key for key in raw if raw[key] != recal[key]}
        assert changed == {'output_dir', 'bn_recalibrate', 'bn_calibration_samples', 'bn_calibration_batch_size'}
        assert recal['output_dir'] == raw['output_dir'] + '_bnrecal'
        assert raw['expanded_weight_dir'] == recal['expanded_weight_dir']


def test_first_evaluation_failure_blocks_paired_calibration():
    with tempfile.TemporaryDirectory() as directory:
        result, rows = run_pipeline(directory, RUN_BN_RECAL_EVAL='true', FAIL_STAGE='eval')
        assert result.returncode != 0
        assert [r['stage'] for r in rows] == ['pretrain', 'selection', 'ft', 'eval']


@pytest.mark.parametrize('head', ['analog', 'digital'])
def test_actual_selection_loads_unitless_checkpoint_and_draws_coupled_hardware(tmp_path, head):
    from baseline import evaluate_physical_feedforward_cifar as evaluator
    from baseline.baseline_cifar_configs import get_baseline_config, build_model, RGGB_DEFAULTS
    from diagnostic_scripts.select_feedforward_one_over_q import args_from_ft_command, select
    from final_linear import select_model_head, config_from_args, configure_feedforward_head
    from physical_feedforward import convert_wide_resnet_to_physical
    from torch.utils.data import DataLoader, TensorDataset
    source = tmp_path / 'source.pth'
    argv = ['--model_name', 'wrn_flexible_cifar_avgpool_shortcut', '--dataset', 'cifar10', '--img_type', 'CiFAIR',
        '--resume_checkpoint', str(source), '--wrn_depth', '10', '--wrn_first_stage_channels', '16',
        '--physical_feedforward', 'true', '--physical_level', '2',
        '--linear_bias', 'false', '--final_head_type', head, '--final_adc_noise_lsb', '0.5',
        '--R', '50000', '--C', '5e-13', '--v_dd', '0.6', '--one_over_q', '6',
        '--toggle_timing_mode', 'fixed', '--toggle_y_time', '1e-8', '--enob', 'none',
        '--w_bits', '5', '--weight_quant_factor_bits', '1',
        '--nonlinear_R_train_mode', 'exact_curve', '--nonlinear_R_curve_sharing', 'shared',
        '--nonlinear_R_table', str(ROOT/'hardware_data/mc_45_corners/coupler_full_range_0p6_150pts'),
        '--enable_measured_activation', 'true', '--enable_measured_pooling', 'true',
        '--activation_curve_path', str(ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage'),
        '--activation_corner', 'TT_25_1_MC18', '--activation_corner_mode', 'random_per_forward',
        '--activation_random_curve_sharing', 'per_spin', '--ft_corner_coupled_sampling', 'true',
        '--fuse_measured_activation', 'false']
    with patch.dict(os.environ, SELECTION_SAMPLES='4', SELECTION_BATCHES='2', SELECTION_BATCH_SIZE='2'):
        args = args_from_ft_command(argv)
    torch.manual_seed(7)
    cfg = get_baseline_config(args.model_name, case='custom_noresize')
    cfg.update(RGGB_DEFAULTS, wrn_depth=10, wrn_first_stage_channels=16, linear_bias=False)
    model = build_model(args.model_name, cfg, 10)
    select_model_head(model, head, config_from_args(args))
    convert_wide_resnet_to_physical(model, physical=False, qat=False, physical_level=2,
        R=args.R, C=args.C, v_dd=.6, one_over_q=6, toggle_timing_mode='fixed', toggle_y_time=1e-8)
    configure_feedforward_head(model, args, physical=False)
    inputs = torch.randn(4, 4, 8, 8) * .1
    # A real short unitless optimizer step before saving the source checkpoint.
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.zero_grad()
    model(inputs[:2]).square().mean().backward()
    optimizer.step()
    torch.save({'net': model.state_dict(), 'input_quant_bits': None,
                'center_student_input': False, 'measured_activation_scope': 'all'}, source)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(4, dtype=torch.long)), batch_size=2)
    checks = []
    original = evaluator.configure_selection_hardware_sampling
    def check_sampling(model, args):
        original(model, args)
        norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        before = [m.running_mean.clone() for m in norms]
        assert all(not m.training for m in norms)
        assert not model.training  # Functional final dropout also remains disabled.
        assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
        def after(module, inputs, output):
            for norm, mean in zip(norms, before):
                torch.testing.assert_close(norm.running_mean, mean)
            assert module._last_ft_corner_coupled_corner is not None
            corner = module._last_ft_corner_coupled_corner
            from physical_feedforward import iter_physical_blocks
            from measured_activation import MEASURED_ACTIVATION_TYPES
            from measured_pooling import MeasuredAvgPool2d
            for block in iter_physical_blocks(module):
                allowed = block._nonlinear_R_training_pkg['ft_corner_curve_indices'][corner]
                assert set(block._nonlinear_R_training_curve_indices.values()) <= set(allowed)
            for activation in module.modules():
                if isinstance(activation, MEASURED_ACTIVATION_TYPES):
                    assert set(activation._sampled_curve_indices.flatten().tolist()) <= set(
                        activation._ft_corner_curve_indices[corner])
                if isinstance(activation, MeasuredAvgPool2d):
                    for assignment in activation._curve_assignments.values():
                        assert set(assignment.flatten().tolist()) <= set(activation._ft_corner_curve_indices[corner])
            assert torch.isfinite(output).all()
            checks.append(module._last_ft_corner_coupled_corner)
        model.register_forward_hook(after)
    with patch.object(evaluator, 'get_bn_calibration_data', return_value=loader), \
         patch.object(evaluator, 'configure_selection_hardware_sampling', side_effect=check_sampling), \
         patch('torch.cuda.is_available', return_value=False):
        selected = select(args, [6, 8], tmp_path/'selection', sanity_tolerance=100.)
    assert selected in (6, 8) and len(checks) == 4
    assert source.is_file()


