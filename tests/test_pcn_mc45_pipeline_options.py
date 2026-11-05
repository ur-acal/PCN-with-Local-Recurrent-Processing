"""Exercise the real PCN evaluation function without launching inference."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def resolve_eval(tmp_path, **overrides):
    launcher = ROOT / 'launch_scripts/run_kdcrd_then_ft.sbatch'
    lines = launcher.read_text().splitlines()
    stop = next(i for i, line in enumerate(lines) if line.startswith('# PHASE 1: TRAIN'))
    env = dict(os.environ, IS_SLURM='0', TOGGLE_MODE='odexinit', TASK='cifar100',
        IMG_TYPE='CiFAIR', LOGDIR=str(tmp_path/'logs'),
        OUTPUT_SAVE_PATH=str(tmp_path/'checkpoints'),
        PRETRAIN_SAVE_PATH=str(tmp_path/'checkpoints'),
        FT_OUTPUT_SAVE_PATH=str(tmp_path/'checkpoints'),
        COMB_LIST='example\t1\t4 24\t24 24\t0 0',
        NONLINEAR_R_TABLE='coupler_monte_v2_tt_25_1_mc34_singleton.csv')
    for key in ('POST_FT_EVAL_CORNER_IDS', 'POST_FT_EVAL_N_TRIALS',
                'MC_COUPLER_NONLINEAR_VARIATION_SOURCE', 'EXPANDED_WEIGHT_DIR'):
        env.pop(key, None)
    invocation = overrides.pop('_invocation',
        'eval_one_combo_model example TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_test post_ft "${INF_ODE_BLOCK}"')
    env.update(overrides)
    script = f'''
source <(head -n {stop} "{launcher}")
trap - TERM INT
python() {{ printf 'EVAL_ARG:%s\\n' "$@"; }}
{invocation}
'''
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return [line[len('EVAL_ARG:'):] for line in result.stdout.splitlines()
            if line.startswith('EVAL_ARG:')]


def test_legacy_two_corner_defaults_unchanged(tmp_path):
    args = resolve_eval(tmp_path)
    i = args.index('--corner_ids')
    assert args[i+1:i+3] == ['FS_V0_T0', 'FS_V2_T1']
    assert args[args.index('--n_trials')+1] == '5'
    assert args[args.index('--mc_coupler_nonlinear_variation_source')+1].endswith('_singleton.csv')
    assert args[args.index('--expanded_w_dir')+1] == 'expanded_weights'


def test_s2_dtc_pipeline_forwards_joint_solver_and_initialization(tmp_path):
    for mode in ('s2_dtc', 's2_dtc_pulse'):
        args = resolve_eval(tmp_path, TOGGLE_MODE=mode, S2_DTC_METHOD='rk2',
                            S2_DTC_SUBSTEPS='1', S2_DTC_EVAL_SUBSTEPS='3',
                            S2_Y_INIT_WITH='0', S2_Z_INIT_WITH='x')
        assert args[args.index('--dtc_dynamics')+1] == 's2'
        assert args[args.index('--s2_dtc_method')+1] == 'rk2'
        assert args.count('--s2_dtc_substeps') == 1
        assert args[args.index('--s2_dtc_substeps')+1] == '3'
        assert args[args.index('--y_init_with')+1] == '0'
        assert args[args.index('--z_init_with')+1] == 'x'


def test_s2_dtc_scheduler_exports_without_submitting():
    for mode in ('s2_dtc', 's2_dtc_pulse'):
        env = dict(os.environ, TC_DRY_RUN='true', TOGGLE_MODE=mode,
                   TASK='cifar100', IMG_TYPE='CiFAIR', SEARCH_ARCH='two_stage_fixed',
                   PCN_CHAN_0_LIST='36', PCN_NUM_LAYERS_LIST='8:10',
                   TWO_STAGE_POOL_POSITION='before_expansion', NUM_COMB_PER_NUM_LAYER='1',
                   S2_DTC_METHOD='rk2', S2_DTC_SUBSTEPS='1', S2_DTC_EVAL_SUBSTEPS='2',
                   S2_Y_INIT_WITH='0', S2_Z_INIT_WITH='x', S2_DTC_ORDERING_PAIRS='2' if mode=='s2_dtc' else '0')
        result = subprocess.run(['bash', '-c', 'source launch_scripts/slurm_search_config.sh'],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        for item in ('ODE_BLOCK=S2DTCFFFB', 'S2_DTC_METHOD=rk2',
                     'S2_DTC_SUBSTEPS=1', 'S2_DTC_EVAL_SUBSTEPS=2',
                     'S2_Y_INIT_WITH=0', 'S2_Z_INIT_WITH=x'):
            assert item in result.stdout, result.stdout
        assert 'S2_DTC_ORDERING_PAIRS=' + env['S2_DTC_ORDERING_PAIRS'] in result.stdout


def test_real_ft_command_receives_its_own_substeps(tmp_path):
    for mode, default in (('s2_dtc', '1'), ('s2_dtc_pulse', '2')):
        for override in (None, '3'):
            settings = dict(TOGGLE_MODE=mode, S2_DTC_EVAL_SUBSTEPS='8',
                _invocation='finetune_one_combo_model example TIMMPCNetNoBatchNorm_test "${FT_ODE_BLOCK}"')
            if override:
                settings['S2_DTC_FT_SUBSTEPS'] = override
            args = resolve_eval(tmp_path, **settings)
            assert 'train_ode_cifar.py' in args
            assert args.count('--s2_dtc_substeps') == 1
            assert args[args.index('--s2_dtc_substeps')+1] == (override or default)
            assert args[args.index('--ode_block')+1] == ('S2DTCFFFB' if mode == 's2_dtc' else 'S2DTCPulseFFFB')


def test_ordering_pairs_forwarded_only_to_ft(tmp_path):
    settings = dict(TOGGLE_MODE='s2_dtc', S2_DTC_ORDERING_PAIRS='4')
    ft = resolve_eval(tmp_path, **settings,
        _invocation='finetune_one_combo_model example TIMMPCNetNoBatchNorm_test "${FT_ODE_BLOCK}"')
    assert ft[ft.index('--s2_dtc_ordering_pairs') + 1] == '4'
    evaluation = resolve_eval(tmp_path, **settings)
    assert '--s2_dtc_ordering_pairs' not in evaluation
    pretrain = resolve_eval(tmp_path, **settings,
        _invocation='python "${TOGGLE_ARGS[@]}"')
    assert '--s2_dtc_ordering_pairs' not in pretrain


@pytest.mark.parametrize('settings', [dict(TOGGLE_MODE='s2_dtc_pulse'),
    dict(S2_DTC_METHOD='euler'), dict(S2_DTC_FT_SUBSTEPS='2')])
def test_ordering_invalid_pipeline_fails_before_pretraining(tmp_path, settings):
    env = dict(TOGGLE_MODE='s2_dtc', S2_DTC_ORDERING_PAIRS='2')
    env.update(settings)
    with pytest.raises(AssertionError, match='ordering correction requires'):
        resolve_eval(tmp_path, **env)


def test_legacy_ft_receives_only_legacy_toggle_arguments(tmp_path):
    args = resolve_eval(tmp_path, TOGGLE_MODE='odexinit',
        _invocation='finetune_one_combo_model example TIMMPCNetNoBatchNorm_test "${FT_ODE_BLOCK}"')
    assert '--s2_dtc_substeps' not in args
    assert args[args.index('--ode_block')+1] == 'ToggleODEXInitFFFB'


def test_s2_dtc_mc45_worker_and_python_routing(tmp_path):
    import sys
    from unittest.mock import patch
    from scripts.run_toggle_nonideality_ablation import parse_args, build_command
    from ode_inference import parse_args as inference_args
    env = dict(os.environ, IS_SLURM='0', REPO_ROOT=str(ROOT), DTC_DYNAMICS='s2',
               OUTPUT_DIR=str(tmp_path / 'mc45'), S2_DTC_EVAL_METHOD='dopri5',
               S2_Y_INIT_WITH='0', S2_Z_INIT_WITH='x', S2_DTC_EVAL_SUBSTEPS='3',
               MODEL_NAME='TIMMQAT_C100_S2DTCFFFB_CiFAIR', MODEL_DIR=str(tmp_path))
    script = '''
python() { printf 'EVAL_ARG:%s\\n' "$@"; }
source launch_scripts/run_mc45_toggle_ablation.sh
'''
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = [line[len('EVAL_ARG:'):] for line in result.stdout.splitlines()
            if line.startswith('EVAL_ARG:')]
    with patch.object(sys, 'argv', argv):
        args = parse_args()
    command = build_command(args, 'all_known')
    with patch.object(sys, 'argv', command[2:]):
        resolved = inference_args()
    assert resolved.ode_block == 'S2DTCPulseFFFB'
    assert resolved.ode_wrapper == 'S2DTCQATTester2State'
    assert resolved.method == 'dopri5'
    assert resolved.y_init_with == '0' and resolved.z_init_with == 'x'
    assert resolved.s2_dtc_substeps == 3
    args.dtc_dynamics = 'toggle'
    import pytest
    with pytest.raises(ValueError, match='require --dtc_dynamics s2'):
        build_command(args, 'all_known')


def test_full_mc45_can_use_full_bank_after_singleton_ft(tmp_path):
    args = resolve_eval(tmp_path, POST_FT_EVAL_CORNER_IDS='all',
        POST_FT_EVAL_N_TRIALS='10',
        MC_COUPLER_NONLINEAR_VARIATION_SOURCE='coupler_monte_v2',
        EXPANDED_WEIGHT_DIR=str(tmp_path/'expanded'))
    assert '--corner_ids' not in args
    assert args[args.index('--n_trials')+1] == '10'
    assert args[args.index('--mc_coupler_nonlinear_variation_source')+1] == 'coupler_monte_v2'
    assert args[args.index('--expanded_w_dir')+1] == str(tmp_path/'expanded')
