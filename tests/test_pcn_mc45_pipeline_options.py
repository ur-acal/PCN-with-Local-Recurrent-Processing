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


@pytest.mark.parametrize('mode', ['odexinit', 's2_dtc', 's2_dtc_pulse'])
def test_rgb_pipeline_evaluation_uses_explicit_input_and_batch(tmp_path, mode):
    import sys
    from unittest.mock import patch
    from scripts.run_toggle_nonideality_ablation import parse_args, build_command
    from ode_inference import parse_args as inference_args
    argv = resolve_eval(tmp_path, TOGGLE_MODE=mode, IMG_TYPE='rgb',
                        POST_FT_EVAL_BATCH_SIZE='16', POST_FT_EVAL_N_TRIALS='3')
    with patch.object(sys, 'argv', argv[argv.index('scripts/run_toggle_nonideality_ablation.py'):]):
        args = parse_args()
    assert args.img_type == 'rgb' and args.task == 'cifar100'
    assert args.test_bs == 16 and args.n_trials == 3
    command = build_command(args, 'all_known')
    with patch.object(sys, 'argv', command[2:]):
        parsed = inference_args()
    assert parsed.img_type == 'rgb' and parsed.task == 'cifar100'
    assert parsed.test_bs == 16 and parsed.test_expanded
    assert parsed.ode_block == ('TogglePulseODEXInitFFFB' if mode == 'odexinit'
                                else 'S2DTCPulseFFFB')


def test_standalone_corner_input_fallbacks_remain_legacy():
    import sys
    from unittest.mock import patch
    from scripts.run_toggle_nonideality_ablation import parse_args, build_command
    for model, task, img in [('TIMMQAT_C100_CiFAIR', 'cifar100', 'CiFAIR'),
                             ('TIMMQAT_scanGFI', 'cifar10', 'scanGFI')]:
        with patch.object(sys, 'argv', ['ablation', '--model_name', model]):
            args = parse_args()
        command = build_command(args, 'all_known')
        assert command[command.index('--task')+1] == task
        assert command[command.index('--img_type')+1] == img


@pytest.mark.parametrize('skip', ['true', 'false'])
def test_pretrain_and_ft_can_skip_only_post_ft_evaluation(tmp_path, skip):
    env = dict(os.environ, IS_SLURM='0', TOGGLE_MODE='s2_dtc', TASK='cifar100',
               IMG_TYPE='rgb', mode='default', SKIP_POST_FT_EVAL=skip,
               LOGDIR=str(tmp_path/'logs'), OUTPUT_SAVE_PATH=str(tmp_path/'ckpt'),
               COMB_LIST='example\t1\t3 24\t24 24\t0 0', TRACE=str(tmp_path/'stages'))
    env.pop('CONTINUATION_SOURCE_MODEL', None)
    env.pop('CONTINUATION_SOURCE_SELECTION', None)
    shell = r'''
python() {
  [[ "$1" != -u ]] || shift
  case "$1" in
    train_ode_cifar.py)
      if [[ " $* " == *' --model_name '* ]]; then
        echo ft >> "$TRACE"
        echo '----- Train finished, Model Name: fixture_ft -----'
      else
        echo pt >> "$TRACE"
        echo '----- Train finished, Model Name: fixture_pt -----'
      fi ;;
    scripts/run_toggle_nonideality_ablation.py) echo eval >> "$TRACE" ;;
    shell_utils/pipeline_node_run_parse.py) ;;
    *) return 43 ;;
  esac
}
export -f python
bash launch_scripts/run_kdcrd_then_ft.sbatch
'''
    result = subprocess.run(['bash', '-c', shell], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path/'stages').read_text().splitlines() == (
        ['pt', 'ft'] if skip == 'true' else ['pt', 'ft', 'eval'])


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


@pytest.mark.parametrize('mode', ['odexinit', 's2_dtc', 's2_dtc_pulse', 'none'])
def test_ft_microbatch_and_accumulation_are_independent_of_dynamics(tmp_path, mode):
    settings = dict(TOGGLE_MODE=mode, GRAD_ACCUM_STEPS='1',
        _invocation='finetune_one_combo_model example TIMMPCNetNoBatchNorm_test "${FT_ODE_BLOCK}"')
    default = resolve_eval(tmp_path, **settings)
    assert default[default.index('--batch_size') + 1] == '128'
    assert default[default.index('--grad_accum_steps') + 1] == '1'
    args = resolve_eval(tmp_path, **settings, FT_BATCH_SIZE='64', FT_GRAD_ACCUM_STEPS='2')
    assert args[args.index('--batch_size') + 1] == '64'
    assert args[args.index('--grad_accum_steps') + 1] == '2'
    # Launch arguments are accepted by the actual training parser.
    from train_ode_cifar import get_args
    from unittest.mock import patch
    import sys
    with patch.object(sys, 'argv', args[args.index('train_ode_cifar.py'):]):
        parsed = get_args()
    assert parsed.batch_size == 64 and parsed.grad_accum_steps == 2


def test_pretraining_has_independent_microbatch_and_accumulation(tmp_path):
    args = resolve_eval(tmp_path, TOGGLE_MODE='s2_dtc',
        PRETRAIN_BATCH_SIZE='32', PRETRAIN_GRAD_ACCUM_STEPS='4',
        FT_BATCH_SIZE='64', FT_GRAD_ACCUM_STEPS='2',
        _invocation='run_one_combo "${COMBS[0]}" "${TRAIN_ODE_BLOCK}"')
    assert args[args.index('--batch_size') + 1] == '32'
    assert args[args.index('--grad_accum_steps') + 1] == '4'


def test_s2_rgb_pretraining_matches_tc_rgb_augmentation_selection(tmp_path):
    import sys
    from unittest.mock import patch
    from train_ode_cifar import get_args
    parsed = []
    for mode, tc in [('s2_dtc', 'false'), ('none', 'true')]:
        argv = resolve_eval(tmp_path, TOGGLE_MODE=mode, TC_NONIDEALITIES=tc,
                            IMG_TYPE='rgb', TIMM_RE_PROB='0.0',
                            _invocation='run_one_combo "${COMBS[0]}" "${TRAIN_ODE_BLOCK}"')
        with patch.object(sys, 'argv', argv[argv.index('train_ode_cifar.py'):]):
            parsed.append(get_args())
    assert all(args.img_type == 'rgb' and args.timm_trainer for args in parsed)
    assert [args.timm_aug_level for args in parsed] == ['none', 'none']
    assert [args.timm_re_prob for args in parsed] == [0.0, 0.0]


@pytest.mark.parametrize('steps,timm', [('0', 'true'), ('2', 'false')])
def test_training_parser_rejects_invalid_or_unsupported_accumulation(steps, timm):
    from train_ode_cifar import get_args
    from unittest.mock import patch
    import sys
    with patch.object(sys, 'argv', ['train_ode_cifar.py', '--grad_accum_steps', steps,
                                  '--timm_trainer', timm]):
        with pytest.raises(SystemExit):
            get_args()


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
