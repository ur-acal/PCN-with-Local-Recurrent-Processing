"""Exercise the real PCN evaluation function without launching inference."""
import os
from pathlib import Path
import subprocess

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
    env.update(overrides)
    script = f'''
source <(head -n {stop} "{launcher}")
trap - TERM INT
python() {{ printf 'EVAL_ARG:%s\\n' "$@"; }}
eval_one_combo_model example TIMMQAT5bNoneaNT0p0mulTIMMPCNetNoBatchNorm_test post_ft "${{INF_ODE_BLOCK}}"
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


def test_full_mc45_can_use_full_bank_after_singleton_ft(tmp_path):
    args = resolve_eval(tmp_path, POST_FT_EVAL_CORNER_IDS='all',
        POST_FT_EVAL_N_TRIALS='10',
        MC_COUPLER_NONLINEAR_VARIATION_SOURCE='coupler_monte_v2',
        EXPANDED_WEIGHT_DIR=str(tmp_path/'expanded'))
    assert '--corner_ids' not in args
    assert args[args.index('--n_trials')+1] == '10'
    assert args[args.index('--mc_coupler_nonlinear_variation_source')+1] == 'coupler_monte_v2'
    assert args[args.index('--expanded_w_dir')+1] == str(tmp_path/'expanded')
