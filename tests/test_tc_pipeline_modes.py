"""TC mode discovery/routing tests: CPU checkpoints and mocked stage commands."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from pc_conv import PCConvReLU6
from launch_scripts.find_tc_pretrain import find_pretrain
from test_tc_pipeline import TCPipelineTests, ROOT


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'run_config.json').write_text(json.dumps(dict(input_quant_bits=None, center_student_input=False)))
        self.name = 'TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_1REP'
        self.request = dict(inp=[3,16], out=[16,32], pool=[1,0], task='cifar10', img_type='rgb', block='ODEXInitFFFB')
        self.checkpoint = dict(epoch=300, acc=.96, net_type='PCNetNoBatchNorm',
            init_args=dict(model_args=dict(inp_channels=[3,16], out_channels=[16,32], max_pool=[1,0],
                num_classes=10, stride=[1,1], kernel_size=[3,3], avg_pooling=True, first_bn=False,
                pc_conv_layer=PCConvReLU6), kwargs=dict(bias=False, tie_weights=False, tie_bp=False, bypass=False)),
            net={'linear.weight':torch.zeros(10,32), 'PcConvs.0.FFconv.weight':torch.zeros(16,3,3,3),
                 'PcConvs.1.FFconv.weight':torch.zeros(32,16,3,3)})
        self.path = self.write(self.name)

    def write(self, name, checkpoint=None):
        folder=self.root/name; folder.mkdir(exist_ok=True)
        path=folder/f'{name}_last_ckpt.pth'
        torch.save(checkpoint or self.checkpoint,path)
        return path

    def test_match_and_rejections(self):
        self.assertEqual(find_pretrain(self.root, **self.request), self.path)
        for change in (dict(task='cifar100'),dict(img_type='CiFAIR'),dict(block='S2NoisyIYAsXZAs0'),
                       dict(pool=[0,1]),dict(out=[16,64]),dict(bits=12),dict(center=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                find_pretrain(self.root, **dict(self.request, **change))
        other=self.root/'wrong_experiment';other.mkdir()
        (other/'run_config.json').write_text((self.root/'run_config.json').read_text())
        with self.assertRaises(ValueError): find_pretrain(other, **self.request)
        d=copy.deepcopy(self.checkpoint);d['epoch']=299;self.write(self.name,d)
        with self.assertRaises(ValueError): find_pretrain(self.root, **self.request)
        self.write(self.name)
        self.write(self.name.replace('1REP','2REP'))
        with self.assertRaisesRegex(ValueError,'found 2'): find_pretrain(self.root, **self.request)

    def test_toggle_pretrain_discovery(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['init_args']['model_args']['inp_channels'] = [4, 16]
        checkpoint['net']['PcConvs.0.FFconv.weight'] = torch.zeros(16, 4, 3, 3)
        name = self.name.replace('_ODEXInitFFFB_', '_ToggleODEXInitFFFB_').replace(
            '_1REP', '_CiFAIR_1REP')
        path = self.write(name, checkpoint)
        request = dict(self.request, inp=[4, 16], img_type='CiFAIR',
                       block='ToggleODEXInitFFFB')
        self.assertEqual(find_pretrain(self.root, **request), path)

    def test_toggle_ft_and_eval_routes_from_verified_checkpoint(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['init_args']['model_args']['inp_channels'] = [4, 16]
        checkpoint['net']['PcConvs.0.FFconv.weight'] = torch.zeros(16, 4, 3, 3)
        name = self.name.replace('_ODEXInitFFFB_', '_ToggleODEXInitFFFB_').replace(
            '_1REP', '_CiFAIR_1REP')
        last = self.write(name, checkpoint)
        torch.save(checkpoint, last.with_name(f'{name}_best_ckpt.pth'))

        worker = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        guard = worker[worker.index('mode="${mode:-default}"'):worker.index('# ---- Conda activation')]
        guard_result = subprocess.run(['bash', '-c', guard], cwd=ROOT,
            env=dict(os.environ, mode='ft_and_eval', TOGGLE_MODE='odexinit',
                     SWITCH_INF='false', TC_NONIDEALITIES='false'),
            text=True, capture_output=True)
        self.assertEqual(guard_result.returncode, 0, guard_result.stderr)

        phases = worker[worker.index('echo "==== PHASE 1:'):worker.index('# Merge summaries')]
        preamble = '''
set -o pipefail
COMBS=($'combo\\t0\\t4 16\\t16 32\\t1 0')
STRIDE=(1); KERNEL=(3)
run_one_combo() { echo CALL:pretrain; return 99; }
extract_model_name() { sed -n 's/.*Model Name: \\(.*\\) -----.*/\\1/p' "$1"; }
finetune_one_combo_model() { echo "CALL:ft:$2"; mkdir -p "$LOGDIR/$1"; echo 'Train finished, Model Name: fixture_ft -----' > "$LOGDIR/$1/finetune_${FT_ODE_BLOCK}.log"; }
eval_one_combo_model() { echo "CALL:eval:$2"; }
'''
        env = dict(os.environ, PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            mode='ft_and_eval', OUTPUT_SAVE_PATH=str(self.root), PRETRAIN_SAVE_PATH=str(self.root),
            FT_OUTPUT_SAVE_PATH=str(self.root), LOGDIR=str(self.root/'toggle_logs'),
            TRAIN_ODE_BLOCK='ToggleODEXInitFFFB', FT_ODE_BLOCK='ToggleODEXInitFFFB',
            INF_ODE_BLOCK='TogglePulseODEXInitFFFB', FINAL_EVAL_ONLY='false',
            TASK='cifar10', IMG_TYPE='CiFAIR', PCN='PCNetNoBatchNorm',
            INPUT_QUANT_BITS='none', CENTER_STUDENT_INPUT='false')
        result = subprocess.run(['bash', '-c', preamble+phases], cwd=ROOT,
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        self.assertNotIn('CALL:pretrain', result.stdout)
        self.assertIn('CALL:ft:'+name, result.stdout)
        self.assertIn('CALL:eval:fixture_ft', result.stdout)

    def test_real_pcn_phase_routing(self):
        worker=(ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        phases=worker[worker.index('echo "==== PHASE 1:'):worker.index('# Merge summaries')]
        preamble='''
set -o pipefail
COMBS=($'combo\\t0\\t3 16\\t16 32\\t1 0')
STRIDE=(1); KERNEL=(3)
run_one_combo() { mkdir -p "$LOGDIR/combo"; echo "Train finished, Model Name: $FIXTURE -----" > "$LOGDIR/combo/train_${TRAIN_ODE_BLOCK}.log"; echo CALL:pretrain; }
extract_model_name() { sed -n 's/.*Model Name: \\(.*\\) -----.*/\\1/p' "$1"; }
finetune_one_combo_model() { echo "CALL:ft:$2"; [[ "${FAIL_FT:-false}" != true ]] || return 9; mkdir -p "$LOGDIR/$1"; echo 'Train finished, Model Name: fixture_ft -----' > "$LOGDIR/$1/finetune_${FT_ODE_BLOCK}.log"; }
eval_one_combo_model() { echo "CALL:eval:$2"; }
'''
        env=dict(os.environ, PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            OUTPUT_SAVE_PATH=str(self.root), PRETRAIN_SAVE_PATH=str(self.root),
            FT_OUTPUT_SAVE_PATH=str(self.root), LOGDIR=str(self.root/'logs'), FIXTURE=self.name,
            EXP='test',SLURM_JOB_ID='fixture',TRAIN_ODE_BLOCK='ODEXInitFFFB',FT_ODE_BLOCK='ODEXInitFFFB',
            INF_ODE_BLOCK='ODEXInitFFFB',FINAL_EVAL_ONLY='true',TASK='cifar10',IMG_TYPE='rgb',
            PCN='PCNetNoBatchNorm',INPUT_QUANT_BITS='none',CENTER_STUDENT_INPUT='false')
        expected={'default':['pretrain','ft','eval'],'pretrain_only':['pretrain'],
                  'ft_only':['ft'],'ft_and_eval':['ft','eval']}
        for mode, stages in expected.items():
            result=subprocess.run(['bash','-c',preamble+phases],cwd=ROOT,env=dict(env,mode=mode),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            calls=[line.split(':')[1] for line in result.stdout.splitlines() if line.startswith('CALL:')]
            self.assertEqual(calls,stages)
            if 'ft' in stages: self.assertIn('CALL:ft:'+self.name,result.stdout)
        result=subprocess.run(['bash','-c',preamble+phases],cwd=ROOT,
            env=dict(env,mode='ft_and_eval',FAIL_FT='true'),text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('CALL:eval',result.stdout)


class QATContinuationTests(unittest.TestCase):
    """Run the real continuation/worker shells, mocking only stage commands."""

    def run_continuation(self, directory, **overrides):
        directory = Path(directory)
        name = 'TIMMQAT5b_PCNetNoBatchNorm_S2DTCFFFB_C100_CiFAIR_1REP'
        source = directory/'source'/name
        source.mkdir(parents=True)
        # Shell-routing fixture, not a numerical checkpoint.
        for suffix in ('full_param_last', 'latest'):
            (source/f'{name}_{suffix}_ckpt.pth').write_bytes(b'fixture')
        bindir = directory/'bin'
        bindir.mkdir()
        (bindir/'activate').write_text('[[ "$1" == base ]] || return 44\n')
        env = dict(PATH=str(bindir)+':'+os.environ['PATH'], HOME=os.environ['HOME'],
            IS_SLURM='1', SLURM_JOB_ID='fixture', SLURM_CPUS_PER_TASK='16',
            TRACE=str(directory), TOGGLE_MODE='s2_dtc_pulse', TC_NONIDEALITIES='false',
            SWITCH_INF='false', VALIDATION_MODE='false', TASK='cifar100', IMG_TYPE='CiFAIR',
            PRETRAIN_SAVE_PATH=str(source.parent), CONTINUATION_SOURCE_MODEL=name,
            FT_OUTPUT_SAVE_PATH=str(directory/'output'), LOGDIR=str(directory/'logs'),
            COMB_LIST='combo\t0\t4 24\t24 24\t0 0', FINAL_EVAL_ONLY='true',
            FT_NUM_EPOCHS='20', FT_LEARNING_RATE='0.0001', WARMUP_FT='0',
            NONLINEAR_R_TABLE='coupler_low_volt_cascode', R_VAL='67e3', C_VAL='500e-15',
            V_DD='0.6', TOGGLE_ONE_OVER_Q='6', TOGGLE_TIMING_MODE='fixed',
            TOGGLE_Y_TIME='10e-9', Z_OVER_Y_TIME='1', SCALE_TRAIN_RECIPE='1',
            MC_RELU_MONTE_CARLO_SOURCE='1022_RELU_Voltage', MEASURED_ACTIVATION_SCOPE='all',
            ACTIVATION_CORNER_MODE='random_per_forward', ACTIVATION_RANDOM_CURVE_SHARING='per_spin',
            FT_CORNER_COUPLED_SAMPLING='true', ENABLE_MEASURED_POOLING='true',
            FINAL_HEAD_TYPE='digital', FINAL_ADC_NOISE_LSB='0.5', LINEAR_BIAS='false',
            S2_DTC_METHOD='rk2', S2_DTC_SUBSTEPS='1', S2_DTC_FT_SUBSTEPS='1',
            S2_DTC_ORDERING_PAIRS='0', S2_DTC_EVAL_METHOD='rk2', S2_DTC_EVAL_SUBSTEPS='8',
            CHECKPOINT_ODE_RHS_TRAINING='true', CHECKPOINT_ODE_RHS_PORTION='1.0',
            POST_FT_EVAL_CKPT='last', POST_FT_EVAL_CORNER_IDS='FS_V0_T0 FS_V2_T1',
            POST_FT_EVAL_N_TRIALS='5')
        env.update(overrides)
        shell = r'''
conda() { [[ "$1" == activate && "$2" == scanbase ]]; }
python() {
  [[ "$1" != -u ]] || shift
  case "$1" in
    train_ode_cifar.py)
      printf '%s\0' "$@" > "$TRACE/ft.argv"
      [[ "${FAIL_FT:-false}" != true ]] || return 42
      echo '----- Train finished, Model Name: fixture_ft -----'
      ;;
    scripts/run_toggle_nonideality_ablation.py)
      printf '%s\0' "$@" > "$TRACE/eval.argv"
      ;;
    shell_utils/pipeline_node_run_parse.py) ;;
    *) echo "Unexpected Python stage: $1" >&2; return 43 ;;
  esac
}
export -f conda python
bash ./launch_scripts/run_cont_qat.sbatch
'''
        result = subprocess.run(['bash', '-c', shell], cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=30)
        def captured(stage):
            path = directory/f'{stage}.argv'
            return path.read_bytes().decode().split('\0')[:-1] if path.exists() else []
        return result, captured('ft'), captured('eval'), env

    def test_20_and_40_epoch_commands_and_eval_use_existing_worker(self):
        from unittest.mock import patch
        from train_ode_cifar import get_args
        from scripts.run_toggle_nonideality_ablation import parse_args
        for epochs in ('20', '40'):
            with self.subTest(epochs=epochs), tempfile.TemporaryDirectory() as td:
                result, ft, evaluation, env = self.run_continuation(td, FT_NUM_EPOCHS=epochs)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                with patch.dict(os.environ, env), patch.object(sys, 'argv', ft):
                    args = get_args()
                self.assertEqual(args.model_name, env['CONTINUATION_SOURCE_MODEL'])
                self.assertEqual(args.ckpt, 'full_param_last')
                self.assertEqual(args.num_epochs, int(epochs))
                self.assertEqual(args.learning_rate, .0001)
                self.assertEqual(args.warmup_epoch, 0)
                self.assertEqual(args.ode_block, 'S2DTCPulseFFFB')
                self.assertEqual(args.method, 'rk2')
                self.assertEqual(args.n_steps, 5)
                self.assertEqual(args.s2_dtc_substeps, 1)
                self.assertEqual(args.s2_dtc_ordering_pairs, 0)
                self.assertTrue(args.checkpoint_ode_rhs_training)
                self.assertEqual(args.checkpoint_ode_rhs_portion, 1.)
                self.assertEqual(args.final_head_type, 'digital')
                self.assertFalse(args.linear_bias)
                self.assertEqual(args.measured_activation_scope, 'all')
                self.assertEqual(args.activation_random_curve_sharing, 'per_spin')
                self.assertTrue(args.ft_corner_coupled_sampling)
                with patch.object(sys, 'argv', evaluation):
                    args = parse_args()
                self.assertEqual(args.ckpt, 'last')
                self.assertEqual(args.dtc_dynamics, 's2')
                self.assertEqual(args.s2_dtc_method, 'rk2')
                self.assertEqual(args.s2_dtc_substeps, 8)
                self.assertEqual(args.corner_ids, ['FS_V0_T0', 'FS_V2_T1'])
                self.assertEqual(args.n_trials, 5)

    def test_latest_and_ft_only(self):
        with tempfile.TemporaryDirectory() as td:
            result, ft, evaluation, _ = self.run_continuation(
                td, mode='ft_only', CONTINUATION_SOURCE_SELECTION='latest')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(ft[ft.index('--ckpt')+1], 'latest')
            self.assertFalse(evaluation)

    def test_rgb_n4_continuation_batch_accumulation_and_three_corner_trials(self):
        from unittest.mock import patch
        from train_ode_cifar import get_args
        from scripts.run_toggle_nonideality_ablation import parse_args, build_command
        with tempfile.TemporaryDirectory() as td:
            result, ft, evaluation, env = self.run_continuation(
                td, IMG_TYPE='rgb', NORMALIZE_STUDENT_INPUT='false',
                S2_DTC_FT_SUBSTEPS='4', FT_BATCH_SIZE='16',
                FT_GRAD_ACCUM_STEPS='8', POST_FT_EVAL_BATCH_SIZE='16',
                POST_FT_EVAL_N_TRIALS='3')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with patch.dict(os.environ, env), patch.object(sys, 'argv', ft):
                args = get_args()
            self.assertEqual(args.s2_dtc_substeps, 4)
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.grad_accum_steps, 8)
            self.assertEqual(args.img_type, 'rgb')
            self.assertFalse(args.normalize_student_input)
            self.assertEqual(args.ckpt, 'full_param_last')
            with patch.object(sys, 'argv', evaluation):
                args = parse_args()
            self.assertEqual(args.n_trials, 3)
            self.assertEqual(args.test_bs, 16)
            self.assertEqual(args.s2_dtc_substeps, 8)
            self.assertEqual(args.ckpt, 'last')
            command = build_command(args, 'all_known')
            self.assertEqual(command[command.index('--img_type')+1], 'rgb')

    def test_invalid_source_or_mode_stops_before_python(self):
        for change in (dict(mode='default'), dict(CONTINUATION_SOURCE_SELECTION='last'),
                       dict(CONTINUATION_SOURCE_MODEL='missing_QAT'),
                       dict(CONTINUATION_SOURCE_MODEL='../QAT'),
                       dict(FT_OUTPUT_SAVE_PATH=''), dict(COMB_LIST='one\ntwo')):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                result, ft, evaluation, _ = self.run_continuation(td, **change)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(ft)
                self.assertFalse(evaluation)

    def test_preserve_source_and_propagate_ft_failure(self):
        with tempfile.TemporaryDirectory() as td:
            result, ft, evaluation, _ = self.run_continuation(
                td, FT_OUTPUT_SAVE_PATH=str(Path(td)/'source'))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(ft)
            self.assertFalse(evaluation)
        with tempfile.TemporaryDirectory() as td:
            result, ft, evaluation, _ = self.run_continuation(td, FAIL_FT='true')
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(ft)
            self.assertFalse(evaluation)

    def test_slurm_resources_match_existing_worker(self):
        def directives(name):
            return [line for line in (ROOT/'launch_scripts'/name).read_text().splitlines()
                    if line.startswith('#SBATCH')]
        # Ignore descriptive comments, not directive values.
        original = [line.split(' #')[0].rstrip() for line in directives('run_kdcrd_then_ft.sbatch')]
        self.assertEqual(directives('run_cont_qat.sbatch'), original)


class CNNModeTests(unittest.TestCase):
    def test_modes_keep_identical_stage_arguments(self):
        harness=TCPipelineTests()
        with tempfile.TemporaryDirectory() as td:
            result, baseline=harness.run_pipeline(td)
            self.assertEqual(result.returncode,0,result.stderr)
            # Reuse fake Python/checkpoint fixtures, bypass only sbatch/conda.
            env=dict(os.environ, PATH=str(Path(td)/'bin')+':'+os.environ['PATH'],TC_FEEDFORWARD='true',
                REPO_ROOT=str(ROOT),MODEL_NAME='wrn_28_2_cifar_nobn_no_bias_avgpool',TASK='cifar100',IMG_TYPE='rgb',
                PRETRAIN_OUTPUT_DIR=td+'/pre',FT_OUTPUT_DIR=td+'/ft',RESULT_PATH=td+'/results',PIPELINE_TRACE=td+'/trace.jsonl',
                MODEL_CKPT=td+'/ambient_checkpoint_must_not_override_TC_discovery.pth')
            # Same submission defaults as the original captured run.
            shell='''conda() { :; }; export -f conda
sbatch() { bash "${@: -1}"; }
source launch_scripts/slurm_search_feedforward_config.sh
'''
            for mode, indices in [('pretrain_only',[0]),('ft_only',[1]),('ft_and_eval',[1,2])]:
                Path(env['PIPELINE_TRACE']).write_text('')
                result=subprocess.run(['bash','-c',shell],cwd=ROOT,env=dict(env,mode=mode),text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                rows=[json.loads(line) for line in Path(env['PIPELINE_TRACE']).read_text().splitlines()]
                self.assertEqual([r['argv'] for r in rows],[baseline[i]['argv'] for i in indices])


if __name__=='__main__': unittest.main()
