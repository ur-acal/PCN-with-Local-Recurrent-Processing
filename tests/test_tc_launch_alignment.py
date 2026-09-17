"""Lock TC defaults at real shell/Python boundaries, without submitting jobs."""
import contextlib
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from baseline.train_baseline_cifar import parse_args as cnn_train
from baseline.evaluate_physical_feedforward_cifar import parse_args as cnn_eval
from train_ode_cifar import get_args as pcn_train
from ode_inference import parse_args as pcn_eval

ROOT = Path(__file__).resolve().parents[1]


def environment(**values):
    # Do not let the developer's exported experiment settings mask defaults.
    env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LD_LIBRARY_PATH')
           if key in os.environ}
    env.update(TASK='cifar100', IMG_TYPE='rgb', **values)
    return env


def parsed(parser, argv):
    with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
        return parser()


def cnn_stages(**overrides):
    command = '''
mkdir() { :; }
sbatch() {
  python() { :; }; export -f python
  bash launch_scripts/run_feedforward_cifar_pretrain.sh
  export MODEL_CKPT=/tmp/not-loaded.pth
  bash launch_scripts/run_feedforward_physical_ft.sh
  bash launch_scripts/run_feedforward_physical_eval.sh
}
source launch_scripts/slurm_search_feedforward_config.sh
'''
    env = environment(TC_FEEDFORWARD='true', REPO_ROOT=str(ROOT),
                      MODEL_NAME='wrn_28_2_cifar_nobn_no_bias_avgpool', **overrides)
    out = subprocess.check_output(['bash', '-c', command], cwd=ROOT, env=env, text=True)
    argv = [shlex.split(line[len('Running:'):])[1:] for line in out.splitlines()
            if line.startswith('Running:')]
    assert len(argv) == 3, out
    return [parsed(parser, args) for parser, args in
            zip((cnn_train, cnn_train, cnn_eval), argv)]


def pcn_stage(stage, state='1', **overrides):
    env = environment(TC_DRY_RUN='true', TC_STATE=state,
                      MODEL_NAME='TIMMPCNet_C100_ODEXInitFFFB', **overrides)
    out = subprocess.check_output(['bash', 'launch_scripts/run_tc_nonidealities.sh', stage],
                                  cwd=ROOT, env=env, text=True)
    return parsed(pcn_train if stage == 'ft' else pcn_eval, shlex.split(out)[2:])


class TCLaunchAlignmentTests(unittest.TestCase):
    def test_feedforward_slurm_envelope_uses_proven_cluster_pattern(self):
        submitter = (ROOT/'launch_scripts/slurm_search_feedforward_config.sh').read_text()
        worker = (ROOT/'launch_scripts/feedforward_pretrain_then_ft.sbatch').read_text()
        required = (
            '#!/bin/bash -l', '#SBATCH -p ising', '#SBATCH -N 1',
            '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=16',
            '#SBATCH --gres=gpu:1', '#SBATCH -t 90:10:00',
            '#SBATCH -o /scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/slurm_jobs/slurm_%j.out',
            'source activate base', 'conda activate scanbase')
        for line in required:
            self.assertIn(line, worker)
        self.assertIn(
            'REPO_ROOT="${REPO_ROOT:-/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing}"',
            submitter)
        self.assertIn('SBATCH_SCRIPT="${REPO_ROOT}/launch_scripts/feedforward_pretrain_then_ft.sbatch"',
                      submitter)
        executable = '\n'.join(
            line for text in (submitter, worker) for line in text.splitlines()
            if not line.lstrip().startswith('#'))
        for forbidden in ('bash -lc', 'conda shell.', 'module load', 'module swap'):
            self.assertNotIn(forbidden, executable)

    def test_rgb_teacher_uses_shared_cache_without_requiring_a_preload(self):
        submitter = (ROOT/'launch_scripts/slurm_run_rgb_teacher.sh').read_text()
        worker = (ROOT/'launch_scripts/run_rgb_teacher.sbatch').read_text()
        self.assertIn('TORCH_HOME=${torch_home}', submitter)
        self.assertIn('export TORCH_HOME=', worker)
        self.assertNotIn("if not cached.is_file():", worker)
        self.assertIn('torchvision will download it', worker)

    def test_pcn_architecture_list_overrides_preserve_defaults_and_schedules(self):
        base = environment(SHOW_COMB_ONLY='1', NUM_COMB_PER_NUM_LAYER='20')
        default = subprocess.check_output(
            ['bash', 'launch_scripts/slurm_search_config.sh'],
            cwd=ROOT, text=True, env=base)
        self.assertIn('CHAN_0=24 NUM_LAYERS=16', default)
        self.assertNotIn('NUM_LAYERS=22', default)

        configured = dict(base, PCN_CHAN_0_LIST='16',
                          PCN_NUM_LAYERS_LIST='22 28')
        output = subprocess.check_output(
            ['bash', 'launch_scripts/slurm_search_config.sh'],
            cwd=ROOT, text=True, env=configured)
        self.assertIn('CHAN_0=16 NUM_LAYERS=22', output)
        self.assertIn('N22_C16_n07_n16_n26', output)
        self.assertIn('CHAN_0=16 NUM_LAYERS=28', output)
        self.assertIn('N28_C16_n09_n18_n28', output)

        selected = dict(base, PCN_CHAN_0_LIST='16',
                        PCN_NUM_LAYERS_LIST='22',
                        NUM_COMB_PER_NUM_LAYER='3', COMB_SEL_SET='1 3')
        output = subprocess.check_output(
            ['bash', 'launch_scripts/slurm_search_config.sh'],
            cwd=ROOT, text=True, env=selected)
        self.assertIn('N22_C16_n06_n16_n27', output)
        self.assertNotIn('N22_C16_n06_n17_n26', output)
        self.assertIn('N22_C16_n07_n16_n26', output)

    def test_pcn_slurm_defaults_select_rgb_teacher(self):
        out = subprocess.check_output(['bash', 'launch_scripts/slurm_search_config.sh'],
            cwd=ROOT, text=True, env=environment(TC_NONIDEALITIES='true', TC_DRY_RUN='true'))
        commands = [shlex.split(line) for line in out.splitlines() if line.startswith('sbatch ')]
        self.assertTrue(commands)
        for command in commands:
            exports = next(x for x in command if x.startswith('--export='))
            self.assertIn('TEACHER_CKPT=./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill.pth', exports)
            self.assertIn('TEACHER_ARCH_SOURCE=torchvision', exports)
            self.assertIn('ENABLE_MEASURED_POOLING=true', exports)

    def test_all_hardware_defaults_after_slurm_export_and_stage_transition(self):
        pre, ft, evaluation = cnn_stages()
        self.assertTrue(pre.physical_pretraining)
        self.assertFalse(pre.enable_measured_activation)
        self.assertFalse(pre.enable_measured_pooling)
        common = dict(R=1e4, C=49e-15, v_dd=.1, one_over_q=1., w_bits=5,
                      enob=None, enable_spin_variation=True, sigma_spin=.1,
                      spin_variation_mean=1., spin_variation_seed=4096,
                      enable_coupler_noise=True, coupler_noise_p=.6e-12,
                      coupler_noise_seed=4096, enable_summing_current_noise=True,
                      summing_current_p=.6e-12, summing_noise_seed=4096,
                      tc_noise_reference_R=50e3, tc_curve_sampling='histogram',
                      enable_measured_activation=True, enable_measured_pooling=True,
                      activation_interpolation='piecewise_linear',
                      activation_fit_constraint='auto',
                      activation_normalize_positive_endpoint=False,
                      measured_pooling_nominal_R=1e4,
                      nonlinear_R_curve_seed=4096,
                      enable_slow_summing_current=False, enable_slow_coupler_noise=False)
        mean = './hardware_data/res_vs_vin_10k_150k.csv'
        common.update(nonlinear_R_table=mean, measured_pooling_curve_path=mean,
                      tc_covariance_table='./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
        for stage, cnn in (('ft', ft), ('eval', evaluation)):
            for state in ('1', '2'):
                pcn = pcn_stage(stage, state)
                for key, expected in common.items():
                    self.assertEqual(getattr(pcn, key), expected, ('PCN', state, stage, key))
                    self.assertEqual(getattr(cnn, key), expected, ('CNN', stage, key))
                self.assertEqual(cnn.activation_curve_path, pcn.activation_curve_path)
                self.assertEqual(cnn.activation_corner, pcn.activation_corner)
                self.assertIsNone(pcn.weight_quant_factor_bits)
                self.assertEqual(cnn.weight_quant_factor_bits, -1)
                self.assertTrue(pcn.nonlinear_R)
                self.assertFalse(cnn.enable_dtc_nonideality)
                self.assertFalse(cnn.one_shot_conv)
                self.assertEqual(cnn.tc_method, pcn.method)
                self.assertEqual(cnn.tc_tol, pcn.tol)
                if stage == 'ft':
                    self.assertEqual(cnn.nonlinear_R_train_mode, 'exact_curve')
                    self.assertEqual(pcn.tc_conv_method, 'shared')
                    self.assertEqual(cnn.noise_level, 0.)
                    self.assertEqual(pcn.noise_level, 0.)
                else:
                    self.assertTrue(cnn.enable_nonlinear_R)
                    self.assertTrue(cnn.use_expanded_weights and pcn.test_expanded)
                    self.assertEqual(cnn.nonlinear_R_curve_sharing, 'per_coupler')
                    self.assertEqual(cnn.activation_curve_sharing, 'per_spin')
                    self.assertEqual(pcn.activation_curve_sharing, 'per_spin')
                    self.assertEqual(cnn.n_trials, pcn.noisy_trials)
                    self.assertEqual(cnn.n_trials, 10)
                    self.assertFalse(cnn.diff_mismatch or pcn.diff_mismatch)

    def test_explicit_ablation_and_common_source_overrides(self):
        options = dict(ENABLE_MEASURED_POOLING='false', ENABLE_COUPLER_NOISE='false',
                       TC_MEAN_TABLE='./custom_means.csv', TC_COVARIANCE_TABLE='./custom_cov.csv')
        _, ft, evaluation = cnn_stages(**options)
        for stage, cnn in (('ft', ft), ('eval', evaluation)):
            pcn = pcn_stage(stage, **options)
            for key in ('enable_measured_pooling', 'enable_coupler_noise'):
                self.assertFalse(getattr(pcn, key) or getattr(cnn, key))
            for key in ('nonlinear_R_table', 'tc_covariance_table', 'measured_pooling_curve_path'):
                self.assertEqual(getattr(pcn, key), getattr(cnn, key))

    def test_rgb_teachers_and_learning_rates(self):
        pre, ft, _ = cnn_stages()
        expected = './checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill.pth'
        for args in (pre, ft, pcn_stage('ft')):
            self.assertEqual(args.teacher_ckpt, expected)
            self.assertEqual(args.teacher_arch, 'efficientnet_v2_l')
        self.assertIn('lr=0.1,', pre.override)
        self.assertIn('lr=0.005,', ft.override)
        for task in ('cifar10', 'cifar100'):
            script = 'source launch_scripts/rgb_teacher_defaults.sh; echo "$TEACHER_CKPT $TEACHER_ARCH"'
            env = environment(); env['TASK'] = task
            out = subprocess.check_output(['bash', '-c', script], cwd=ROOT, env=env, text=True)
            self.assertIn(f'efficientnet_v2_l_{task}_rgb_OldNoTimm_MatchDistill.pth', out)
        env = environment(); env.update(IMG_TYPE='CiFAIR', TEACHER_CKPT='keep.pth')
        self.assertEqual(subprocess.check_output(['bash', '-c', script], cwd=ROOT,
                                                env=env, text=True).split()[0], 'keep.pth')

    def test_python_failure_through_tee_and_no_downstream_stages(self):
        source = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        prefix, tail = source.split('# PHASE 1:', 1)
        prefix = prefix.replace('source activate base', 'true').replace('conda activate scanbase', 'true')
        with tempfile.TemporaryDirectory() as directory:
            prefix = '\n'.join('LOGDIR='+shlex.quote(directory) if line.startswith('LOGDIR=')
                               else line for line in prefix.splitlines())
            env = environment(TC_NONIDEALITIES='true', COMB_LIST='audit', OUTPUT_SAVE_PATH=directory)
            failure = prefix + '''
python() { return 42; }
finetune_one_combo_model audit model ODEXInitFFFB
exit $?
'''
            result = subprocess.run(['bash', '-c', failure], cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(result.returncode, 42)
            for failed in ('pretrain', 'ft'):
                mocks = '''
extract_model_name() { echo model; }
eval_one_combo_model() { echo UNEXPECTED_EVALUATION; }
'''
                mocks += 'run_one_combo() { return '+('42' if failed == 'pretrain' else '0')+'; }\n'
                mocks += 'finetune_one_combo_model() { echo START_FT; return 42; }\n'
                result = subprocess.run(['bash', '-c', prefix+'\n'+mocks+'\n# PHASE 1:'+tail],
                                        cwd=ROOT, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn('UNEXPECTED_EVALUATION', result.stdout)
                if failed == 'pretrain':
                    self.assertNotIn('START_FT', result.stdout)


if __name__ == '__main__':
    unittest.main()
