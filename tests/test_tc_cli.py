"""Resolve real launch commands without loading checkpoints or submitting jobs."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from train_ode_cifar import get_args
from ode_inference import parse_args
from tc_cli import wrapper_options, reset_after_probe, record_trial
from test_tc_dense_training import make_block, wrap

ROOT = Path(__file__).resolve().parents[1]


def resolved(stage, state='1', **extra):
    env = dict(os.environ, TC_NONIDEALITIES='true', TC_DRY_RUN='true',
               TC_STATE=state, TOGGLE_MODE='none', SWITCH_INF='false',
               MODEL_NAME='inspection_only', MODEL_DIR='./inspection_only',
               TEACHER_CKPT='./inspection_only.pth', **extra)
    script = 'run_ode_mixed_ft.sh' if stage == 'ft' else 'run_ode_wrapped_inference.sh'
    output = subprocess.check_output(['bash', str(ROOT/'launch_scripts'/script)],
                                     env=env, text=True, cwd=ROOT)
    argv = shlex.split(output)[2:]
    with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
        return (get_args if stage == 'ft' else parse_args)()


class TCCommandTests(unittest.TestCase):
    def test_local_distillation_forwarding(self):
        for method in ('srrl', 'kd'):
            defaults = resolved('ft', DISTILL_METHOD=method,
                                DISTILL_ALPHA='', DISTILL_TEMPERATURE='')
            self.assertEqual(defaults.distill_method, method)
            self.assertEqual(defaults.distill_alpha, 0.3)
            self.assertEqual(defaults.distill_temperature, 2.0)
            custom = resolved('ft', DISTILL_METHOD=method,
                              DISTILL_ALPHA='0.6', DISTILL_TEMPERATURE='4.0')
            self.assertEqual(custom.distill_alpha, 0.6)
            self.assertEqual(custom.distill_temperature, 4.0)

    def test_model_defaults_and_checkpoint_wrapper(self):
        def command(stage, name, **overrides):
            env = dict(os.environ)
            for key in ('TASK', 'IMG_TYPE', 'CKPT', 'TEACHER_CKPT', 'TEACHER_ARCH',
                        'TC_STATE', 'TC_CONV_METHOD'):
                env.pop(key, None)
            env.update(TC_DRY_RUN='true', MODEL_NAME=name,
                       TOGGLE_MODE='none', SWITCH_INF='false', **overrides)
            out = subprocess.check_output(
                ['bash', 'launch_scripts/run_tc_nonidealities.sh', stage],
                cwd=ROOT, env=env, text=True)
            tokens = shlex.split(out)[3:]
            return dict(zip(tokens[::2], tokens[1::2]))

        ft = command('ft', 'TIMMQAT5b_C100_16Layers_CiFAIR_1REP')
        self.assertEqual(ft['--dataset'], 'cifar100')
        self.assertEqual(ft['--img_type'], 'CiFAIR')
        self.assertEqual(ft['--ckpt'], 'full_param_best')
        self.assertEqual(ft['--tc_conv_method'], 'shared')
        self.assertTrue(ft['--teacher_ckpt'].endswith(
            'efficientnet_v2_l_cifar100_CiFAIR_OldNoTimm_MatchDistill.pth'))
        for suffix, task, teacher in (
                ('C100_scanGFI', 'cifar100', 'b4_100.pth'),
                ('scanGFI', 'cifar10', 'b4.pth')):
            args = command('ft', 'TIMMPCNet_' + suffix + '_1REP')
            self.assertEqual(args['--dataset'], task)
            self.assertEqual(args['--img_type'], 'scanGFI')
            self.assertTrue(args['--teacher_ckpt'].endswith(teacher))
            self.assertEqual(args['--ckpt'], 'best')
        rgb = command('eval', 'TIMMQAT5b_C100_16Layers_1REP')
        self.assertEqual(rgb['--img_type'], 'rgb')
        self.assertEqual(rgb['--ode_wrapper'], 'QATTester1State')
        full = command('eval', 'TIMMQAT5b_C100_16Layers_1REP', CKPT='full_param_best')
        self.assertEqual(full['--ode_wrapper'], 'ODEWrapper1State')
        explicit = command('ft', 'TIMMQAT5b_C100_CiFAIR_1REP',
                           TASK='cifar10', IMG_TYPE='rgb', CKPT='best',
                           TEACHER_CKPT='./rgb.pth', TEACHER_ARCH='efficientnet-b4')
        self.assertEqual(explicit['--img_type'], 'rgb')
        self.assertEqual(explicit['--dataset'], 'cifar10')
        self.assertEqual(explicit['--ckpt'], 'best')
        self.assertEqual(explicit['--teacher_arch'], 'efficientnet-b4')

    def test_ft_and_eval_tolerance(self):
        for stage in ('ft', 'eval'):
            self.assertEqual(resolved(stage, TOL='').tol, 1e-6)
            self.assertEqual(resolved(stage, TOL='2e-6').tol, 2e-6)

    def test_shared_sampling_launcher_forwarding(self):
        for sampling in ('uniform','histogram'):
            for stage in ('ft','eval'):
                args=resolved(stage,TC_CONV_METHOD='shared',TC_CURVE_SAMPLING=sampling)
                self.assertEqual(args.tc_conv_method,'shared')
                self.assertEqual(wrapper_options(args)['tc_curve_sampling'],sampling)
                if stage=='eval':
                    self.assertTrue(args.test_expanded)
                    self.assertEqual(args.nonlinear_R_curve_sharing,'per_coupler')

    def test_pooling_uses_selected_distribution(self):
        from types import SimpleNamespace
        from tc_cli import pooling_options
        from measured_pooling import MeasuredAvgPool2d
        package = SimpleNamespace(v_grid=torch.tensor([-.1, .1]),
            means=torch.tensor([[10000., 11000.]]),
            factor=torch.eye(2)*100.)
        wrappers = [SimpleNamespace(ode_block=SimpleNamespace(FFconv=nn.Conv2d(1,1,1)))]
        args = SimpleNamespace(measured_pooling_curve_path='pool.csv',
                               measured_pooling_nominal_R=None, tc_covariance_table='cov.csv')
        with patch('tc_nonidealities.prepare_tc_resistance_curves', return_value=package) as prepare:
            opts = pooling_options(args, wrappers)
        self.assertEqual(prepare.call_args.args, ('pool.csv', 'cov.csv'))
        self.assertEqual(prepare.call_args.kwargs['R'], 10000.)
        self.assertEqual(opts['nominal_R'],10000.)
        torch.testing.assert_close(opts['curve_gaussian']['mean'],package.means[-1])
        pool = MeasuredAvgPool2d(kernel_size=2, seed=123, **opts)
        x = torch.full((2, 3, 4, 4), .01)
        pool.eval()
        torch.testing.assert_close(pool(x),pool(x),rtol=0,atol=0)
        pool.train()
        self.assertFalse(torch.equal(pool(x),pool(x)))

    def test_validation_resets_tc_without_rewinding_rng(self):
        from types import SimpleNamespace
        from trainer import TrainerCiFar
        from test_tc_dense_training import synthetic_package
        block = make_block()
        wrapper = wrap(block, enable_spin_variation=True, spin_variation_seed=31)
        block._tc_curve_package = synthetic_package(block)
        block._tc_curve_generator = torch.Generator().manual_seed(42)
        model = nn.Module(); model.PcConvs = nn.ModuleList([block]); model.eval()
        trainer = SimpleNamespace(model=model)
        x = torch.full((2,2,2,2), .02)
        old_curves = block._tc_curves_for_solve()['FFconv'].clone()
        _, old_spin, _ = block._tc_prepare_noise(x)
        TrainerCiFar.reset_spin_variation_for_inference(trainer)
        new_curves = block._tc_curves_for_solve()['FFconv']
        _, new_spin, _ = block._tc_prepare_noise(x)
        self.assertFalse(torch.equal(old_curves, new_curves))
        self.assertFalse(torch.equal(old_spin, new_spin))
        torch.testing.assert_close(new_curves, block._tc_curves_for_solve()['FFconv'])
        torch.testing.assert_close(new_spin, block._tc_prepare_noise(x)[1])

    def test_0906_activation_defaults(self):
        ft, evaluation = resolved('ft'), resolved('eval')
        self.assertTrue(ft.activation_curve_path.endswith('0906_RELU_Voltage/tt_25_1.csv'))
        self.assertEqual(ft.activation_corner, 'MC18')
        self.assertTrue(evaluation.activation_curve_path.endswith('0906_RELU_Voltage'))
        self.assertEqual(evaluation.activation_corner, 'TT_25_1_MC18')
        self.assertEqual(evaluation.activation_curve_sharing, 'per_spin')

    def test_local_commands_reach_real_parsers(self):
        for state, block in [('1', 'ODEXInitFFFB'), ('2', 'S2NoisyIYAsXZAs0')]:
            for stage in ('ft', 'eval'):
                args = resolved(stage, state)
                self.assertEqual(args.ode_block, block)
                self.assertIsNone(args.enob)
                self.assertIsNone(args.weight_quant_factor_bits)
                self.assertEqual(args.w_bits, 5)
                self.assertTrue(args.nonlinear_R)
                self.assertTrue(args.enable_measured_pooling)
                self.assertTrue(args.enable_measured_activation)
                opts = wrapper_options(args)
                self.assertTrue(opts['enable_summing_current_noise'])
                self.assertTrue(opts['enable_coupler_noise'])
                self.assertEqual(opts['summing_current_p'], .6e-12)
                self.assertEqual(opts['coupler_noise_p'], .6e-12)
                if stage == 'eval':
                    self.assertEqual(args.noisy_trials, 10)
                    self.assertEqual(args.nonlinear_R_curve_sharing, 'per_coupler')
                    self.assertEqual((args.d_start,args.d_end,args.n_sweep_right), (0,1,1))
                    self.assertFalse(args.diff_mismatch)
                else:
                    self.assertEqual(args.noise_level, 0)

    def test_explicit_switches_and_trial_seeds(self):
        args = resolved('eval', N_TRIALS='3', ENABLE_COUPLER_NOISE='false',
                        SPIN_VARIATION_MEAN='1.2', NONLINEAR_R_CURVE_SEED='71')
        self.assertEqual(args.noisy_trials, 3)
        self.assertFalse(args.enable_coupler_noise)
        self.assertEqual(args.spin_variation_mean, 1.2)
        a, b = wrapper_options(args,0), wrapper_options(args,1)
        for key in ('spin_variation_seed','summing_noise_seed','coupler_noise_seed',
                    'activation_curve_seed','nonlinear_R_curve_seed'):
            self.assertEqual(b[key], a[key]+1)
        self.assertEqual(a['nonlinear_R_curve_seed'],71)

    def test_disabled_shell_helper_is_empty(self):
        out = subprocess.check_output(['bash','-c',
            'TC_NONIDEALITIES=false; source launch_scripts/tc_nonideality_args.sh ft; '
            'echo ${#TC_ARGS[@]}'], cwd=ROOT,text=True)
        self.assertEqual(out.strip(),'0')

    def test_probe_reset_and_metadata(self):
        block = make_block()
        wrapper = wrap(block, enable_spin_variation=True, enable_coupler_noise=True,
                       spin_variation_seed=31, coupler_noise_seed=41)
        model = nn.Module(); model.PcConvs = nn.ModuleList([block]); model.eval()
        x = torch.full((2,2,2,2), .02)
        reset_after_probe(model)
        c1, y1, z1 = block._tc_prepare_noise(x)
        draw1 = c1.normal(x,0).clone()
        reset_after_probe(model)
        c2, y2, z2 = block._tc_prepare_noise(x)
        torch.testing.assert_close(y1,y2)
        torch.testing.assert_close(z1,z2)
        torch.testing.assert_close(draw1,c2.normal(x,0))
        with tempfile.TemporaryDirectory() as directory:
            args = resolved('eval', TC_METADATA_PATH=directory+'/trials.jsonl')
            record_trial(args,model,1,12.5,'./inspection_only.pth')
            data = json.loads(Path(args.tc_metadata_path).read_text())
            self.assertEqual(data['accuracy_percent'],12.5)
            self.assertEqual(data['data_seed'],args.data_seed+1)
            self.assertIn('physical_duration',data['layers'][0])
            self.assertEqual(data['resolved_trial_options']['nonlinear_R_curve_seed'],
                             args.nonlinear_R_curve_seed+1)

    def test_slurm_dry_run_exports(self):
        env = dict(os.environ, TC_NONIDEALITIES='true', TC_DRY_RUN='true',
                   TC_STATE='2', TOGGLE_MODE='none', SWITCH_INF='false',
                   N_TRIALS='3', TC_FB_ASD_PATH='./reference.csv',
                   TC_EVAL_ACTIVATION_CURVE_PATH='./eval_bank',
                   TC_EVAL_ACTIVATION_CORNER='eval_corner',
                   TC_CONV_METHOD='shared',TC_CURVE_SAMPLING='uniform',
                   MEASURED_POOLING_CURVE_PATH='./pooling', TASK='cifar100')
        out = subprocess.check_output(['bash','launch_scripts/slurm_search_config.sh'],
                                     env=env,cwd=ROOT,text=True,timeout=30)
        commands = [shlex.split(line) for line in out.splitlines() if line.startswith('sbatch ')]
        self.assertTrue(commands)
        for command in commands:
            exports = next(x for x in command if x.startswith('--export='))
            for field in ('TC_NONIDEALITIES=true','TC_STATE=2','N_TRIALS=3',
                          'ODE_BLOCK=S2NoisyIYAsXZAs0','TC_FB_ASD_PATH=./reference.csv',
                          'MEASURED_POOLING_CURVE_PATH=./pooling',
                          'TC_EVAL_ACTIVATION_CURVE_PATH=./eval_bank',
                          'TC_EVAL_ACTIVATION_CORNER=eval_corner',
                          'TC_CONV_METHOD=shared','TC_CURVE_SAMPLING=uniform'):
                self.assertIn(field,exports)

    def test_actual_sbatch_stage_commands_without_execution(self):
        # Run only declarations and the two stage functions. Replace Python by
        # an argv printer, suppress conda activation, redirect logs to /tmp.
        # No pretraining/FT/evaluation process and no sbatch submission occurs.
        source = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        prefix = source[:source.index('# PHASE 1:')]
        prefix = prefix.replace('source activate base','true').replace('conda activate scanbase','true')
        with tempfile.TemporaryDirectory() as directory:
            prefix = '\n'.join(('LOGDIR='+shlex.quote(directory)) if line.startswith('LOGDIR=')
                               else line for line in prefix.splitlines())
            name = 'PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_1.75TEnd_16Layers'
            commands = prefix + '''
python() { printf 'CAPTURE '; printf '%q ' "$@"; printf '\n'; }
finetune_one_combo_model inspect "$MODEL_NAME" "$ODE_BLOCK"
eval_one_combo_model inspect "$MODEL_NAME" post_ft "$ODE_BLOCK"
finetune_one_combo_model inspect "$MODEL_NAME" "$ODE_BLOCK"
'''
            for state in ('1','2'):
                env = dict(os.environ, TC_NONIDEALITIES='true', TC_STATE=state,
                           TOGGLE_MODE='none', SWITCH_INF='false', COMB_LIST='inspect',
                           MODEL_NAME=name, TASK='cifar100', IMG_TYPE='CiFAIR',
                           OUTPUT_SAVE_PATH=directory, N_TRIALS='3')
                out = subprocess.check_output(['bash','-c',commands],env=env,cwd=ROOT,text=True)
                captured = [shlex.split(line)[2:] for line in out.splitlines() if line.startswith('CAPTURE ')]
                self.assertEqual(len(captured),3)
                for argv, parser, stage in zip(captured,(get_args,parse_args,get_args),('ft','eval','ft')):
                    error = io.StringIO()
                    try:
                        with patch.object(sys,'argv',argv), contextlib.redirect_stderr(error):
                            args = parser()
                    except SystemExit:
                        self.fail(error.getvalue().splitlines()[-1])
                    local = resolved(stage,state)
                    for key in ('ode_block','ode_wrapper','w_bits','R','R_max','C','v_dd','one_over_q','k','sde_noise_type',
                                'enob','weight_quant_factor_bits','nonlinear_R','nonlinear_R_table',
                                'tc_covariance_table','enable_measured_pooling','enable_measured_activation',
                                'activation_curve_path','activation_corner',
                                'measured_pooling_curve_path','enable_spin_variation',
                                'enable_summing_current_noise','enable_coupler_noise',
                                'summing_current_p','coupler_noise_p','nonlinear_R_curve_sharing'):
                        self.assertEqual(getattr(args,key),getattr(local,key),key)
                    if stage == 'eval':
                        self.assertEqual(args.activation_curve_sharing, 'per_spin')

                # Only explicit evaluation overrides may narrow/change its bank.
                env.update(TC_EVAL_ACTIVATION_CURVE_PATH='./eval_bank',
                           TC_EVAL_ACTIVATION_CORNER='eval_corner')
                out = subprocess.check_output(['bash','-c',commands],env=env,cwd=ROOT,text=True)
                captured = [shlex.split(line)[2:] for line in out.splitlines() if line.startswith('CAPTURE ')]
                with patch.object(sys, 'argv', captured[1]):
                    evaluation = parse_args()
                self.assertEqual(evaluation.activation_curve_path, './eval_bank')
                self.assertEqual(evaluation.activation_corner, 'eval_corner')


if __name__ == '__main__':
    unittest.main()
