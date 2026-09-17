"""Execute real Slurm/combined/stage shells with fake training, never submit."""
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def fake_python(argv):
    if argv[0] == '-':
        # Run the real lightweight checkpoint-directory resolver from stdin.
        os.execv(sys.executable, [sys.executable, *argv])
    options = dict(zip(argv[1::2], argv[2::2]))
    stage = ('eval' if 'evaluate_' in argv[0] else
             'pretrain' if options.get('--physical_pretraining') == 'true' else 'ft')
    with open(os.environ['PIPELINE_TRACE'], 'a') as handle:
        handle.write(json.dumps(dict(stage=stage, argv=argv, parent_pid=os.getppid()))+'\n')
    if os.environ.get('FAIL_STAGE') == stage:
        return 42
    if stage == 'eval':
        if not Path(options['--checkpoint']).is_file():
            return 43
        print('FAKE EVALUATION: no numerical inference performed')
        return 0
    if os.environ.get('MISSING_CHECKPOINT') == stage:
        return 0
    sys.path.insert(0, str(ROOT))
    from input_preprocessing import resolve_preprocessing, append_preprocessing_suffix
    bits = options.get('--input_quant_bits', 'none')
    bits = None if bits == 'none' else int(bits)
    center = options.get('--center_student_input', 'auto')
    center = None if center == 'auto' else center.lower() in ('true', '1', 'yes')
    path = options['--output_dir']
    bits, center = resolve_preprocessing(options.get('--resume_checkpoint', path), bits, center)
    path = Path(append_preprocessing_suffix(path, bits, center))
    model, task = options['--model_name'], options['--dataset']
    name = f'custom_noresize_{task}_{model}'
    final_only = options.get('--final_eval_only') == 'true'
    selection = 'last' if final_only else 'best'
    suffix = f'full_param_{selection}' if stage == 'ft' else selection
    checkpoint = path/task/'custom_noresize'/model/name/f'{name}_{suffix}_ckpt.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()  # Fixture, deliberately not a real model checkpoint.
    return 0


class TCPipelineTests(unittest.TestCase):
    def run_pipeline(self, directory, **overrides):
        directory = Path(directory)
        bindir = directory/'bin'
        bindir.mkdir()
        stub = bindir/'python'
        stub.write_text(f'#!/bin/bash\nexec {sys.executable} {Path(__file__).resolve()} --fake-python "$@"\n')
        stub.chmod(0o755)
        (bindir/'activate').write_text('[[ "$1" == base ]] || return 44\n')
        env = {k: os.environ[k] for k in ('HOME', 'LD_LIBRARY_PATH') if k in os.environ}
        env.update(PATH=str(bindir)+':'+os.environ['PATH'], TC_FEEDFORWARD='true',
                   REPO_ROOT=str(ROOT), MODEL_NAME='wrn_28_2_cifar_nobn_no_bias_avgpool',
                   TASK='cifar100', IMG_TYPE='rgb', PRETRAIN_OUTPUT_DIR=str(directory/'pre'),
                   FT_OUTPUT_DIR=str(directory/'ft'), RESULT_PATH=str(directory/'results'),
                   PIPELINE_TRACE=str(directory/'trace.jsonl'))
        env.update(overrides)
        script = '''
conda() { [[ "$1" == activate && "$2" == scanbase ]]; }
export -f conda
sbatch() { bash "${@: -1}"; }
source launch_scripts/slurm_search_feedforward_config.sh
'''
        result = subprocess.run(['bash', '-c', script], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        trace = directory/'trace.jsonl'
        rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, rows

    def test_worker_runs_three_stages_with_pcn_evaluation_defaults(self):
        from test_tc_launch_alignment import parsed, cnn_eval, pcn_stage
        for preprocessing in ({}, {'INPUT_QUANT_BITS': '12', 'CENTER_STUDENT_INPUT': 'true', 'N_TRIALS': '2'}):
            with self.subTest(preprocessing=preprocessing), tempfile.TemporaryDirectory() as directory:
                result, rows = self.run_pipeline(directory, **preprocessing)
                self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
                self.assertEqual([r['stage'] for r in rows], ['pretrain', 'ft', 'eval'])
                self.assertEqual(rows[0]['parent_pid'], rows[1]['parent_pid'])
                pre = dict(zip(rows[0]['argv'][1::2], rows[0]['argv'][2::2]))
                ft = dict(zip(rows[1]['argv'][1::2], rows[1]['argv'][2::2]))
                self.assertEqual(pre['--timm_aug_level'], 'none')
                self.assertEqual(ft['--timm_aug_level'], 'no_aug')
                self.assertEqual(pre['--tc_intermediate_activation'], 'relu6')
                self.assertEqual(ft['--tc_intermediate_activation'], 'relu6')
                eval_args = dict(zip(
                    rows[-1]['argv'][1::2], rows[-1]['argv'][2::2]))
                self.assertEqual(
                    eval_args['--tc_intermediate_activation'], 'relu6')
                self.assertIn('lr=0.1,num_epochs=300,', pre['--override'])
                self.assertIn('lr=0.005,num_epochs=140,', ft['--override'])
                cnn = parsed(cnn_eval, rows[-1]['argv'])
                pcn = pcn_stage('eval', N_TRIALS=preprocessing.get('N_TRIALS', '10'))
                self.assertIn('/ft'+('_iq12_ctr' if preprocessing else '')+'/', cnn.checkpoint)
                self.assertTrue(cnn.checkpoint.endswith('_full_param_last_ckpt.pth'))
                self.assertEqual(cnn.n_trials, pcn.noisy_trials)
                self.assertTrue(cnn.use_expanded_weights)
                self.assertTrue(cnn.enable_nonlinear_R)
                self.assertEqual(cnn.nonlinear_R_train_mode, 'none')
                for key in ('activation_curve_path', 'activation_corner', 'activation_curve_sharing',
                            'nonlinear_R_curve_sharing', 'nonlinear_R_table', 'tc_covariance_table',
                            'tc_curve_sampling', 'enable_measured_pooling', 'measured_pooling_curve_path',
                            'measured_pooling_nominal_R', 'enable_spin_variation', 'sigma_spin',
                            'enable_summing_current_noise', 'summing_current_p',
                            'enable_coupler_noise', 'coupler_noise_p', 'tc_noise_reference_R',
                            'R', 'C', 'v_dd', 'enob'):
                    self.assertEqual(getattr(cnn, key), getattr(pcn, key), key)
                self.assertFalse(cnn.one_shot_conv)
                self.assertEqual(cnn.tc_method, pcn.method)
                self.assertEqual(cnn.tc_tol, pcn.tol)
                self.assertIn('FAKE EVALUATION', (Path(directory)/'results/evaluation.log').read_text())

    def test_pcn_training_seed_resets_python_numpy_and_torch(self):
        from train_ode_cifar import seed_training

        seed_training(4096)
        first = (random.random(), np.random.rand(), torch.rand(3))
        seed_training(4096)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])

    def test_failure_never_runs_dependent_stage(self):
        for failure, expected in (
            ({'FAIL_STAGE': 'pretrain'}, ['pretrain']),
            ({'MISSING_CHECKPOINT': 'pretrain'}, ['pretrain']),
            ({'FAIL_STAGE': 'ft'}, ['pretrain', 'ft']),
            ({'MISSING_CHECKPOINT': 'ft'}, ['pretrain', 'ft']),
            ({'FAIL_STAGE': 'eval'}, ['pretrain', 'ft', 'eval']),
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                result, rows = self.run_pipeline(directory, **failure)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual([r['stage'] for r in rows], expected)
                self.assertNotIn('evaluation completed successfully', result.stdout)

    def test_non_tc_pipeline_still_stops_after_ft(self):
        with tempfile.TemporaryDirectory() as directory:
            result, rows = self.run_pipeline(directory, TC_FEEDFORWARD='false')
            self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
            self.assertEqual([r['stage'] for r in rows], ['pretrain', 'ft'])

    def test_explicit_stage_overrides_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result, rows = self.run_pipeline(
                directory, TIMM_AUG_LEVEL='no_aug', EXTRA_OVERRIDE='lr=0.003,num_epochs=2')
            self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
            for row in rows[:2]:
                args = dict(zip(row['argv'][1::2], row['argv'][2::2]))
                self.assertEqual(args['--timm_aug_level'], 'no_aug')
                self.assertEqual(args['--override'], 'lr=0.003,num_epochs=2')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fake-python':
        sys.exit(fake_python(sys.argv[2:]))
    unittest.main()
