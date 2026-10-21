import json
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import pcn_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_supported_training_and_evaluation_activation_preserve_zero_noise_logits(self):
        import torch
        from inference_utils import load_and_prepare_model
        from ode_pc import ODEXInitFFFB, make_ode_block
        from pc_conv import PCConvNoisy
        from pc_model import PCNetBoundaryBN, PCNetNoBatchNorm

        torch.set_num_threads(1)
        options = dict(ode_block=ODEXInitFFFB, method='euler', t_end=.2, n_steps=5)
        for cls in (PCNetNoBatchNorm, PCNetBoundaryBN):
            with self.subTest(model=cls.__name__), tempfile.TemporaryDirectory() as directory:
                with patch('torch.cuda.is_available', return_value=False), torch.no_grad():
                    model = cls(inp_channels=[3, 4], out_channels=[4, 4], max_pool=[0, 1],
                                num_classes=10, cls=0, first_bn=False, bias=False, bypass=False)
                    checkpoint = Path(directory) / 'model.pth'
                    torch.save(dict(net_type=cls.__name__, init_args=model.init_args,
                                    net=model.state_dict()), checkpoint)
                    make_ode_block(model, **options)
                    loaded = load_and_prepare_model(str(checkpoint), device=torch.device('cpu'),
                                noise_level=0., pc_conv_layer=PCConvNoisy, fuse_bn=False,
                                noise_to_bn=False, noise_to_linear=True, noise_to_conv_bias=False,
                                ode_params=options)
                model.eval()
                loaded.eval()
                inputs = torch.randn(2, 3, 4, 4)
                with torch.no_grad():
                    torch.testing.assert_close(loaded(inputs), model(inputs), atol=1e-5, rtol=0)

    def test_unsupported_activation_refused_before_submission_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, BACKEND='slurm', RUN_ROOT=str(Path(directory) / 'run'),
                       PCCONV='PCConvReLU6', REPO_ROOT=str(pipeline.REPO),
                       PIPELINE_JOB_DIR=str(Path(directory) / 'job'), IS_SLURM='0')
            for script in ('launch_pcn_pipeline.sh', 'run_pcn_pipeline.sh'):
                result = subprocess.run(['bash', str(pipeline.REPO / 'launch_scripts' / script)],
                                        cwd=pipeline.REPO, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('requires PCCONV=PCConv', result.stderr)
            self.assertFalse((Path(directory) / 'run').exists())
            self.assertFalse((Path(directory) / 'job').exists())

    def test_worker_lock_survives_helper_exit_and_rejects_duplicate_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / 'job'
            ready = root / 'ready'
            env = dict(os.environ, REPO_ROOT=str(pipeline.REPO), PIPELINE_JOB_DIR=str(job),
                       STAGE='train-test', IS_SLURM='0', ARCHITECTURE='WRN_16_2',
                       CONDITIONS='max_additive', PCCONV='PCConv', READY=str(ready),
                       PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'])
            env.pop('REHEARSAL_ROOT', None)
            shell = '''
source() {
  case "$1" in
    */run_rgb_ode_train.sh) touch "$READY"; read -r signal; return 7 ;;
    *) builtin source "$@" ;;
  esac
}
builtin source launch_scripts/run_pcn_pipeline.sh
'''
            proc = subprocess.Popen(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                    text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(ready.exists(), 'First worker did not reach training')
                before = (job / 'state.json').read_bytes()
                duplicate = subprocess.run(['bash', str(pipeline.REPO / 'launch_scripts/run_pcn_pipeline.sh')],
                                           env={**env, 'STAGE': 'test'}, cwd=pipeline.REPO,
                                           text=True, capture_output=True, timeout=10)
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertIn('Another pipeline worker owns this job', duplicate.stderr)
                self.assertEqual((job / 'state.json').read_bytes(), before)
            finally:
                stdout, stderr = proc.communicate('release\n', timeout=10)
            self.assertEqual(proc.returncode, 7, stdout + stderr)
            with pipeline.job_lock_path(job).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_real_submission_matrix_and_recipe_without_login_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'real'
            capture = Path(directory) / 'submitted'
            env = dict(os.environ, BACKEND='slurm', RUN_ROOT=str(root), CAPTURE=str(capture))
            for key in pipeline.CONFIG_KEYS + ('PCNS', 'DATASETS', 'SIZES', 'REHEARSAL'):
                env.pop(key, None)
            # Exercise the real sourced submission script; only sbatch is absent locally.
            shell = '''
python() { echo "Unexpected login-node Python" >&2; return 99; }
sbatch() {
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$PCN" "$DATASET_NAME" "$ARCHITECTURE" "$INP_CHANNELS" "$OUT_CHANNELS" "$MAX_POOL" "$BN_EVAL_MODE" "$REUSE_ACCEPTED_STEP_TRAINING" "$WARMUP_EPOCH" "$CONDITIONS" "$*" >> "$CAPTURE"
  echo 12345
}
source launch_scripts/launch_pcn_pipeline.sh
'''
            result = subprocess.run(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [line.split('|') for line in capture.read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({tuple(row[:3]) for row in rows}), 16)
            for row in rows:
                self.assertEqual(row[7:10], ['true', '5', 'max_additive,multiplicative,rms_additive'])
                self.assertEqual(row[6], 'paired' if row[0] == 'PCNetBoundaryBN' else 'legacy')
                self.assertIn('--export=ALL,IS_SLURM=1', row[10])
                self.assertTrue(row[10].endswith('/launch_scripts/run_pcn_pipeline.sh'))
                if row[2] == 'WRN_28_4':
                    self.assertEqual(row[3], '3 16 64 64 64 64 128 128 128 128 256 256 256')
                    self.assertEqual(row[5], '0 0 0 0 0 1 0 0 0 1 0 0 0')

    def test_rehearsal_root_must_be_separate_and_new(self):
        with self.assertRaises(ValueError):
            pipeline.initialize_rehearsal(pipeline.REPO / 'logs' / 'rehearsal')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                pipeline.initialize_rehearsal(directory)

    def make_job(self, root, rehearsal=False):
        job = root / 'job'
        job.mkdir()
        config = {key: '' for key in pipeline.CONFIG_KEYS}
        config.update(DATASET_NAME='cifar10', ARCHITECTURE='WRN_16_2',
                      PCCONV='PCConv',
                      CONDITIONS='max_additive', BN_EVAL_MODE='legacy',
                      MAX_ADD_LEVELS='0,0.1', NOISY_TRIALS='10',
                      REHEARSAL_ROOT=str(root) if rehearsal else '')
        pipeline.write_json(job / 'config.json', config)
        name = 'PCNetNoBatchNorm_test'
        checkpoint = job / 'checkpoints' / name / f'{name}_best_ckpt.pth'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint')
        record = dict(status='complete', config=config, rehearsal=rehearsal,
                      model_name=name, checkpoint=str(checkpoint), checkpoint_sha256=pipeline.digest(checkpoint))
        pipeline.write_json(job / 'train_complete.json', record)
        return job, checkpoint

    def test_test_worker_releases_lock_after_successful_handoff_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _ = self.make_job(Path(directory))
            output = job / 'results/cifar10/WRN_16_2/max_additive/result.pkl'
            output.parent.mkdir(parents=True)
            output.write_bytes(pickle.dumps({1.75: {'noise_acc_spec': {
                'Johnson': {0.0: [90.0], 0.1: [50.0] * 10}}}}))
            config = json.loads((job / 'config.json').read_text())
            env = dict(os.environ, **config, REPO_ROOT=str(pipeline.REPO),
                       PIPELINE_JOB_DIR=str(job), STAGE='test', IS_SLURM='0',
                       PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'])
            shell = '''
source() {
  case "$1" in
    */run_rgb_ode_mismatch_eval.sh) return 0 ;;
    *) builtin source "$@" ;;
  esac
}
builtin source launch_scripts/run_pcn_pipeline.sh
'''
            for _ in range(2):
                result = subprocess.run(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads((job / 'state.json').read_text())['status'], 'complete')
                with pipeline.job_lock_path(job).open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_handoff_hash_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            job, checkpoint = self.make_job(Path(directory))
            pipeline.handoff(job)
            self.assertIn(str(checkpoint.parent.parent), (job / 'handoff.env').read_text())
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                pipeline.handoff(job)
            with self.assertRaises(ValueError):
                pipeline.contained(job / 'checkpoints' / '..' / 'foreign.pth', job / 'checkpoints')

    def test_finish_requires_meaningful_full_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _ = self.make_job(Path(directory))
            output = job / 'results/cifar10/WRN_16_2/max_additive/result.pkl'
            output.parent.mkdir(parents=True)
            def save(trials):
                output.write_bytes(pickle.dumps({1.75: {'noise_acc_spec': {'Johnson': trials}}}))
            save({0.0: [90.0], 0.1: [50.0]})
            with self.assertRaises(ValueError):
                pipeline.finish(job)
            save({0.0: [90.0], 0.1: [50.0] * 10})
            pipeline.finish(job)
            self.assertEqual(json.loads((job / 'state.json').read_text())['status'], 'complete')

    def test_production_refuses_rehearsal_and_existing_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline.write_json(root / 'rehearsal.json', {'production_before': {}})
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    pipeline.prepare(root / 'job', 'train')
            job, _ = self.make_job(root, rehearsal=True)
            with patch.dict(os.environ, {'REHEARSAL_ROOT': str(root), 'CONDITIONS': 'max_additive'}, clear=True):
                with self.assertRaises(FileExistsError):
                    pipeline.prepare(job, 'train-test')

    def test_failed_train_never_starts_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / 'job'
            shell = '''
python() {
  case "$2" in
    lock-path) echo "$PIPELINE_JOB_DIR.lock" ;;
    prepare) mkdir -p "$PIPELINE_JOB_DIR"; printf '{}' > "$PIPELINE_JOB_DIR/config.json" ;;
    fail) return 0 ;;
    *) echo "Unexpected handoff/evaluation after failed train"; return 99 ;;
  esac
}
source() {
  case "$1" in
    */run_rgb_ode_train.sh) return 7 ;;
    *) builtin source "$@" ;;
  esac
}
builtin source launch_scripts/run_pcn_pipeline.sh
'''
            env = dict(os.environ, REPO_ROOT=str(pipeline.REPO), PIPELINE_JOB_DIR=str(job),
                       STAGE='train-test', IS_SLURM='0', ARCHITECTURE='WRN_16_2')
            result = subprocess.run(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertNotIn('Unexpected', result.stdout)

    def test_refused_repeat_leaves_existing_status_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _ = self.make_job(Path(directory))
            pipeline.write_json(job / 'state.json', {'status': 'complete'})
            env = dict(os.environ, REPO_ROOT=str(pipeline.REPO), PIPELINE_JOB_DIR=str(job),
                       STAGE='train-test', IS_SLURM='0', ARCHITECTURE='WRN_16_2',
                       PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'])
            env.pop('REHEARSAL_ROOT', None)
            result = subprocess.run(['bash', str(pipeline.REPO / 'launch_scripts/run_pcn_pipeline.sh')],
                                    cwd=pipeline.REPO, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((job / 'state.json').read_text()), {'status': 'complete'})

    def test_pack_is_small_and_separates_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, checkpoint = self.make_job(root)
            pipeline.write_json(job / 'state.json', {'status': 'complete'})
            output = job / 'results/cifar10/WRN_16_2/max_additive/result.pkl'
            output.parent.mkdir(parents=True)
            output.write_bytes(pickle.dumps({1.75: {'noise_acc_spec': {
                'Johnson': {0.0: [90.0], 0.1: [50.0] * 10}}}}))
            (job / 'expanded_weights').mkdir()
            (job / 'expanded_weights/huge.pt').write_bytes(b'not included')
            archive = root / 'results.tar.gz'
            pipeline.pack(root, archive, False)
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
            self.assertTrue(any(n.endswith('train_complete.json') for n in names))
            self.assertFalse(any('expanded_weights' in n or 'checkpoints' in n for n in names))
            models = root / 'models.tar'
            pipeline.pack(root, models, True)
            with tarfile.open(models) as bundle:
                self.assertEqual(bundle.getnames(), [str(checkpoint.relative_to(root))])
            with self.assertRaises(FileExistsError):
                pipeline.pack(root, models, True)
            output.unlink()
            with self.assertRaises(FileNotFoundError):
                pipeline.pack(root, root / 'invalid.tar.gz', False)
            self.assertFalse((root / 'invalid.tar.gz').exists())

    def test_inventory_catches_missing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, _ = self.make_job(root)
            (root / 'planned_jobs.tsv').write_text('PCNetNoBatchNorm\tcifar10\t16_2\nPCNetBoundaryBN\tcifar10\t16_2\n')
            config = json.loads((job / 'config.json').read_text())
            config['PCN'] = 'PCNetNoBatchNorm'
            pipeline.write_json(job / 'config.json', config)
            with self.assertRaises(ValueError):
                pipeline.validate_inventory(root, [job / 'config.json'])

    def test_pool_and_fixed_step_checkpoint_names_are_distinct(self):
        from train_ode_cifar import get_args, _constr_model_name
        argv = ['train', '--pcn', 'PCNetNoBatchNorm', '--pc_conv', 'PCConv',
                '--distill_method', 'none', '--method', 'dopri5']
        with patch.object(sys, 'argv', argv):
            args = get_args()
        original = _constr_model_name(args)
        self.assertNotIn('_avgP_', original)
        args.avg_pooling = True
        self.assertNotEqual(_constr_model_name(args), original)
        args.method = 'euler'
        args.n_steps = 5
        five = _constr_model_name(args)
        self.assertIn('_avgP_5Steps_', five)
        args.n_steps = 10
        self.assertNotEqual(_constr_model_name(args), five)
        self.assertLess(len(five + '_best_ckpt.pth'), 256)

    def test_bn_rehearsal_uses_clean_load_then_explicit_mismatch(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from pc_model import PCNetBoundaryBN
        from ode_inference import parse_args
        from inference_utils import load_and_prepare_model
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline.write_json(root / 'rehearsal.json', {'production_before': {}})
            job, checkpoint = self.make_job(root, rehearsal=True)
            net = PCNetBoundaryBN(inp_channels=[3, 4], out_channels=[4, 4],
                                  max_pool=[0, 1], num_classes=10, cls=0,
                                  first_bn=False, bias=False, bypass=False)
            torch.save(dict(net_type='PCNetBoundaryBN', init_args=net.init_args,
                            net=net.state_dict()), checkpoint)
            record = json.loads((job / 'train_complete.json').read_text())
            record.update(checkpoint_sha256=pipeline.digest(checkpoint), args=dict(
                pcn='PCNetBoundaryBN', method='dopri5', n_steps=10, avg_pooling=False,
                ode_block='ODEXInitFFFB', dataset='cifar10'))
            pipeline.write_json(job / 'train_complete.json', record)
            inputs = torch.randn(2, 3, 4, 4)
            test = DataLoader(TensorDataset(inputs, torch.tensor([0, 1])), batch_size=2)
            calibration = DataLoader(inputs, batch_size=2)
            output = job / 'results/max_additive/result.pkl'
            argv = ['test', '--model_dir', str(checkpoint.parent.parent),
                    '--model_name', checkpoint.parent.name, '--t_end', '1.75',
                    '--pc_conv', 'PCConvNoisy', '--ode_block', 'ODEXInitFFFB',
                    '--bn_eval_mode', 'paired', '--thermal_noise', 'false',
                    '--noise_to_conv_bias', 'false', '--seed', '123', '--test_bs', '2',
                    '--noise_level_list', '0,0.1', '--output_pickle', str(output),
                    '--expanded_w_dir', str(job / 'expanded_weights'),
                    '--rehearsal_root', str(root)]
            with patch.object(sys, 'argv', argv):
                args = parse_args()
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torchvision.datasets.CIFAR10'), \
                 patch('inference_utils.get_test_data', return_value=test), \
                 patch('baseline.run_baseline.build_bn_calibration_loader', return_value=calibration), \
                 patch('inference_utils.load_and_prepare_model', wraps=load_and_prepare_model) as load:
                pipeline.rehearse_evaluation(args)
            self.assertEqual(load.call_args.kwargs['noise_level'], 0.0)
            value = json.loads((output.parent / 'rehearsal_initialization.json').read_text())
            self.assertIn('frozen_accuracy', value)
            self.assertIn('recalibrated_accuracy', value)
            self.assertFalse(output.exists())

    def test_sourced_eval_failure_returns_to_caller_and_stops_other_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'PCNetNoBatchNorm_1.75TEnd'
            checkpoint = root / name / f'{name}_best_ckpt.pth'
            checkpoint.parent.mkdir()
            checkpoint.touch()
            env = dict(os.environ, REPO_ROOT=str(pipeline.REPO), MODEL_DIR=str(root),
                       MODEL_NAME=name, MODEL_INDEX='0', ARCHITECTURE='WRN_16_2',
                       ODE_BLOCK='ODEXInitFFFB', PC_CONV='PCConvNoisy', IS_SLURM='0',
                       OUTPUT_ROOT=str(root / 'results'), DATASETS='all',
                       CONDITIONS='max_additive,multiplicative,rms_additive')
            shell = '''
python() { return 42; }
source launch_scripts/run_rgb_ode_mismatch_eval.sh
status=$?
echo "caller survived: $status"
test "$status" = 42
'''
            result = subprocess.run(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('caller survived: 42', result.stdout)
            self.assertEqual(result.stdout.count('Running cifar10'), 1)

    def test_missing_completion_blocks_eval_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / 'job'
            env = dict(os.environ, REPO_ROOT=str(pipeline.REPO), PIPELINE_JOB_DIR=str(job),
                       STAGE='train-test', IS_SLURM='0', ARCHITECTURE='WRN_16_2',
                       CONDITIONS='max_additive',
                       PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'])
            env.pop('REHEARSAL_ROOT', None)
            shell = '''
source() {
  case "$1" in
    */run_rgb_ode_train.sh) return 0 ;;
    *) builtin source "$@" ;;
  esac
}
builtin source launch_scripts/run_pcn_pipeline.sh
'''
            result = subprocess.run(['bash', '-c', shell], cwd=pipeline.REPO, env=env,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((job / 'state.json').read_text())['status'], 'failed')
            self.assertFalse((job / 'results').exists())


if __name__ == '__main__':
    unittest.main()
