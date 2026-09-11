import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import os
import subprocess
import sys

from baseline.run_wrn_controls import parse_args, paths, run_one, test_command
from baseline.wrn_control_artifacts import collect, create_plan, pack


class PipelineTests(unittest.TestCase):
    def test_slurm_row_four_training_eight_evaluations_sequential_conditions(self):
        with tempfile.TemporaryDirectory() as root:
            env = dict(os.environ, SIMULATE='1', DRY_RUN='0', ROWS='4',
                       DATASETS='cifar10,cifar100', SIZES='16_2,16_4,28_2,28_4',
                       PARALLELISM='4', EVAL_PARALLELISM='8', STAGE='train-test',
                       CONDITIONS='max_additive,multiplicative,rms_additive',
                       OUTPUT_ROOT=root, PYTHON_BIN=sys.executable, SIMULATE_FAILURE='')
            subprocess.run(['bash', 'launch_scripts/slurm_run_wrn_controls.sh'],
                           cwd=Path(__file__).resolve().parents[1], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            output = Path(root) / 'SIMULATED'
            inventory = collect(output)
            self.assertEqual(inventory['complete_tasks'], 8)
            training = [json.loads(p.read_text()) for p in output.glob('row4/*/*/checkpoints/simulation_lifecycle.json')]
            def peak(lives):
                events = sorted([(v['started'], 1) for v in lives] + [(v['finished'], -1) for v in lives])
                current = maximum = 0
                for _, delta in events:
                    current += delta
                    maximum = max(maximum, current)
                return maximum
            self.assertEqual(peak(training), 4)
            previous_end = max(v['finished'] for v in training)
            for condition in env['CONDITIONS'].split(','):
                lives = [json.loads(p.read_text()) for p in output.glob(f'row4/*/*/evaluation/{condition}/simulation_lifecycle.json')]
                self.assertEqual(len(lives), 8)
                self.assertGreaterEqual(min(v['started'] for v in lives), previous_end)
                self.assertEqual(peak(lives), 8)
                previous_end = max(v['finished'] for v in lives)

    def args(self, root, *extra):
        return parse_args(['--output-root', str(root), '--simulate', '--stage', 'train-test',
                           '--rows', '4', '--datasets', 'cifar100', '--sizes', '16_2',
                           '--conditions', 'max_additive,multiplicative,rms_additive', *extra])

    def test_handoff_parallelism_identity_resume_and_archive(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            create_plan(args.output_root, [4], args.datasets, args.sizes, args.conditions, args.stage, True)
            run_one(args, 4, 'cifar100', '16_2')
            directory, checkpoint = paths(args, 4, 'cifar100', '16_2')
            training = json.loads((directory / 'checkpoints/simulation_lifecycle.json').read_text())
            lifecycle = []
            mtimes = []
            for condition in args.conditions:
                folder = directory / 'evaluation' / condition
                life = json.loads((folder / 'simulation_lifecycle.json').read_text())
                lifecycle.append(life)
                self.assertGreaterEqual(life['started'], training['finished'])
                completed = json.loads((folder / 'complete.json').read_text())
                self.assertTrue(completed['identity']['simulated'])
                command = completed['identity']['command']
                self.assertEqual(command[command.index('--checkpoint_override') + 1], str(checkpoint))
                standalone = self.args(root, '--stage', 'test')
                self.assertEqual(command, test_command(standalone, 4, 'cifar100', '16_2', condition))
                mtimes.append((folder / 'run.log').stat().st_mtime_ns)
            self.assertLess(max(life['started'] for life in lifecycle), min(life['finished'] for life in lifecycle))
            run_one(args, 4, 'cifar100', '16_2')
            self.assertEqual(mtimes, [(directory / 'evaluation' / c / 'run.log').stat().st_mtime_ns for c in args.conditions])
            archive = Path(root) / 'SIMULATED_results.tar.gz'
            inventory = pack(args.output_root, archive)
            self.assertEqual(inventory['complete_tasks'], 1)
            with tarfile.open(archive) as tar:
                names = tar.getnames()
                self.assertIn('inventory.json', names)
                self.assertFalse(any(name.endswith('.pth') for name in names))
                self.assertTrue(any(name.endswith('baseline_config.json') for name in names))
                self.assertEqual(sum(name.endswith('full_per_trial.csv') for name in names), 3)
                self.assertTrue(json.load(tar.extractfile('inventory.json'))['simulated'])
                for name in names:
                    if name.endswith('.json'):
                        json.load(tar.extractfile(name))
            with self.assertRaises(FileExistsError):
                pack(args.output_root, archive)

    def test_failed_or_missing_training_never_starts_evaluation(self):
        for failure in ('train', 'missing-checkpoint'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                args = self.args(root, '--simulate-failure', failure)
                with self.assertRaises(RuntimeError):
                    run_one(args, 4, 'cifar100', '16_2')
                directory, _ = paths(args, 4, 'cifar100', '16_2')
                self.assertFalse((directory / 'evaluation').exists())
                self.assertFalse((directory / 'train_complete.json').exists())
                self.assertEqual(json.loads((directory / 'state.json').read_text())['status'], 'failed')

    def test_failed_partial_and_wrong_model_evaluations_cannot_complete(self):
        for failure in ('evaluation', 'wrong-model', 'incomplete-evaluation'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                args = self.args(root, '--simulate-failure', failure)
                with self.assertRaises(RuntimeError):
                    run_one(args, 4, 'cifar100', '16_2')
                directory, _ = paths(args, 4, 'cifar100', '16_2')
                self.assertFalse((directory / 'evaluation/max_additive/complete.json').exists())
                self.assertEqual(json.loads((directory / 'state.json').read_text())['status'], 'failed')
                if failure != 'wrong-model':
                    self.assertTrue((directory / 'evaluation/multiplicative/complete.json').exists())

    def test_pending_tasks_are_in_pack_inventory_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            create_plan(args.output_root, [4, 6], args.datasets, args.sizes, args.conditions, args.stage, True)
            run_one(args, 4, 'cifar100', '16_2')
            inventory = collect(args.output_root)
            self.assertEqual((inventory['complete_tasks'], inventory['expected_tasks']), (1, 2))
            with self.assertRaises(ValueError):
                pack(args.output_root, Path(root) / 'incomplete.tar.gz')
            pack(args.output_root, Path(root) / 'partial.tar.gz', allow_incomplete=True)
            directory, _ = paths(args, 4, 'cifar100', '16_2')
            target = directory / 'evaluation/max_additive/full_per_trial.csv'
            target.write_text(target.read_text().replace('wrn_16_2_cifar_control_r4', 'wrong_model'))
            self.assertEqual(collect(args.output_root)['complete_tasks'], 0)


if __name__ == '__main__':
    unittest.main()
