import json
import tempfile
import unittest
from unittest.mock import patch

from baseline.run_wrn_controls import parse_args, paths, run_one, source_provenance


class GitProvenanceTests(unittest.TestCase):
    def test_missing_git_does_not_block_train_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--rows', '2', '--datasets', 'cifar10', '--sizes', '16_2',
                               '--stage', 'train-test', '--simulate', '--output-root', tmp,
                               '--conditions', 'max_additive,multiplicative,rms_additive'])
            with patch('baseline.run_wrn_controls.subprocess.check_output',
                       side_effect=FileNotFoundError(2, 'No such file or directory', 'git')):
                result = run_one(args, 2, 'cifar10', '16_2')
            self.assertEqual(result['status'], 'complete')
            directory, _ = paths(args, 2, 'cifar10', '16_2')
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertIsNone(manifest['git_commit'])
            self.assertIn('git', manifest['git_error'])
            self.assertIn('baseline/run_wrn_controls.py', manifest['source_sha256'])

    def test_available_git(self):
        with patch('baseline.run_wrn_controls.subprocess.check_output', side_effect=['abc\n', '']):
            metadata = source_provenance()
        self.assertEqual(metadata['git_commit'], 'abc')
        self.assertEqual(metadata['git_status'], '')
        self.assertNotIn('git_error', metadata)


if __name__ == '__main__':
    unittest.main()
