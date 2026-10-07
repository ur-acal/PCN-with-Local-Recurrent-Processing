from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

from rhs_current_clamp import CurrentLimit, read_summary, report_rhs_current_clamp


def table(tmp_path, title='Total separate', unit='µA', low='-2', high='3'):
    path = tmp_path/'summary.md'
    path.write_text(f'# {title} summing-current distributions\n\n'
        f'| Layer | Branch | Lower 99% ({unit}) | Upper 99% ({unit}) |\n'
        '|---|---|---|---|\n'
        f'| layer_01 | FF | {low} | {high} |\n')
    return path


def check_units_and_missing_column(tmp_path):
    path = table(tmp_path)
    mode, rows, _ = read_summary(path, 99)
    assert mode == 'separate'
    assert rows['layer_01', 'FF']['lower_A'] == -2e-6
    assert rows['layer_01', 'FF']['summary_row'] == 5
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Lower 95%'):
        read_summary(path, 95)


def check_invalid_summary(tmp_path, kwargs):
    with unittest.TestCase().assertRaises(ValueError):
        read_summary(table(tmp_path, **kwargs), 99)


def test_total_current_includes_both_noise_increments():
    limit = CurrentLimit(dict(stage='z', lower_A=-2., upper_A=3.))
    block = SimpleNamespace(_stage_capacitance=lambda stage: 2.)
    state = torch.tensor([1., 1., 1.])
    rhs = torch.tensor([0., 0., 0.])
    # dt=0.5, C=2: effective currents are [-4, 0, 8] A.
    updated = limit.apply(block, state, .5, rhs,
                          torch.tensor([-1., 0., 1.]), torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(updated, torch.tensor([.5, 1., 1.75]), rtol=0, atol=0)
    assert limit.counts.tolist() == [1, 1, 3]


def test_shell_args_preserve_spaces_and_defaults():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', '-c',
        'RHS_CURRENT_SUMMARY="/a path/summary.md"; source launch_scripts/rhs_current_args.sh; '
        'printf "%s\\n" "${RHS_CURRENT_ARGS[@]}"'], cwd=root, check=True,
        capture_output=True, text=True)
    assert result.stdout.splitlines() == ['--rhs_current_summary', '/a path/summary.md',
                                          '--rhs_current_bound_percentile', '99']


class CurrentClampTests(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            check_units_and_missing_column(Path(directory))
            for kwargs in [dict(title='Deterministic separate'), dict(low='nan'),
                           dict(high='-3'), dict(unit='V')]:
                with self.subTest(kwargs=kwargs):
                    check_invalid_summary(Path(directory), kwargs)

    def test_total(self):
        test_total_current_includes_both_noise_increments()

    def test_wiring(self):
        test_shell_args_preserve_spaces_and_defaults()

    def test_scheduler_exports_to_sbatch(self):
        root = Path(__file__).resolve().parents[1]
        command = '''
RHS_CURRENT_SUMMARY="/a path/summary.md"
RHS_CURRENT_BOUND_PERCENTILE=95
RHS_CURRENT_AUDIT_PATH="/audit path/run.jsonl"
CORNER_IDS=FS_V2_T1
N_SERVERS=1
sbatch() {
  env | grep '^RHS_CURRENT_' >&2
  printf '%s\\n' "$@" >&2
  echo 12345
}
sleep() { :; }
source launch_scripts/slurm_run_mc45_toggle_ablation.sh
'''
        result = subprocess.run(['bash', '-c', command], cwd=root,
                                check=True, capture_output=True, text=True)
        self.assertIn('RHS_CURRENT_SUMMARY=/a path/summary.md', result.stderr)
        self.assertIn('RHS_CURRENT_BOUND_PERCENTILE=95', result.stderr)
        self.assertIn('RHS_CURRENT_AUDIT_PATH=/audit path/run.jsonl', result.stderr)
        self.assertIn('--export=ALL,', result.stderr)

    def test_disabled_update_and_noise_order(self):
        from ode_pc import TogglePulseFFFB
        calls = []
        block = SimpleNamespace(toggle_fast_path=True,
            enable_summing_current_noise=True, enable_coupler_noise=True,
            _brownian_increment=lambda *a: calls.append('summing') or torch.tensor([.2]),
            _coupler_brownian_increment=lambda *a: calls.append('coupler') or torch.tensor([-.1]),
            project_state=lambda x: x.clamp(-.5, .5))
        result = TogglePulseFFFB.integrate_pulse_slice(
            block, torch.tensor([.1]), .5, None, 'z', 0,
            constant_rhs=torch.tensor([.4]), active_coupler_count=torch.tensor([1.]))
        expected = ((torch.tensor([.1])+.5*torch.tensor([.4]))+torch.tensor([.2]))+torch.tensor([-.1])
        torch.testing.assert_close(result, expected.clamp(-.5, .5), rtol=0, atol=0)
        self.assertEqual(calls, ['summing', 'coupler'])

    def test_audit_identifies_case_and_serializes_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.jsonl'
            args = SimpleNamespace(rhs_current_audit_path=str(path),
                                   ablation_case_name='FS_V2_T1',
                                   model_name='model', ckpt='best')
            limit = CurrentLimit(dict(layer='layer_01', branch='FF', stage='y'))
            limit.counts = torch.tensor([1, 2, 3])
            limit.calls = 4
            report_rhs_current_clamp([limit], args, 0, 61.5)
            import json
            record = json.loads(path.read_text())
            self.assertEqual(record['case'], 'FS_V2_T1')
            self.assertEqual(record['model_name'], 'model')
            self.assertEqual(record['checkpoint'], 'best')


if __name__ == '__main__':
    unittest.main()
