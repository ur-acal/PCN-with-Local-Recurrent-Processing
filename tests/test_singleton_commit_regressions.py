"""Regression checks for evaluator coordinates and diagnostic queue failures."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SingleCurveRegressions(unittest.TestCase):
    def test_evaluator_head_matches_convolution_coordinates(self):
        import torch
        from baseline.cifar_resnet import CIFARResNet
        from baseline.evaluate_physical_feedforward_cifar import parse_args
        from final_linear import (
            configure_feedforward_head, config_from_args, select_model_head)
        from physical_feedforward import convert_wide_resnet_to_physical

        source = ROOT / 'baseline/evaluate_physical_feedforward_cifar.py'
        calls = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'configure_feedforward_head']
        self.assertEqual(len(calls), 1)
        code = compile(ast.Module(body=calls, type_ignores=[]), str(source), 'exec')
        for physical in (False, True):
            for kind in ('digital', 'analog'):
                with self.subTest(physical=physical, head=kind):
                    with patch('sys.argv', ['eval', '--model_name', 'resnet110_cifar',
                                           '--checkpoint', 'unused', '--physical',
                                           str(physical).lower(), '--final_head_type', kind]):
                        args = parse_args()
                    model = CIFARResNet(depth=8, base_width=2, num_classes=3, in_chans=4)
                    select_model_head(model, kind, config_from_args(args))
                    convert_wide_resnet_to_physical(
                        model, physical=physical, physical_level=2,
                        v_dd=args.v_dd, one_over_q=args.one_over_q)
                    exec(code, dict(model=model, args=args,
                                    configure_feedforward_head=configure_feedforward_head))
                    self.assertEqual(model.fc.physical, physical)
                    with torch.no_grad():
                        output = model.fc(torch.zeros(2, model.fc.in_features))
                    self.assertTrue(torch.isfinite(output).all())

    def _queue_fixture(self, root):
        logs = root / 'logs/local_runs/tt_singleton_50ep'
        logs.mkdir(parents=True)
        launchers = root / 'launch_scripts'
        launchers.mkdir()
        runner = launchers / 'run_cifair100_tt_singleton_ft_eval.sh'
        runner.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ -f "${TT_SINGLETON}" && "${STAGE}" == eval ]]
expected="tt_singleton_50ep_CiFAIR100_${MODEL_KIND}_${COUPLER_FAMILY}"
if [[ "${MODEL_KIND}" == wrn_c27 ]]; then
  expected+="_fixedTiming15ns_vdd0p1_qinv5"
fi
[[ "${RUN_TAG}" == "${expected}" ]]
printf 'CHECKED %s %s %s %s\\n' "${MODEL_KIND}" "${COUPLER_FAMILY}" "${TT_SINGLETON}" "${RUN_TAG}"
if [[ "${MOCK_FAIL:-false}" == true ]]; then exit 7; fi
''')
        env = dict(os.environ, REPO_ROOT=str(root))
        for family, mc in [('coupler_monte', 81), ('coupler_monte_v2', 59)]:
            csv = root / f'{family}_mc{mc}.csv'
            csv.write_text('x,y\n0,0.00002\n')
            env['COUPLER_MONTE_TT_SINGLETON' if family == 'coupler_monte'
                else 'COUPLER_MONTE_V2_TT_SINGLETON'] = str(csv)
            (logs / f'wrn_c27_{family}_fixed15ns_vdd0p1_qinv5.pid').write_text('999999999')
            (logs / f'wrn_c27_{family}_fixed15ns_vdd0p1_qinv5.ft.log').write_text(
                '----- Train finished -----\n')
            name = 'custom_noresize_cifar100_wrn_flexible_cifar_avgpool_shortcut'
            directory = root / ('saved_ckpt_runs/tt_singleton_50ep_CiFAIR100_wrn_c27_'
                                f'{family}_fixedTiming15ns_vdd0p1_qinv5_ft_iq12/cifar100/'
                                f'custom_noresize/wrn_flexible_cifar_avgpool_shortcut/{name}')
            directory.mkdir(parents=True)
            for suffix in ('_full_param_best_ckpt.pth', '_last_ckpt.pth'):
                (directory / (name + suffix)).touch()
        return env

    def test_legacy_queue_forwards_curves_and_validated_namespaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = self._queue_fixture(Path(temporary))
            result = subprocess.run(
                ['bash', str(ROOT / 'launch_scripts/queue_cifair100_tt_singleton_mc45.sh')],
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('CHECKED '), 4)
            self.assertIn('ALL_FOUR_MC45_COMPLETE', result.stdout)

    def test_legacy_queue_propagates_child_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = self._queue_fixture(Path(temporary))
            env['MOCK_FAIL'] = 'true'
            result = subprocess.run(
                ['bash', str(ROOT / 'launch_scripts/queue_cifair100_tt_singleton_mc45.sh')],
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertNotIn('ALL_FOUR_MC45_COMPLETE', result.stdout)

    def test_noise_scan_collects_every_child_status(self):
        for failing in ('2', ''):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint = root / 'source.pth'
                checkpoint.touch()
                python = root / 'mock-python'
                python.write_text('''#!/usr/bin/env bash
set -euo pipefail
while (( $# )); do
  if [[ "$1" == --one_over_q ]]; then q="$2"; break; fi
  shift
done
if [[ "$q" == "${MOCK_FAIL_Q}" ]]; then exit 9; fi
printf '%s\\n' "$q" > "${OUTPUT_ROOT}/finished_${q}"
''')
                python.chmod(0o755)
                env = dict(os.environ, REPO_ROOT=str(root), PYTHON=str(python),
                           MODEL_CKPT=str(checkpoint), OUTPUT_ROOT=str(root / 'outputs'),
                           LOG_ROOT=str(root / 'logs'), PARALLEL_JOBS='2',
                           ONE_OVER_Q_VALUES='1 2 3', MOCK_FAIL_Q=failing)
                result = subprocess.run(
                    ['bash', str(ROOT / 'launch_scripts/run_wrn_c27_one_over_q_noise_scan.sh')],
                    env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 1 if failing else 0,
                                 result.stdout + result.stderr)
                self.assertEqual('ALL COMPLETE' in result.stdout, not bool(failing))
                for q in ('1', '3'):
                    self.assertTrue((root / f'outputs/finished_{q}').exists())


if __name__ == '__main__':
    unittest.main()
