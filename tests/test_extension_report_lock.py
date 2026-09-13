"""Exercise real runner lock/error handling without loading models or GPUs."""
import ast
import fcntl
from pathlib import Path
from types import SimpleNamespace
import tempfile
import traceback
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ('extend_switch_m1_15_batches.py',
           'extend_switch_m1_15_batches_n5_n10.py',
           'extend_switch_selected_15_batches.py')


def load_main(name, out, run, publish):
    tree = ast.parse((ROOT / 'scripts' / name).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    namespace = dict(OUT=out, base=SimpleNamespace(OUT=out), fcntl=fcntl,
                     traceback=traceback, _run_locked=run, publish=publish)
    exec(compile(ast.Module(body=[main], type_ignores=[]), name, 'exec'), namespace)
    return namespace['main']


class ExtensionReportLockTests(unittest.TestCase):
    def test_duplicate_launch_does_not_publish_or_run(self):
        for name in RUNNERS:
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                report = out / 'report.md'
                report.write_text('active report')
                run = Mock()
                publish = Mock(side_effect=lambda *a: report.write_text('failed'))
                with open(out / 'supervisor.lock', 'a') as owner:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        load_main(name, out, run, publish)()
                run.assert_not_called()
                publish.assert_not_called()
                self.assertEqual(report.read_text(), 'active report')

    def test_owned_failure_publishes_before_unlock_and_reraises(self):
        for name in RUNNERS:
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                def publish(state, failures):
                    self.assertEqual(state, 'failed')
                    self.assertIn('injected failure', failures['supervisor'])
                    with open(out / 'supervisor.lock', 'a') as other:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handler = Mock(side_effect=publish)
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    load_main(name, out, Mock(side_effect=RuntimeError('injected failure')), handler)()
                handler.assert_called_once()
                with open(out / 'supervisor.lock', 'a') as other:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_success_releases_lock_without_failure_report(self):
        for name in RUNNERS:
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                run, publish = Mock(), Mock()
                load_main(name, out, run, publish)()
                run.assert_called_once()
                publish.assert_not_called()
                with open(out / 'supervisor.lock', 'a') as other:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main()
