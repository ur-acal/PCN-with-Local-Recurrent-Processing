"""Check runner worker-failure exit codes without models or GPU evaluations."""
import ast
import concurrent.futures
import contextlib
import io
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import tempfile
import traceback
import unittest
from unittest.mock import Mock

from test_extension_report_lock import ROOT, RUNNERS, load_main


class ExtensionExitStatusTests(unittest.TestCase):
    def test_worker_results_determine_exit_status_and_preserve_report(self):
        for name in RUNNERS:
            for failed in (False, True):
                with self.subTest(runner=name, failed=failed):
                    tree = ast.parse((ROOT / 'scripts' / name).read_text())
                    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name == '_run_locked')
                    # Run the real scheduling/reporting tail, excluding preparation.
                    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.With))
                    run.body = ast.parse('failures = {}').body + run.body[start:]
                    future = concurrent.futures.Future()
                    if failed:
                        future.set_exception(RuntimeError('injected worker failure'))
                    else:
                        future.set_result({})
                    pool = Mock()
                    pool.submit.return_value = future
                    executor = Mock()
                    executor.__enter__ = Mock(return_value=pool)
                    executor.__exit__ = Mock(return_value=False)
                    api = SimpleNamespace(ProcessPoolExecutor=Mock(return_value=executor),
                        wait=concurrent.futures.wait, FIRST_COMPLETED=concurrent.futures.FIRST_COMPLETED)
                    publish = Mock()
                    ns = dict(futures=api, multiprocessing=multiprocessing, CONFIGS=[('case',)],
                              worker=Mock(), base=SimpleNamespace(worker=Mock()),
                              s=SimpleNamespace(case_id=lambda c: 'case'), publish=publish,
                              traceback=traceback)
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])),
                                 name, 'exec'), ns)
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = ns['_run_locked']()
                    self.assertEqual(code, int(failed))
                    state, failures = publish.call_args.args
                    self.assertEqual(state, 'finished with failures' if failed else 'complete')
                    self.assertEqual(set(failures), {'case'} if failed else set())
                    if failed:
                        self.assertIn('injected worker failure', failures['case'])
                    with tempfile.TemporaryDirectory() as tmp:
                        handler = Mock()
                        self.assertEqual(load_main(name, Path(tmp), lambda: code, handler)(), code)
                        handler.assert_not_called()  # Preserve the detailed worker report.
                    guard = next(n for n in tree.body if isinstance(n, ast.If)
                                 and '__name__' in ast.unparse(n.test))
                    with self.assertRaises(SystemExit) as exit_info:
                        exec(compile(ast.Module(body=[guard], type_ignores=[]), name, 'exec'),
                             {'__name__': '__main__', 'main': lambda: code})
                    self.assertEqual(exit_info.exception.code, code)


if __name__ == '__main__':
    unittest.main()
