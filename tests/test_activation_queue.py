import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baseline import run_activation_queue as queue


class QueueTests(unittest.TestCase):
    def test_resume_rejects_live_controller_or_orphan_worker_without_cleanup(self):
        for controller in (True, False):
            with self.subTest(controller=controller), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / 'results'
                output.mkdir()
                plan = queue.waves()
                for wave in plan:
                    for task in wave:
                        task['status'] = 'queued'
                plan[0][0].update(status='running', pid=None if controller else os.getpid())
                state = dict(status='running', pid=os.getpid() if controller else None, waves=plan)
                path = output / 'queue_state.json'
                path.write_text(json.dumps(state))
                sentinel = output / 'runs' / plan[0][0]['id'] / 'partial.pt'
                sentinel.parent.mkdir(parents=True)
                sentinel.write_bytes(b'keep')
                before = path.read_bytes()
                argv = ['queue', '--output', str(output), '--monitor', str(root / 'progress.md'), '--resume']
                with patch('sys.argv', argv), patch.object(queue.subprocess, 'Popen') as launch:
                    with self.assertRaisesRegex(RuntimeError, 'still alive'):
                        queue.main()
                    launch.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b'keep')
                self.assertEqual(path.read_bytes(), before)

    def test_queue_lock_rejects_second_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with queue.queue_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'Another activation queue'):
                    with queue.queue_lock(output):
                        self.fail('Concurrent queue acquired lock')
            with queue.queue_lock(output):
                pass

    def test_worker_keeps_lock_after_controller_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with queue.queue_lock(output) as fd:
                worker = subprocess.Popen([sys.executable, '-c', 'input()'],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, pass_fds=(fd,))
            try:
                with self.assertRaisesRegex(RuntimeError, 'Another activation queue'):
                    with queue.queue_lock(output):
                        self.fail('Orphaned worker lost the queue lock')
            finally:
                worker.communicate(b'release\n', timeout=10)
            self.assertEqual(worker.returncode, 0)
            with queue.queue_lock(output):
                pass

    def test_resume_completed_queue_preserves_data_and_clean_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'results'
            plan = queue.waves()
            for wave in plan:
                for task in wave:
                    task['status'] = 'complete'
                    path = output / 'runs' / task['id'] / 'model' / 'state.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'status': 'complete'}))
            queue.atomic(output / 'queue_state.json', json.dumps(dict(status='complete', waves=plan)))
            cache = output / 'clean_cache' / 'sentinel.pt'
            cache.parent.mkdir()
            cache.write_bytes(b'keep')
            argv = ['queue', '--output', str(output), '--monitor', str(root / 'progress.md'), '--resume']
            with patch('sys.argv', argv), patch.object(queue.subprocess, 'Popen') as launch:
                queue.main()
                launch.assert_not_called()
            self.assertEqual(cache.read_bytes(), b'keep')
            self.assertEqual(json.loads((output / 'queue_state.json').read_text())['status'], 'complete')

    def test_stopped_queue_resumes_only_partial_task_and_preserves_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'results'
            plan = queue.waves()
            for wave in plan:
                for task in wave:
                    task['status'] = 'complete'
                    path = output / 'runs' / task['id'] / 'model' / 'state.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'status': 'complete'}))
            partial = plan[0][0]
            partial.update(status='failed', pid=None)
            directory = output / 'runs' / partial['id']
            (directory / 'model/state.json').write_text(json.dumps({'status': 'failed'}))
            (directory / 'partial.pt').write_bytes(b'partial')
            cache = output / 'clean_cache' / f"{partial['dataset']}_{partial['size']}_{partial['family']}"
            cache.mkdir(parents=True)
            (cache / 'complete.json').write_text('{}')
            (cache / 'activation.pt').write_bytes(b'keep completed cache')
            log = output / 'logs' / (partial['id'] + '.log')
            log.parent.mkdir()
            log.write_text('old log')
            queue.atomic(output / 'queue_state.json', json.dumps(dict(status='failed', waves=plan)))
            launched = []

            class FakeProcess:
                pid = 123

                def __init__(self, cmd, **kwargs):
                    launched.append(cmd)
                    self.output = Path(cmd[cmd.index('--output') + 1])
                    self_test.assertFalse((self.output / 'partial.pt').exists())
                    self_test.assertTrue((cache / 'activation.pt').exists())

                def poll(self):
                    path = self.output / 'model/state.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'status': 'complete'}))
                    return 0

            self_test = self
            argv = ['queue', '--output', str(output), '--monitor', str(root / 'progress.md'), '--resume']
            with patch('sys.argv', argv), patch.object(queue.subprocess, 'Popen', FakeProcess):
                queue.main()
            self.assertEqual(len(launched), 1)
            self.assertEqual((cache / 'activation.pt').read_bytes(), b'keep completed cache')
            self.assertNotIn('old log', log.read_text())

    def test_plan(self):
        waves = queue.waves()
        tasks = [t for w in waves for t in w]
        self.assertEqual(len(tasks), 120)
        self.assertEqual(len({t['id'] for t in tasks}), 120)
        self.assertTrue(all(len(w) <= 8 for w in waves))
        self.assertNotIn('finaldrop', {t['family'] for t in tasks})
        tracks = {(t['dataset'], t['family'], t['size'], t['condition']) for t in tasks}
        self.assertEqual(len(tracks), 40)

    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = []
            launched = []
            class FakeProcess:
                pid = 123
                def __init__(self, cmd, **kwargs):
                    output = Path(cmd[cmd.index('--output') + 1])
                    wave = len(launched) // 8
                    self_test.assertLess(len(active), 8, 'Queue exceeded WRN parallelism')
                    active.append(self)
                    launched.append(cmd)
                    self.output, self.wave = output, wave
                def poll(self):
                    active.remove(self)
                    folder = self.output / 'model'
                    folder.mkdir(parents=True)
                    (folder / 'state.json').write_text(json.dumps({'status': 'complete'}))
                    return 1 if fail else 0
            self_test = self
            argv = ['queue', '--output', str(root / 'results'), '--monitor', str(root / 'monitor.md')]
            with patch('sys.argv', argv), patch.object(queue.subprocess, 'Popen', FakeProcess):
                if fail:
                    with self.assertRaises(SystemExit):
                        queue.main()
                else:
                    queue.main()
            state = json.loads((root / 'results/queue_state.json').read_text())
            self.assertEqual(state['status'], 'failed' if fail else 'complete')
            self.assertEqual(len(launched), 8 if fail else 120)
            self.assertTrue((root / 'monitor.md').exists())
            self.assertIn('std_BN_wd1e-3_lr1e-2_fd0p25_ptinit',
                          (root / 'monitor.md').read_text())

    def test_failure_blocks_next_wave(self):
        self.exercise(fail=True)

    def test_success(self):
        self.exercise()


if __name__ == '__main__':
    unittest.main()
