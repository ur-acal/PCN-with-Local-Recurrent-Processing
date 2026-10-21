"""Persistent, barrier-ordered activation diagnostic queue; no training."""
import argparse
import csv
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {"max_additive": [0.08, 0.09, 0.10], "multiplicative": [0.30, 0.35, 0.40]}
SIZES = ("16_2", "16_4", "28_2", "28_4")
SELECTED_TOP3 = {
    ("cifar10", "max_additive"): ("r10", "r17", "finaldrop"),
    ("cifar100", "max_additive"): ("r10", "r7", "r18"),
    ("cifar10", "multiplicative"): ("finaldrop", "r14", "r7"),
    ("cifar100", "multiplicative"): ("r7", "r14", "r4"),
}
FAMILY_LABELS = {
    "finaldrop": "std_BN_wd1e-3_lr1e-1_fd0p25 (existing W2)",
    "r4": "avgSC_BN_wd1e-3_lr1e-1_fd0p25",
    "r7": "avgMain_BN_wd1e-3_lr1e-1_fd0p25",
    "r10": "std_BN_wd1e-3_lr1e-2_fd0p25_ptinit",
    "r14": "maxMain_BN_wd1e-3_lr1e-1_fd0p25",
    "r17": "avgMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit",
    "r18": "maxMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit",
}


def legacy_waves():
    groups = [(f, ("cifar10", "cifar100")) for f in ("wrn", "wd1e3", "finaldrop")]
    groups += [("pcn", (d,)) for d in ("cifar10", "cifar100")]
    return [[dict(family=f, dataset=d, size=s, condition=c, level=v,
                  id=f"{d}_{s}_{f}_{c}_{v:g}") for d in datasets for s in SIZES]
            for f, datasets in groups for c, levels in GRIDS.items() for v in levels]


def waves(parallelism=8):
    """Selected new WRNs only; finaldrop/W2 is reused from the completed study."""
    result = []
    for condition, levels in GRIDS.items():
        for level in levels:
            tasks = []
            for dataset in ("cifar10", "cifar100"):
                for family in SELECTED_TOP3[dataset, condition]:
                    if family == "finaldrop":
                        continue
                    tasks.extend(dict(family=family, dataset=dataset, size=size,
                                      condition=condition, level=level,
                                      id=f"{dataset}_{size}_{family}_{condition}_{level:g}")
                                 for size in SIZES)
            result.extend(tasks[i:i + parallelism] for i in range(0, len(tasks), parallelism))
    return result


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def command(task, args):
    return [sys.executable, "-u", "-m", "baseline.run_activation_diagnostics", "--execute",
            "--families", task["family"], "--datasets", task["dataset"], "--sizes", task["size"],
            "--condition", task["condition"], "--level", str(task["level"]),
            "--claims", "1,2", "--trials", "10", "--batch-size", "128", "--workers", "0",
            "--pcn-root", str(args.pcn_root), "--cache-root", str(args.output / "clean_cache"),
            "--legacy-control-root", str(args.legacy_control_root),
            "--pool-control-root", str(args.pool_control_root),
            "--output", str(args.output / "runs" / task["id"])]


def record(args, state):
    atomic(args.output / "queue_state.json", json.dumps(state, indent=2))
    tasks = [t for wave in state["waves"] for t in wave]
    lines = ["# Selected Top-Three WRN Activation Verification", "",
             "## Fixed selections", ""]
    for (dataset, condition), families in SELECTED_TOP3.items():
        labels = ", ".join(FAMILY_LABELS[f] for f in families)
        lines.append(f"- {dataset} / {condition}: {labels}")
    lines += ["", "`finaldrop`/W2 is reused from the completed activation study and is not queued here.",
             "Pool-after-add WRNs are measured at the residual sum before downsampling pooling.", "",
             f"Status: {state['status']}; "
             f"completed {sum(t['status'] == 'complete' for t in tasks)}/{len(tasks)} model-level tasks.",
             "WRN concurrency: 8. Both claims, full test set, 10 trials; frozen and recalibrated BN.",
             "", "| Dataset | Model family | Size | Mismatch | Levels done |", "|---|---|---|---|---|"]
    keys = dict.fromkeys((t['dataset'], t['family'], t['size'], t['condition']) for t in tasks)
    for key in keys:
        n = sum(t['status'] == 'complete' for t in tasks
                if (t['dataset'], t['family'], t['size'], t['condition']) == key)
        lines.append("| " + " | ".join(key) + f" | {n}/3 |")
    lines += ["", "## Active / Failed"]
    for t in tasks:
        if t['status'] not in ('running', 'failed'):
            continue
        files = list((args.output / 'runs' / t['id']).glob('*/per_trial.csv'))
        count = 0
        if files:
            with files[0].open() as stream:
                count = sum(1 for _ in csv.DictReader(stream))
        lines.append(f"- {t['id']}: {t['status']}, {count} metric rows written; PID {t.get('pid')}.")
    atomic(args.monitor, "\n".join(lines) + "\n")


def task_state(args, task):
    states = list((args.output / 'runs' / task['id']).glob('*/state.json'))
    if len(states) != 1:
        return None
    try:
        return json.loads(states[0].read_text()).get('status')
    except (OSError, json.JSONDecodeError):
        return None


def clear_incomplete_task(args, task):
    """Remove only artifacts that cannot be reused for an incomplete task."""
    run_dir = args.output / 'runs' / task['id']
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cache_dir = args.output / 'clean_cache' / (
        f"{task['dataset']}_{task['size']}_{task['family']}")
    if cache_dir.exists() and not (cache_dir / 'complete.json').exists():
        shutil.rmtree(cache_dir)

    log_path = args.output / 'logs' / f"{task['id']}.log"
    if log_path.exists():
        log_path.unlink()


@contextmanager
def queue_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    # Never unlink the lock: replacing its inode would allow two owners.
    with (output / 'queue.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another activation queue owns this output directory') from exc
        yield stream.fileno()


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def require_stopped(previous):
    tasks = [task for wave in previous.get('waves', []) for task in wave]
    running = [task for task in tasks if task.get('status') == 'running']
    if previous.get('hostname', socket.gethostname()) != socket.gethostname():
        if previous.get('status') == 'running' or running:
            raise RuntimeError('Cannot verify queue processes on another host; refusing recovery')
        return
    if previous.get('status') == 'running' and pid_alive(previous.get('pid')):
        raise RuntimeError('Queue controller is still alive; refusing recovery')
    if any(pid_alive(task.get('pid')) for task in running):
        raise RuntimeError('Queue workers are still alive; refusing recovery')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monitor', type=Path, required=True)
    parser.add_argument('--pcn-root', type=Path, default=ROOT.parent / 'ScAN-PCN/saved_ckpt')
    parser.add_argument('--legacy-control-root', type=Path,
                        default=ROOT / 'checkpoint/baselines_wrn_controls')
    parser.add_argument('--pool-control-root', type=Path,
                        default=ROOT / 'logs/wrn_pool_controls_slurm')
    parser.add_argument('--parallelism', type=int, default=8)
    parser.add_argument('--launch-stagger', type=float, default=0.0,
                        help='seconds between worker launches')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='resume an existing queue, preserving completed tasks')
    args = parser.parse_args()
    if args.parallelism < 1:
        parser.error('--parallelism must be positive')
    plan = waves(args.parallelism)
    assert sum(map(len, plan)) == 120
    if args.dry_run:
        print(json.dumps([[command(t, args) for t in w] for w in plan], indent=2))
        return
    with queue_lock(args.output) as lock_fd:
        run(args, plan, lock_fd)


def run(args, plan, lock_fd):
    queue_path = args.output / 'queue_state.json'
    if queue_path.exists() and not args.resume:
        raise FileExistsError('Queue already exists; refusing to overwrite')
    (args.output / 'logs').mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads(queue_path.read_text())
        require_stopped(previous)
        previous_tasks = {
            task['id']: task
            for wave in previous.get('waves', [])
            for task in wave
        }
        if set(previous_tasks) != {task['id'] for wave in plan for task in wave}:
            raise ValueError('Existing queue does not match the current selected-top3 plan')
        state = previous
        state.update(status='running', pid=os.getpid(), hostname=socket.gethostname(),
                     resumed=time.time(), waves=plan)
        for wave in plan:
            for task in wave:
                task.update(command=command(task, args))
                if task_state(args, task) == 'complete':
                    task.update(status='complete', exit_code=0)
                else:
                    clear_incomplete_task(args, task)
                    task.update(status='queued')
    else:
        state = dict(status='running', pid=os.getpid(), hostname=socket.gethostname(),
                     started=time.time(), waves=plan)
        for wave in plan:
            for t in wave:
                t.update(status='queued', command=command(t, args))
    record(args, state)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for index, wave in enumerate(plan):
        wave = [task for task in wave if task['status'] != 'complete']
        if not wave:
            continue
        print(f'Starting wave {index + 1}/{len(plan)}: '
              f'{wave[0]["condition"]} {wave[0]["level"]}, workers={len(wave)}', flush=True)
        active = []
        for worker_index, t in enumerate(wave):
            with (args.output / 'logs' / (t['id'] + '.log')).open('x') as log:
                proc = subprocess.Popen(t['command'], cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
            t.update(status='running', pid=proc.pid, started=time.time())
            active.append((t, proc))
            record(args, state)
            if args.launch_stagger > 0 and worker_index + 1 < len(wave):
                time.sleep(args.launch_stagger)
        while active:
            for t, proc in list(active):
                code = proc.poll()
                if code is None:
                    continue
                result = list((args.output / 'runs' / t['id']).glob('*/state.json'))
                complete = code == 0 and len(result) == 1 and json.loads(result[0].read_text())['status'] == 'complete'
                t.update(status='complete' if complete else 'failed', exit_code=code, ended=time.time())
                active.remove((t, proc))
            record(args, state)
            if active:
                time.sleep(5)
        if any(t['status'] == 'failed' for t in wave):
            state['status'] = 'failed'
            record(args, state)
            raise SystemExit('Wave failed; later levels have not been launched')
    state.update(status='complete', ended=time.time())
    record(args, state)


if __name__ == '__main__':
    main()
