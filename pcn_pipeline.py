"""Checkpoint handoff and isolated, shortened rehearsals for the PCN workers.

This module never submits jobs or starts shells. Workers call its small CLI
after activating their existing environment; the real training/eval entrypoints
use the rehearsal helpers explicitly, never through a mocked Python executable.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile

REPO = Path(__file__).resolve().parent
CONFIG_KEYS = (
    'PCN', 'DATASET_NAME', 'ARCHITECTURE', 'MODEL_INDEX', 'INP_CHANNELS',
    'OUT_CHANNELS', 'MAX_POOL', 'PCCONV', 'ODE_BLOCK', 'T_END', 'METHOD',
    'N_STEPS', 'AVG_POOLING', 'WARMUP_EPOCH', 'IS_TIMM', 'TIMM_SCHED',
    'REUSE_ACCEPTED_STEP_TRAINING', 'CONDITIONS', 'NOISY_TRIALS', 'BASE_SEED',
    'TEST_BS', 'MUL_LEVELS', 'MAX_ADD_LEVELS', 'RMS_ADD_LEVELS', 'BN_EVAL_MODE',
    'NOISE_TO_BN', 'REHEARSAL_ROOT',
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Path escapes its isolated root: {path} (root={root})')
    return path


def rehearsal_root(root):
    root = Path(root).resolve()
    if not (root / 'rehearsal.json').is_file():
        raise ValueError(f'Not an initialized rehearsal root: {root}')
    return root


def require_real_path(path):
    path = Path(path).resolve()
    if any((p / 'rehearsal.json').exists() for p in (path, *path.parents)):
        raise ValueError(f'Real execution cannot consume rehearsal artifacts: {path}')


def snapshot():
    # Metadata audit, not copies or reads of large checkpoint/cache tensors.
    result = {}
    for name in ('saved_ckpt', 'expanded_weights', 'logs'):
        base = REPO / name
        if not base.exists():
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            for entry in dirs + files:
                path = Path(directory) / entry
                stat = path.lstat()
                result[str(path.relative_to(REPO))] = [stat.st_size, stat.st_mtime_ns]
    return result


def initialize_rehearsal(root):
    root = Path(root).resolve()
    # Keep even the rehearsal inventory outside real naming/cache/skip roots.
    if root == REPO or root.is_relative_to(REPO):
        raise ValueError('Use a rehearsal directory outside this repository, e.g. /tmp/pcn-rehearsal-...')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'rehearsal.json', dict(kind='rehearsal', production_before=snapshot()))


def job_lock_path(job):
    job = Path(job).resolve()
    root = os.environ.get('REHEARSAL_ROOT')
    if root:
        contained(job, rehearsal_root(root))
    else:
        require_real_path(job)
    job.parent.mkdir(parents=True, exist_ok=True)
    return job.with_name(job.name + '.lock')


def prepare(job, stage, lock_fd=None):
    job = Path(job).resolve()
    lock_path = job_lock_path(job)
    if lock_fd is not None:
        actual, expected = os.fstat(lock_fd), lock_path.stat()
        if (actual.st_ino, actual.st_dev) != (expected.st_ino, expected.st_dev):
            raise ValueError('Job lock descriptor does not match this job')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another pipeline worker owns this job: {job}') from exc
    root = os.environ.get('REHEARSAL_ROOT')
    config = {key: os.environ.get(key, '') for key in CONFIG_KEYS}
    conditions = config['CONDITIONS'].split(',')
    if (not set(conditions) <= {'max_additive', 'multiplicative', 'rms_additive'}
            or len(conditions) != len(set(conditions))):
        raise ValueError('Choose unique max_additive, multiplicative, rms_additive conditions')
    if stage == 'test':
        previous = json.loads((job / 'config.json').read_text())
        if config != previous:
            raise ValueError('Test recipe differs from the training job configuration')
        handoff(job)
    else:
        job.mkdir(parents=True, exist_ok=False)
        write_json(job / 'config.json', config)
        sources = ('train_ode_cifar.py', 'ode_inference.py', 'pcn_pipeline.py',
                   'pc_model.py', 'ode_pc.py', 'trainer_timm.py', 'pcn_bn_evaluation.py',
                   'launch_scripts/run_rgb_ode_train.sh', 'launch_scripts/run_rgb_ode_mismatch_eval.sh',
                   'launch_scripts/run_pcn_pipeline.sh', 'launch_scripts/launch_pcn_pipeline.sh')
        write_json(job / 'source_hashes.json', {name: digest(REPO / name) for name in sources})
    for name in ('logs', 'checkpoints', 'expanded_weights', 'tmp', 'mpl', 'pycache', 'home', 'xdg'):
        (job / name).mkdir(exist_ok=True)
    write_json(job / 'state.json', dict(status='initializing', stage=stage, rehearsal=bool(root)))


def validate_training_paths(args):
    if args.rehearsal_root:
        root = rehearsal_root(args.rehearsal_root)
        contained(args.save_path, root)
        if not args.completion_json or args.model_name:
            raise ValueError('Rehearsal requires its own completion record and fresh initialization')
        contained(args.completion_json, root)
        # Rehearsals read installed CIFAR data, never download/extract datasets.
        from torchvision.datasets import CIFAR10, CIFAR100
        dataset = CIFAR100 if args.dataset == 'cifar100' else CIFAR10
        dataset(root=str(REPO.parent / 'data'), train=True, download=False)
        dataset(root=str(REPO.parent / 'data'), train=False, download=False)
    else:
        require_real_path(args.save_path)


def training_complete(args, model_name):
    import torch
    validate_training_paths(args)
    directory = Path(args.save_path).resolve() / model_name
    best = directory / f'{model_name}_best_ckpt.pth'
    last = directory / f'{model_name}_last_ckpt.pth'
    if (directory / 'training_collapse.json').exists() or not best.is_file() or not last.is_file():
        raise RuntimeError('Training did not finish with best and last checkpoints; evaluation blocked')
    state = torch.load(last, map_location='cpu', weights_only=False)
    expected_epoch = 1 if args.rehearsal_root else args.num_epochs
    if state['epoch'] != expected_epoch or state['net_type'] != args.pcn:
        raise RuntimeError('Training checkpoint epoch/model mismatch; evaluation blocked')
    record = dict(status='complete', model_name=model_name, checkpoint=str(best),
                  checkpoint_sha256=digest(best), last_checkpoint=str(last),
                  rehearsal=bool(args.rehearsal_root), args=vars(args),
                  config=json.loads((Path(args.completion_json).parent / 'config.json').read_text()))
    write_json(args.completion_json, record)
    write_json(Path(args.completion_json).parent / 'state.json',
               dict(status='trained', rehearsal=bool(args.rehearsal_root), checkpoint=str(best)))


def handoff(job, write_env=True):
    job = Path(job).resolve()
    record = json.loads((job / 'train_complete.json').read_text())
    config = json.loads((job / 'config.json').read_text())
    path = contained(record['checkpoint'], job / 'checkpoints')
    expected = path.parent / f"{record['model_name']}_best_ckpt.pth"
    if (record['status'] != 'complete' or path != expected or path.parent.name != record['model_name']
            or record['config'] != config or record['rehearsal'] != bool(config['REHEARSAL_ROOT'])
            or digest(path) != record['checkpoint_sha256']):
        raise ValueError('Invalid checkpoint handoff or changed recipe')
    if record['rehearsal']:
        contained(path, rehearsal_root(config['REHEARSAL_ROOT']))
    env = dict(MODEL_NAME=record['model_name'], MODEL_DIR=str(job / 'checkpoints'))
    if write_env:
        (job / 'handoff.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k, v in env.items()))
    return record


def short_loader(loader, batch_size):
    from torch.utils.data import DataLoader, Subset
    count = min(batch_size, len(loader.dataset))
    return DataLoader(Subset(loader.dataset, range(count)), batch_size=count,
                      shuffle=False, num_workers=0, drop_last=False)


def rehearse_training(trainer, args):
    import math
    validate_training_paths(args)
    job = Path(args.completion_json).parent
    write_json(job / 'training_args.json', vars(args))
    # Preserve the actual timm transforms, Mixup, optimizer and 300-epoch
    # scheduler/warmup setup. Only shorten the dataset and execute one epoch.
    trainer.train_dataloader = short_loader(trainer.train_dataloader, args.batch_size)
    trainer.val_dataloader = short_loader(trainer.val_dataloader, args.batch_size)
    loss = trainer.train_one_epoch(0)
    accuracy, _, _, _ = trainer.evaluate(trainer.val_dataloader)
    if not math.isfinite(float(loss)):
        raise RuntimeError('Rehearsal training produced nonfinite loss')
    checkpoint = trainer._save_model_ckpt(accuracy, 1, '_best_ckpt.pth')
    trainer._save_model_ckpt(accuracy, 1, '_last_ckpt.pth')
    write_json(job / 'short_training.json', dict(train_batches=1, validation_batches=1,
               loss=float(loss), accuracy=float(accuracy), recipe_epochs=args.num_epochs,
               dependency_completion_simulated=True))
    print(f'----- Train finished, Model Name: {trainer.model_name} -----')
    print(f'----- Model path: {checkpoint} -----')


def rehearse_evaluation(args):
    import torch
    from types import SimpleNamespace
    from inference_utils import get_test_data, load_and_prepare_model
    from ode_inference import get_t_end, seed_mismatch_trial
    from ode_pc import ODEBLOCK_CLASSES
    from pc_model import PC_CONV_CLASS
    from pcn_bn_evaluation import evaluate_pair
    from baseline.run_baseline import BNRecalibrationConfig, build_bn_calibration_loader
    from baseline.run_wrn_bn_recalibration_experiment import evaluate_with_predictions
    from torchvision.datasets import CIFAR10, CIFAR100

    root = rehearsal_root(args.rehearsal_root)
    checkpoint = contained(Path(args.model_dir) / args.model_name / f'{args.model_name}_best_ckpt.pth', root)
    output = contained(Path(args.output_pickle).parent, root)
    contained(args.expanded_w_dir, root)
    if args.test_expanded or args.thermal_noise or args.noise_to_bn or args.noise_to_conv_bias:
        raise ValueError('Rehearsals require the same unexpanded FP weight-only mismatch policy')
    job = Path(args.model_dir).parent
    record = handoff(job)
    if checkpoint != Path(record['checkpoint']):
        raise ValueError('Evaluation did not load its training dependency')
    expected = record['args']
    if expected.get('pc_conv', 'PCConv') != 'PCConv' or args.pc_conv != 'PCConvNoisy':
        raise ValueError('Evaluation activation differs from its training dependency')
    if (args.method != expected['method'] or args.n_steps != expected['n_steps']
            or args.ode_block != expected['ode_block'] or args.task != expected['dataset']):
        raise ValueError('Evaluation solver/dynamics/dataset differs from its dependency')
    write_json(output / 'evaluation_args.json', vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dataset = CIFAR100 if args.task == 'cifar100' else CIFAR10
    for train in (True, False):
        dataset(root=args.data_dir or '../data', train=train, download=False)
    loader = short_loader(get_test_data(args.test_bs, task=args.task, data_root=args.data_dir), args.test_bs)
    levels = [float(x) for x in args.noise_level_list.split(',')]
    level = max(levels)
    ode_params = dict(ode_block=ODEBLOCK_CLASSES[args.ode_block], t_end=get_t_end(args),
                      method=args.method, tol=args.tol, n_steps=args.n_steps, ts_scale=args.ts_scale,
                      mismatch_type=args.mismatch_type, additive_scale_mode=args.additive_scale_mode)
    with torch.no_grad():
        paired = args.bn_eval_mode == 'paired'
        if not paired:
            trial_seed = seed_mismatch_trial(args.seed, args.model_index, level, 0)
        # Match the established paired evaluator: load clean, then perturb
        # explicitly. The old loader's nonzero ODE path always noises BN.
        model = load_and_prepare_model(str(checkpoint), device=device, noise_level=0.0 if paired else level,
                    pc_conv_layer=PC_CONV_CLASS[args.pc_conv], noise_to_bn=False,
                    noise_to_linear=True, noise_to_conv_bias=False, fuse_bn=False, ode_params=ode_params)
        if paired:
            clean_bn = {n: v.clone() for n, v in model.state_dict().items()
                        if n.startswith(('BNs.', 'BNend.'))}
            trial_seed = seed_mismatch_trial(args.seed, args.model_index, level, 0)
            model.noise_level = level
            for block in model.PcConvs:
                block.noise_level = level
            if level > 0:
                model.add_noise(noise_to_bn=False, noise_to_linear=True, noise_to_conv_bias=False,
                                mismatch_type=args.mismatch_type, additive_scale_mode=args.additive_scale_mode)
            if not all(torch.equal(v, model.state_dict()[n]) for n, v in clean_bn.items()):
                raise ValueError('Rehearsal perturbed error-free BN parameters/buffers')
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      model_class=type(model).__name__, pooling=type(model.max_pool2d).__name__,
                      level=level, seed=trial_seed, base_seed=args.seed, test_batches=1, test_expanded=args.test_expanded,
                      stop_reason='before_optional_expanded_weight_cache_generation')
        if args.bn_eval_mode == 'paired':
            cfg = BNRecalibrationConfig(enabled=True, num_samples=str(args.test_bs),
                batch_size=args.test_bs, subset_seed=args.calibration_subset_seed, num_workers=0,
                diagnostics_dir=str(output / 'bn_diagnostics'))
            calibration = build_bn_calibration_loader(model,
                SimpleNamespace(dataset=args.task, data_dir=args.data_dir or '../data', pin_memory=False),
                {'timm_input_size': (3, 32, 32)}, cfg)
            frozen, recal, _ = evaluate_pair(model, loader, calibration, device, cfg, args.model_name, level, 0)
            result.update(frozen_accuracy=frozen, recalibrated_accuracy=recal)
        else:
            result['accuracy'] = evaluate_with_predictions(model, loader, device)[0]
        if result['model_class'] != record['args']['pcn']:
            raise ValueError('Checkpoint model class differs from training')
        expected_pool = 'AvgPool2d' if expected['avg_pooling'] else 'MaxPool2d'
        if result['pooling'] != expected_pool:
            raise ValueError('Checkpoint pooling differs from training')
        write_json(output / 'rehearsal_initialization.json', result)
    # Deliberately no result.pkl, paired_complete.json, or real eval completion.


def finish(job, record_status=True):
    import pickle
    import math
    job = Path(job).resolve()
    record = handoff(job, write_env=False)
    config = record['config']
    if record['rehearsal']:
        # Rehearsal checkpoints must never be accepted as real dependencies.
        contained(record['checkpoint'], rehearsal_root(config['REHEARSAL_ROOT']))
    results = job / 'results' / config['DATASET_NAME'] / config['ARCHITECTURE']
    for condition in config['CONDITIONS'].split(','):
        directory = results / condition
        if record['rehearsal']:
            value = json.loads((directory / 'rehearsal_initialization.json').read_text())
            if value['checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Rehearsal checkpoint handoff mismatch')
            if list(directory.rglob('result.pkl')) or (directory / 'paired_complete.json').exists():
                raise ValueError('Rehearsal created a production evaluation marker')
        else:
            modes = ('frozen', 'recalibrated') if config['BN_EVAL_MODE'] == 'paired' else ('',)
            for mode in modes:
                with (directory / mode / 'result.pkl').open('rb') as stream:
                    value = pickle.load(stream)
                trials = next(iter(value.values()))['noise_acc_spec']['Johnson']
                key = {'multiplicative': 'MUL_LEVELS', 'max_additive': 'MAX_ADD_LEVELS', 'rms_additive': 'RMS_ADD_LEVELS'}[condition]
                levels = [float(x) for x in config[key].split(',')]
                if set(trials) != set(levels) or any(
                    len(trials[level]) != (1 if level == 0 else int(config['NOISY_TRIALS']))
                    or not all(math.isfinite(float(x)) and 0 <= float(x) <= 100 for x in trials[level])
                    for level in levels
                ):
                    raise ValueError(f'Incomplete/invalid evaluation: {directory}')
            if config['BN_EVAL_MODE'] == 'paired':
                paired = json.loads((directory / 'paired_complete.json').read_text())
                if paired['checkpoint_sha256'] != record['checkpoint_sha256']:
                    raise ValueError('Paired evaluation used another checkpoint')
    status = 'rehearsed' if record['rehearsal'] else 'complete'
    if record_status:
        write_json(job / 'state.json', dict(status=status, checkpoint=record['checkpoint']))
    return record


def audit(root):
    root = rehearsal_root(root)
    before = json.loads((root / 'rehearsal.json').read_text())['production_before']
    after = snapshot()
    changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    configs = list(root.rglob('config.json'))
    validate_inventory(root, configs)
    for path in root.rglob('*'):
        contained(path, root)
    for path in configs:
        job = path.parent
        if json.loads((job / 'state.json').read_text())['status'] != 'rehearsed':
            raise ValueError(f'Incomplete rehearsal: {job}')
        if list((job / 'expanded_weights').rglob('*')):
            raise ValueError('FP rehearsal unexpectedly created expanded weights')
    markers = list(root.rglob('result.pkl')) + list(root.rglob('paired_complete.json'))
    if markers:
        raise ValueError('Rehearsal contains production evaluation markers')
    report = dict(jobs=len(configs), production_changes=changes, isolated=not changes,
                  expanded_weight_files=0, production_completion_markers=len(markers))
    write_json(root / 'isolation_audit.json', report)
    if changes:
        raise ValueError(f'Production artifacts changed during rehearsal: {changes[:5]}')
    print(json.dumps(report))


def validate_inventory(root, configs):
    plan = Path(root) / 'planned_jobs.tsv'
    if plan.exists():
        expected = {tuple(line.split('\t')) for line in plan.read_text().splitlines()}
        actual = set()
        for path in configs:
            config = json.loads(path.read_text())
            actual.add((config['PCN'], config['DATASET_NAME'], config['ARCHITECTURE'].removeprefix('WRN_')))
        if actual != expected or len(configs) != len(expected):
            raise ValueError('Job inventory is incomplete or contains unexpected/duplicate jobs')


def pack(root, archive, checkpoints):
    root = Path(root).resolve()
    configs = list(root.rglob('config.json'))
    if not configs:
        raise ValueError('No pipeline jobs found')
    validate_inventory(root, configs)
    files = [] if checkpoints else [p for p in (root / 'planned_jobs.tsv', root / 'submissions.tsv') if p.exists()]
    for path in configs:
        job = path.parent
        state = json.loads((job / 'state.json').read_text())
        if state['status'] != 'complete':
            raise ValueError(f'Only completed real runs can be packed: {job}')
        record = finish(job, record_status=False)
        if checkpoints:
            files.append(Path(record['checkpoint']))
        else:
            files.extend(p for p in job.rglob('*') if p.is_file()
                         and p.suffix in {'.json', '.csv', '.pkl', '.log'}
                         and not any(x in p.relative_to(job).parts for x in
                                     ('checkpoints', 'tmp', 'pycache', 'home', 'mpl', 'xdg', 'expanded_weights')))
    with tarfile.open(archive, 'x' if checkpoints else 'x:gz') as bundle:
        for path in files:
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(f'{len(files)} files: {archive}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init-rehearsal', 'lock-path', 'prepare', 'handoff', 'finish', 'fail', 'audit', 'pack'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', default='train-test', choices=['train', 'test', 'train-test'])
    parser.add_argument('--archive')
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.action == 'init-rehearsal':
        initialize_rehearsal(args.root)
    elif args.action == 'lock-path':
        print(job_lock_path(args.root))
    elif args.action == 'prepare':
        if args.lock_fd is None:
            parser.error('--lock-fd is required for prepare; use the pipeline worker')
        prepare(args.root, args.stage, args.lock_fd)
    elif args.action == 'handoff':
        handoff(args.root)
    elif args.action == 'finish':
        finish(args.root)
    elif args.action == 'fail':
        if Path(args.root, 'config.json').is_file():
            write_json(Path(args.root) / 'state.json', dict(status='failed'))
    elif args.action == 'audit':
        audit(args.root)
    elif args.action == 'pack':
        if not args.archive:
            parser.error('--archive is required for pack')
        pack(args.root, args.archive, args.checkpoints)


if __name__ == '__main__':
    main()
