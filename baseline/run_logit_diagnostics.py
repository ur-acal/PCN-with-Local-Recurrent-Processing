"""Replay C2 only against existing clean caches; preserve all original evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil

from .generate_activation_claims_summary import EVIDENCE
from .run_activation_queue import atomic, legacy_waves

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'logs/activation_logit_diagnostics'


def check_replay(metrics, transition, predictions, original):
    old = json.loads((original / 'summary.json').read_text())
    for key in ('samples', 'clean_correct', 'noisy_correct', 'correct_to_wrong', 'wrong_to_correct'):
        if transition[key] != old[key]:
            raise ValueError(f'Replay classification differs: {original}: {key}')
    with (original / 'predictions.csv').open() as stream:
        expected = [tuple(int(r[k]) for k in ('sample_id', 'label', 'clean_prediction', 'noisy_prediction'))
                    for r in csv.DictReader(stream)]
    if predictions != expected:
        raise ValueError(f'Replay predictions differ: {original}')
    with (original / 'metrics.csv').open() as stream:
        old_metric = next(r for r in csv.DictReader(stream) if r['layer'] == 'logits')
    for key in ('error_energy', 'reference_energy'):
        if not math.isclose(metrics[key], float(old_metric[key]), rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f'Replay logit energy differs: {original}: {key}')


def replay(args):
    if args.pcn_full:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from .activation_diagnostics import (evaluate_cached, make_helper, selected_parameters,
                                         run_diagnostics, summarize, consolidate_clean_logits)
    from .run_activation_diagnostics import load_model, sha256
    from .run_baseline import (BNRecalibrationConfig, build_bn_calibration_loader,
                               recalibrate_batchnorm_statistics)

    source = args.source / 'runs' / args.task
    manifest = json.loads((source / 'manifest.json').read_text())
    spec = manifest['models'][0]
    if args.pcn_full and spec['family'] != 'pcn':
        raise ValueError('Full deterministic replacement is restricted to PCN')
    original = source / spec['id']
    model_info = json.loads((original / 'model.json').read_text())
    if json.loads((original / 'state.json').read_text()) != dict(status='complete', smoke=False):
        raise ValueError('Original run incomplete or smoke')
    config = manifest['arguments']
    if config['trials'] != 10 or config['batch_size'] != 128 or manifest['smoke']:
        raise ValueError('Unexpected original protocol')
    condition, level = config['condition'], config['level']
    output = args.output / 'runs' / args.task
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        if not json.loads((output / 'complete.json').read_text()).get('logits_saved'):
            raise ValueError('Older run has no saved logits; preserve it and choose a new output directory')
        return
    if args.pcn_full:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if sha256(spec['checkpoint']) != model_info['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    cache = args.source / 'clean_cache' / spec['id']
    metadata = json.loads((cache / 'complete.json').read_text())
    signature = metadata['signature']
    if (signature['checkpoint_sha256'] != model_info['checkpoint_sha256'] or
            signature['dataset'] != spec['dataset'] or signature['samples'] != 10000 or
            signature['batch_size'] != 128 or signature['torch_version'] != str(torch.__version__)):
        raise ValueError('Clean cache provenance differs')
    # Metrics code changed; model, solver, and perturbation implementation must not.
    for name in ('baseline/run_baseline.py', 'pc_model.py', 'ode_pc.py', 'mismatch_utils.py'):
        if sha256(ROOT / name) != signature['implementation_hashes'][name]:
            raise ValueError(f'Model/mismatch implementation changed: {name}')
    paths = sorted(cache.glob('batch_*/samples.pt'))
    if len(paths) != metadata['batches']:
        raise ValueError('Clean cache incomplete')
    model, kind, cfg = load_model(spec, device)
    if json.loads(json.dumps(cfg, default=str)) != model_info['cfg']:
        raise ValueError('Model configuration changed')
    selected = selected_parameters(model)
    if sorted(selected) != model_info['selected_parameters']:
        raise ValueError('Mismatch parameter selection changed')
    helper = make_helper(model, selected, condition)
    loader_args = argparse.Namespace(dataset=spec['dataset'], data_dir=signature['data_root'],
                                    batch_size=128, num_workers=0, pin_memory=True)
    recal_cfg = BNRecalibrationConfig(enabled=True, num_samples='5120', batch_size=128,
        subset_seed=20240618, num_workers=0, diagnostics_dir=str(output / 'bn_diagnostics'))
    calibration = build_bn_calibration_loader(model, loader_args, cfg, recal_cfg) if kind == 'wrn' else None
    policies = ('frozen', 'recalibrated') if kind == 'wrn' else ('no_bn',)
    identity = dict(spec=spec, checkpoint_sha256=model_info['checkpoint_sha256'],
                    original=str(original), condition=condition, level=level,
                    pcn_full=args.pcn_full,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    implementation_hashes={p: sha256(ROOT / p) for p in
                        ('baseline/activation_diagnostics.py', 'baseline/run_logit_diagnostics.py')})
    identity_path = output / 'provenance.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Supplemental run provenance changed; use a new output directory')
    atomic(identity_path, json.dumps(identity, indent=2))
    if args.pcn_full:
        # Same noise realizations, but all C2 quantities are replaced together;
        # original nondeterministic outputs are not treated as an exact target.
        for trial in range(10):
            p = original / f'claim2/network/{condition}/level_{level:g}/trial_{trial}/perturbation.json'
            old = json.loads(p.read_text())
            seed = config['base_seed'] + 100000 * spec['model_index'] + 1000 * round(level * 1e6) + trial
            if old['selected_parameters'] != sorted(selected) or old['seed'] != seed:
                raise ValueError('Original perturbation provenance inconsistent')
        full_output = output / spec['id']
        full_output.mkdir(exist_ok=False)
        atomic(full_output / 'model.json', json.dumps(model_info, indent=2))
        atomic(full_output / 'state.json', json.dumps(dict(status='running', smoke=False)))
        try:
            rows = run_diagnostics(model, kind, None, None, None, full_output, device,
                claims=(2,), levels={condition: [level]}, trials=10,
                base_seed=config['base_seed'], model_index=spec['model_index'],
                cache_directory=cache, cache_signature=signature,
                clean_logits_path=args.output / 'clean_logits' / (spec['id'] + '.pt'))
            summarize(rows, full_output)
            changes = []
            for trial in range(10):
                suffix = f'claim2/network/{condition}/level_{level:g}/trial_{trial}/no_bn'
                old = json.loads((original / suffix / 'summary.json').read_text())
                new = json.loads((full_output / suffix / 'summary.json').read_text())
                saved = torch.load(full_output / suffix / 'noisy_logits.pt', weights_only=True)
                if saved['logits'].shape != (10000, 100 if spec['dataset'] == 'cifar100' else 10):
                    raise ValueError('Incomplete saved logits')
                if int((saved['logits'].argmax(1) == saved['labels']).sum()) != new['noisy_correct']:
                    raise ValueError('Saved logits disagree with metrics')
                changes.append(dict(trial=trial, original_accuracy=old['noisy_accuracy'],
                                    replacement_accuracy=new['noisy_accuracy'],
                                    delta_pp=new['noisy_accuracy'] - old['noisy_accuracy']))
            atomic(output / 'old_vs_new.json', json.dumps(changes, indent=2))
            atomic(full_output / 'state.json', json.dumps(dict(status='complete', smoke=False)))
        except Exception as exc:
            atomic(full_output / 'state.json', json.dumps(dict(status='failed', error=repr(exc))))
            raise
        atomic(output / 'complete.json', json.dumps(dict(status='complete', trials=10, policies=policies,
                                                        logits_saved=True, pcn_full=True)))
        return
    consolidate_clean_logits(paths, args.output / 'clean_logits' / (spec['id'] + '.pt'))
    for trial in range(10):
        trial_source = original / f'claim2/network/{condition}/level_{level:g}/trial_{trial}'
        perturbation = json.loads((trial_source / 'perturbation.json').read_text())
        expected_seed = config['base_seed'] + 100000 * spec['model_index'] + 1000 * round(level * 1e6) + trial
        if (perturbation['selected_parameters'] != sorted(selected) or
                perturbation['seed'] != expected_seed):
            raise ValueError('Original perturbation provenance inconsistent')
        helper.restore_clean_state()
        model.eval()
        helper.noise_sigma, helper.seed = level, expected_seed
        helper.add_noise()
        for policy in policies:
            destination = output / f'trial_{trial}' / policy / 'summary.json'
            if destination.exists():
                continue
            if policy == 'recalibrated':
                recalibrate_batchnorm_statistics(model, calibration, device, recal_cfg,
                    f'claim2_network_{condition}', level, trial)
            metrics, transition, predictions = evaluate_cached(model, kind, paths, device, logits_only=True,
                                                                logits_path=destination.parent / 'noisy_logits.pt')
            check_replay(metrics['logits'].result(), transition, predictions, trial_source / policy)
            transition.update(original_logits=metrics['logits'].result(), seed=helper.seed,
                              trial=trial, bn_policy=policy, replay_validated=True)
            atomic(destination, json.dumps(transition, indent=2))
            print(f'{args.task} trial={trial} BN={policy}: validated and saved', flush=True)
    atomic(output / 'complete.json', json.dumps(dict(status='complete', trials=10, policies=policies,
                                                    logits_saved=True)))


def queue(args):
    from .generate_activation_claims_summary import generate
    plan = legacy_waves()
    if args.pcn_full:
        plan = [wave for wave in plan if wave[0]['family'] == 'pcn']
    total = sum(map(len, plan))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    state = {t['id']: 'queued' for wave in plan for t in wave}
    args.output.mkdir(parents=True, exist_ok=True)
    backup = args.output / 'report_before_logit_diagnostics.md'
    if args.report.exists() and not backup.exists():
        shutil.copy2(args.report, backup)

    def record():
        atomic(args.output / 'queue_state.json', json.dumps(state, indent=2))
        lines = ['# Claim 2 Logit Diagnostics', '',
                 f"Complete: {list(state.values()).count('complete')}/{total}; WRN 8-way, PCN 4-way.",
                 'Original results and clean cache are preserved. Full PCN C2 replacement is deterministic; '
                 'old/new accuracy differences are recorded, not rejected.' if args.pcn_full else
                 'Original Claim 1/2 results are preserved. Replays require original predictions to match.', '']
        lines += [f'- {k}: {v}' for k, v in state.items() if v not in ('queued', 'complete')]
        atomic(args.output / 'progress.md', '\n'.join(lines) + '\n')

    def launch(task):
        name = task['id']
        command = [sys.executable, '-u', '-m', 'baseline.run_logit_diagnostics',
                   '--source', str(args.source), '--output', str(args.output), '--task', name]
        if args.pcn_full:
            command.append('--pcn-full')
        with (args.output / (name + '.log')).open('a') as stream:
            return subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)

    record()
    for wave in plan:
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {}
            for task in wave:
                state[task['id']] = 'running'
                futures[pool.submit(launch, task)] = task['id']
            record()
            for future in as_completed(futures):
                name = futures[future]
                state[name] = 'complete' if future.result() == 0 else 'failed'
                record()
        if 'failed' in state.values():
            raise RuntimeError('Replay failed; report not regenerated. Inspect task logs.')
    generate(args.source, args.report,
             logit_source=args.wrn_logit_source if args.pcn_full else args.output,
             pcn_c2_source=args.output if args.pcn_full else None)
    atomic(args.output / 'report_complete.json', json.dumps(dict(report=str(args.report))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=EVIDENCE / 'activation_claims_20260912')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path, default=EVIDENCE / 'claim1_claim2_comparison.md')
    parser.add_argument('--task', help='One original model-level task; otherwise run the persistent queue')
    parser.add_argument('--pcn-full', action='store_true',
                        help='Deterministic full C2 replacement for all PCNs, reusing original clean cache and C1')
    parser.add_argument('--wrn-logit-source', type=Path, default=DEFAULT_OUTPUT,
                        help='Already validated WRN logit metrics for report generation after PCN replacement')
    args = parser.parse_args()
    if args.pcn_full and args.output == DEFAULT_OUTPUT:
        args.output = ROOT / 'logs/activation_pcn_c2_deterministic'
    if args.task:
        replay(args)
    else:
        queue(args)


if __name__ == '__main__':
    main()
