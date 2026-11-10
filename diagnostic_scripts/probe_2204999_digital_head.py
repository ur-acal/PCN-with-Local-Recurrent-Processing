"""No-update, four-batch head/q probe using the downloaded WRN recipes.

Uses production loading/conversion and hardware sampling. Training-mode probes
use batch BN and dropout, but never step an optimizer or save model state.
Results are isolated from real training/evaluation outputs and caches.
"""
import argparse
import copy
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from baseline import evaluate_physical_feedforward_cifar as evaluator
from diagnostic_scripts.select_feedforward_one_over_q import args_from_ft_command


def recipe(job):
    text = (ROOT / f'logs/slurm_jobs/slurm_{job}.out').read_text()
    commands = [shlex.split(line.removeprefix('Running: '))[2:]
                for line in text.splitlines()
                if line.startswith('Running: python baseline/train_baseline_cifar.py')
                and '--resume_checkpoint' in line]
    args = args_from_ft_command(commands[-1])
    args.max_eval_batches = 4
    args.selection_samples = 512
    args.checkpoint = str(ROOT / args.checkpoint)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all')
    parser.add_argument('--q-values', nargs='+', type=float,
                        help='Coarse digital-head q sweep; defaults retain the original diagnostic cases.')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results/WRN_C10_2204999_audit')
    options = parser.parse_args()
    if options.q_values and any(not (0 < q < float('inf')) for q in options.q_values):
        parser.error('--q-values must be positive finite values')
    out = options.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    digital, analog = recipe(2204999), recipe(2196952)
    cases = [
        ('digital_q2', digital, {}),
        ('digital_q2_adc_off', digital, {'final_adc_noise_lsb': 0.}),
        ('digital_q2_head_quant_off', digital,
         {'final_adc_noise_lsb': 0., 'final_head_quantize': False}),
        ('digital_q32', digital, {'one_over_q': 32.}),
        ('analog_q32', analog, {}),
        ('analog_q2', analog, {'one_over_q': 2.}),
    ]
    if options.q_values:
        cases = [(f'digital_q{q:g}', digital, {'one_over_q': q})
                 for q in sorted(set(options.q_values))]
        existing = [out/f'{label}.json' for label, _, _ in cases
                    if (out/f'{label}.json').exists()]
        if existing or (out/'sweep_summary.json').exists():
            parser.error('Refusing to overwrite existing sweep results; choose a new --output-dir')
    sweep_rows = []
    for label, base, overrides in cases:
        if options.case not in ('all', label):
            continue
        args = copy.deepcopy(base)
        for key, value in overrides.items():
            setattr(args, key, value)
        args.result_path = str(out / label)
        report = dict(label=label, configuration=vars(args), batches=[])

        def inspect(model, loader, device, max_eval_batches=4):
            # The production selector freezes BN; actual FT does not. This
            # probe explicitly uses FT forward mode rather than selection mode.
            model.train()
            correct = total = 0
            captured = {}

            def head_inputs(module, inputs):
                x = inputs[0].detach()
                captured.update(
                    head_input_min=float(x.min()), head_input_max=float(x.max()),
                    head_input_rms=float(x.square().mean().sqrt()),
                    head_input_nonzero_percent=float((x != 0).float().mean() * 100),
                    head_input_sample_std=float(x.std(dim=0).mean()),
                    head_input_mean_unitless=float(x.mean() / module.q))

            handle = model.fc.register_forward_pre_hook(head_inputs)
            try:
                for index, (x, target) in enumerate(loader):
                    x, target = x.to(device), target.to(device)
                    model.zero_grad(set_to_none=True)
                    raw = model(x)
                    logits = model.fc.logits_for_loss(raw)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    loss.backward()
                    count = int(raw.argmax(1).eq(target).sum())
                    correct += count
                    total += target.numel()
                    gradients = [p.grad.detach().float().square().sum()
                                 for p in model.parameters() if p.grad is not None]
                    report['batches'].append(dict(
                        index=index, correct=count, samples=target.numel(),
                        ce=float(loss.detach()),
                        logits_class_std=float(logits.detach().std(dim=1).mean()),
                        gradient_norm=float(torch.stack(gradients).sum().sqrt()),
                        head_gradient_norm=float(model.fc.weight.grad.norm()),
                        **captured))
                    if index + 1 >= max_eval_batches:
                        break
            finally:
                handle.remove()
            report['accuracy_percent'] = 100 * correct / total
            return report['accuracy_percent']

        evaluator.evaluate = inspect
        evaluator.evaluate_once(args)
        (out / f'{label}.json').write_text(json.dumps(report, indent=2) + '\n')
        print('HEAD_PROBE', label, report['accuracy_percent'],
              report['batches'][0], flush=True)
        sweep_rows.append(dict(one_over_q=args.one_over_q,
                               accuracy_percent=report['accuracy_percent']))
        if options.q_values:
            winner = max(sweep_rows, key=lambda row: (row['accuracy_percent'], -row['one_over_q']))
            (out/'sweep_summary.json').write_text(json.dumps(dict(
                method='training-mode BN/dropout; four batches; no optimizer updates; fresh PT reload per q',
                source_job=2204999, candidates=sweep_rows,
                selected_one_over_q=winner['one_over_q'], complete=len(sweep_rows)==len(cases)), indent=2)+'\n')
        torch.cuda.empty_cache()
    if options.q_values and sweep_rows:
        print('TRAINMODE_SWEEP_SELECTED', winner, flush=True)


if __name__ == '__main__':
    main()
