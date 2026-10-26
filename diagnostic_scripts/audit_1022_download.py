"""Extract compact, line-referenced evidence from downloaded 1022 Slurm logs."""
import json
from pathlib import Path
import re
import statistics
import copy
import shlex
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'results/1022_download_audit'
    output.mkdir(exist_ok=True)
    if '--sanity-head-ablation' in sys.argv:
        return sanity_head_ablation(output)
    jobs = {}
    for job in [*range(2196406,2196410), *range(2196950,2196958)]:
        path = root/f'logs/slurm_jobs/slurm_{job}.out'
        data = dict(log=str(path), final=[], selection=[], evaluations=[], errors=[], commands=[])
        lines = path.read_bytes().decode(errors='replace').split('\n')
        evidence = []
        for number, line in enumerate(lines, 1):
            if '\r' in line:
                continue
            line = line.strip()
            if line.startswith('----- Final'):
                data['final'].append(dict(line=number, text=line))
            if line.startswith(('SELECTION_', '[SELECTION]')):
                data['selection'].append(dict(line=number, text=line))
            if line.startswith('Running: python'):
                data['commands'].append(dict(line=number, command=line))
            match = re.fullmatch(r'Running (\w+) \((\d+) trials\)', line)
            if match:
                data['evaluations'].append(dict(corner=match[1], trials=[],
                    expected_trials=int(match[2]), line=number))
            match = re.fullmatch(r'trial (\d+): ([\d.]+)%', line)
            if match and data['evaluations']:
                data['evaluations'][-1]['trials'].append(float(match[2]))
            if ('RuntimeError:' in line or 'printf:' in line or 'Traceback' in line):
                data['errors'].append(dict(line=number, text=line))
            if (line.startswith(('WARNING:root:', 'Running:', '-----', '[', 'SELECTION_',
                                 'RuntimeError:', 'Traceback', '| FS_', 'Final validation'))
                    or (data['evaluations'] and re.fullmatch(r'trial \d+: [\d.]+%', line))):
                evidence.append(f'{number}: {line}')
        for evaluation in data['evaluations']:
            trials = evaluation['trials']
            evaluation['mean_percent'] = statistics.mean(trials) if trials else None
            evaluation['completed_trials'] = len(trials)
            evaluation['status'] = ('complete' if len(trials) == evaluation['expected_trials']
                                    else 'incomplete')
        (output/f'{job}_evidence.txt').write_text('\n'.join(evidence)+'\n')
        jobs[str(job)] = data
    (output/'log_summary.json').write_text(json.dumps(jobs, indent=2)+'\n')
    for job, data in jobs.items():
        print(job, [d['text'] for d in data['final']],
              [(d['corner'], d['mean_percent'], d['status'],
                f"{d['completed_trials']}/{d['expected_trials']}") for d in data['evaluations']],
              [d['text'] for d in data['selection'] if 'COMPLETE' in d['text']],
              [d['text'] for d in data['errors']])


def sanity_head_ablation(output):
    """Replay failed job's exact ten-batch sanity; vary only classifier defects."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diagnostic_scripts.select_feedforward_one_over_q import args_from_ft_command
    from baseline import evaluate_physical_feedforward_cifar as evaluator
    logs = json.loads((output/'log_summary.json').read_text())
    # The FT launcher prints its resolved trainer command, then passes exactly
    # those arguments into the selector via --from_ft_command.
    command = next(row['command'] for row in logs['2196953']['commands']
                   if '--resume_checkpoint' in row['command'])
    argv = shlex.split(command)[3:]
    args = args_from_ft_command(argv)
    for key in ('enable_nonlinear_R', 'enable_measured_activation',
                'enable_measured_pooling', 'enable_spin_variation',
                'enable_summing_current_noise', 'enable_coupler_noise',
                'enable_dtc_nonideality', 'enable_slow_summing_current',
                'enable_slow_coupler_noise'):
        setattr(args, key, False)
    args.w_bits = 8
    args.weight_quant_factor_bits = -1
    args.nonlinear_R_train_mode = 'none'
    args.ft_corner_coupled_sampling = False
    args.activation_corner_mode = 'fixed'
    args.selection_training_hardware = False
    args.one_over_q = 128
    rows = []
    for label, physical, adc, quantize in (
        ('unitless_reference', False, .5, True),
        ('original_physical_sanity', True, .5, True),
        ('physical_ADC_noise_off', True, 0., True),
        ('physical_head_quantization_and_ADC_off', True, 0., False)):
        current = copy.copy(args)
        current.physical = physical
        current.final_adc_noise_lsb = adc
        current.final_head_quantize = quantize
        current.result_path = str(output/'sanity_head_ablation'/label)
        accuracy = evaluator.evaluate_once(current)
        rows.append(dict(label=label, accuracy_percent=accuracy,
                         configuration=vars(current)))
        (output/'sanity_head_ablation.json').write_text(json.dumps(rows, indent=2, default=str)+'\n')
        print('SANITY_HEAD_ABLATION', label, accuracy, flush=True)


if __name__ == '__main__':
    main()
