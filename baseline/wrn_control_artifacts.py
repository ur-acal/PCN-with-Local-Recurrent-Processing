"""Submission inventory and portable results-only archives for WRN controls."""

import argparse
import io
import json
from pathlib import Path
import tarfile
import time
import uuid

from baseline.wrn_control_specs import model_name


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def create_plan(root, rows, datasets, sizes, conditions, stage, simulated=False):
    folder = Path(root) / 'submissions'
    folder.mkdir(parents=True, exist_ok=True)
    plan = folder / f'{uuid.uuid4().hex}.json'
    tasks = [dict(row=r, dataset=d, size=s, model=model_name(r, s),
                  directory=f'row{r}/{d}/{s}', submission_status='planned')
             for r in rows for d in datasets for s in sizes]
    write_json(plan, dict(created=time.time(), simulated=simulated, stage=stage,
                          conditions=conditions, tasks=tasks))
    return plan


def record_submission(plan, row, dataset, size, job_id):
    data = json.loads(plan.read_text())
    for task in data['tasks']:
        if (task['row'], task['dataset'], task['size']) == (row, dataset, size):
            task.update(submission_status='submitted', job_id=job_id)
            write_json(plan, data)
            return
    raise ValueError('Submission was not in the planned task list')


def collect(root):
    from baseline.run_wrn_controls import validate_results
    plans = sorted((root / 'submissions').glob('*.json'))
    if not plans:
        raise ValueError(f'No submission inventory found: {root}')
    expected = {}
    simulated = set()
    for plan in plans:
        data = json.loads(plan.read_text())
        simulated.add(data['simulated'])
        for task in data['tasks']:
            entry = expected.setdefault(task['directory'], dict(task, conditions=set(), submissions=[]))
            entry['submissions'].append(dict(plan=str(plan.relative_to(root)), **task))
            if data['stage'] != 'train':
                entry['conditions'].update(data['conditions'])
    if len(simulated) != 1:
        raise ValueError('Cannot mix simulated and real submissions')
    inventory = dict(simulated=simulated.pop(), collected=time.time(), tasks=[])
    for relative, entry in sorted(expected.items()):
        task = root / relative
        errors = []
        try:
            manifest = json.loads((task / 'manifest.json').read_text())
            identity = manifest['identity']
            if (identity['model'] != entry['model'] or identity['dataset'] != entry['dataset']
                    or identity['row'] != entry['row'] or identity['size'] != entry['size']
                    or manifest.get('simulated', False) != inventory['simulated']):
                raise ValueError('Task manifest identity mismatch')
            trained = json.loads((task / 'train_complete.json').read_text())
            state = json.loads((task / 'state.json').read_text())
            if state['status'] != 'complete':
                errors.append(f"Task state: {state['status']}")
            for condition in sorted(entry['conditions']):
                output = task / 'evaluation' / condition
                completion = json.loads((output / 'complete.json').read_text())
                checked = completion['identity']
                command = checked['command']
                def value(flag):
                    return command[command.index(flag) + 1]
                if (checked['checkpoint_sha256'] != trained['sha256']
                        or checked['simulated'] != inventory['simulated']
                        or value('--model_name_override') != entry['model']
                        or value('--datasets') != entry['dataset']
                        or value('--architectures') != 'WRN_' + entry['size']
                        or value('--checkpoint_override') != trained['checkpoint']):
                    raise ValueError(f'Wrong evaluation identity: {condition}')
                if completion['artifacts'] != validate_results(output, command):
                    raise ValueError(f'Changed evaluation artifacts: {condition}')
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
        entry['conditions'] = sorted(entry['conditions'])
        entry.update(complete=not errors, errors=errors)
        inventory['tasks'].append(entry)
    inventory['expected_tasks'] = len(expected)
    inventory['complete_tasks'] = sum(t['complete'] for t in inventory['tasks'])
    return inventory


def pack(root, archive, allow_incomplete=False):
    root, archive = root.resolve(), archive.resolve()
    inventory = collect(root)
    if not allow_incomplete and inventory['complete_tasks'] != inventory['expected_tasks']:
        raise ValueError(f"Only {inventory['complete_tasks']}/{inventory['expected_tasks']} tasks complete; "
                         'use --allow-incomplete to collect failed/pending-task evidence')
    files = list((root / 'submissions').glob('*.json'))
    for task in inventory['tasks']:
        folder = root / task['directory']
        files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv', '.log'}]
    files = sorted(set(files))
    if any(not p.resolve().is_relative_to(root) for p in files):
        raise ValueError('Archive input escapes the experiment root')
    inventory['files'] = [str(p.relative_to(root)) for p in files]
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects any previous archive.
    with archive.open('xb') as handle:
        with tarfile.open(fileobj=handle, mode='w:gz') as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            payload = (json.dumps(inventory, indent=2) + '\n').encode()
            info = tarfile.TarInfo('inventory.json')
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('plan')
    create.add_argument('--output-root', type=Path, required=True)
    create.add_argument('--rows', required=True)
    create.add_argument('--datasets', required=True)
    create.add_argument('--sizes', required=True)
    create.add_argument('--conditions', required=True)
    create.add_argument('--stage', choices=('train', 'test', 'train-test'), required=True)
    create.add_argument('--simulate', action='store_true')
    record = commands.add_parser('submitted')
    record.add_argument('--plan', type=Path, required=True)
    record.add_argument('--row', type=int, required=True)
    record.add_argument('--dataset', required=True)
    record.add_argument('--size', required=True)
    record.add_argument('--job-id', required=True)
    bundle = commands.add_parser('pack')
    bundle.add_argument('--output-root', type=Path, required=True)
    bundle.add_argument('--archive', type=Path, required=True)
    bundle.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.action == 'plan':
        root = args.output_root.resolve()
        if args.simulate:
            root /= 'SIMULATED'
        print(create_plan(root, list(map(int, args.rows.split(','))), args.datasets.split(','),
                          args.sizes.split(','), args.conditions.split(','), args.stage, args.simulate))
    elif args.action == 'submitted':
        record_submission(args.plan, args.row, args.dataset, args.size, args.job_id)
    else:
        inventory = pack(args.output_root, args.archive, args.allow_incomplete)
        print(f"Packed {inventory['complete_tasks']}/{inventory['expected_tasks']} completed tasks "
              f"into {args.archive}; simulated={inventory['simulated']}")


if __name__ == '__main__':
    main()
