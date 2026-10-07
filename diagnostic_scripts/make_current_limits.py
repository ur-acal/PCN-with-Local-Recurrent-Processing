#!/usr/bin/env python3
"""Expand statistics or explicit manual bounds into evaluator-ready layer tables."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from current_stages import stage_members
from rhs_current_clamp import read_summary


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def generate(args):
    members = stage_members(args.num_layers, args.pool_positions)
    groups = members if args.granularity == 'stage' else {
        layer: [layer] for layers in members.values() for layer in layers}
    if args.no_final_linear:
        groups.pop('final_linear')
    mode = 'combined' if args.branches == 'pooled' else 'separate'
    expected = {(group, branch) for group in groups for branch in
                (('COMBINED',) if mode == 'combined' else
                 (('FF',) if group == 'final_linear' else ('FF', 'FB')))}
    if args.summary:
        import hashlib
        import json
        summary_path = Path(args.summary).resolve()
        summary_text = summary_path.read_text()
        summary_sha256 = hashlib.sha256(summary_text.encode()).hexdigest()
        metadata_path = summary_path.parent.parent.parent/'run_config.json'
        source_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if metadata_path.is_file() and args.granularity == 'stage':
            recorded = source_metadata.get('stage_members')
            if recorded is not None and recorded != members:
                raise ValueError('Requested pool positions/layer count disagree with source stage metadata.')
        parsed_mode, rows, _ = read_summary(args.summary, args.percentile)
        if parsed_mode != mode:
            raise ValueError('Source branch mode does not match --branches.')
        bounds = {key: (row['lower_A'], row['upper_A']) for key, row in rows.items()}
    else:
        import math
        bounds = {}
        for group, branch, low, high in args.bound or []:
            key = (group, branch.upper())
            if key in bounds:
                raise ValueError(f'Duplicate manual assignment: {key}')
            low, high = float(low)*1e-6, float(high)*1e-6
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f'Invalid bounds for {key}')
            bounds[key] = (low, high)
        summary_path = None
        summary_sha256 = None
        source_metadata = {}
    if set(bounds) != expected:
        raise ValueError(f'Missing assignments: {expected-set(bounds)}; unexpected: {set(bounds)-expected}')
    suffix = f'p{args.percentile:g}' if args.summary else 'manual'
    name = args.name or f'current_limits_{args.granularity}_{args.branches}_{suffix}.md'
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('--name must be a filename, not a path.')
    directory = Path(args.output_dir)
    ancestor = directory.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if len(os.fsencode(name)) > os.pathconf(ancestor, 'PC_NAME_MAX'):
        raise ValueError('Filename exceeds filesystem NAME_MAX; choose a shorter --name.')
    path = directory / name
    if path.exists() and not args.overwrite:
        raise FileExistsError(f'{path} exists; use --overwrite explicitly.')
    provenance = (['Source: manually specified bounds.'] if summary_path is None else [
        f'Source summary: `{display_path(summary_path)}`',
        f'Source summary SHA-256: `{summary_sha256}`',
        *[f'Source {key}: `{display_path(source_metadata[key]) if key == "model_dir" else source_metadata[key]}`'
          for key in
          ('model_name', 'model_dir', 'dataset_split', 'corners', 'git_commit')
          if key in source_metadata],
    ])
    lines = [f'# Custom {mode} current limits', '',
             'Explicit signed total-current bounds. Percentile selection is not used by the evaluator.', '',
             'Values are written in amperes to preserve parsed bounds exactly (no unit-roundtrip rounding).',
             '', *provenance, '',
             '| Layer | Branch | Lower (A) | Upper (A) |', '|---|---|---:|---:|']
    for (group, branch), (low, high) in sorted(bounds.items()):
        for layer in groups[group]:
            lines.append(f'| {layer} | {branch} | {low:.17g} | {high:.17g} |')
    directory.mkdir(parents=True, exist_ok=True)
    with path.open('w' if args.overwrite else 'x') as handle:
        handle.write('\n'.join(lines)+'\n')
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', help='Total-current summary; stage rows allowed with --granularity stage.')
    p.add_argument('--percentile', type=float, default=95)
    p.add_argument('--bound', action='append', nargs=4, metavar=('GROUP', 'BRANCH', 'LOW_UA', 'HIGH_UA'))
    p.add_argument('--granularity', choices=('layer', 'stage'), default='layer')
    p.add_argument('--branches', choices=('separate', 'pooled'), default='separate')
    p.add_argument('--num_layers', type=int, required=True)
    p.add_argument('--pool_positions', type=int, nargs='*', default=[])
    p.add_argument('--no_final_linear', action='store_true')
    p.add_argument('--output_dir', default=str(ROOT/'hardware_data/summing_current_limit'))
    p.add_argument('--name')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if bool(args.summary) == bool(args.bound):
        p.error('Specify exactly one of --summary or --bound (repeat --bound for each row).')
    return args


if __name__ == '__main__':
    print(generate(parse_args()))
