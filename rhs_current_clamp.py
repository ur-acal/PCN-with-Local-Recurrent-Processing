"""Evaluation-only, slice-averaged total-current limits from diagnostic tables."""

import hashlib
import fcntl
import json
import math
from pathlib import Path
import re

import torch


def add_rhs_current_args(parser):
    parser.add_argument('--rhs_current_summary', default=None)
    parser.add_argument('--rhs_current_bound_percentile', type=float, default=99.)
    parser.add_argument('--rhs_current_audit_path', default=None,
                        help='Append per-trial layer mapping and clipping counts as JSONL.')


def read_summary(path, percentile):
    path = Path(path).resolve()
    text = path.read_text()
    if not math.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError('Current bound percentile must be between 0 and 100.')
    title = text.splitlines()[0].strip().lower()
    explicit = title in ('# custom separate current limits', '# custom combined current limits')
    if not explicit and title not in ('# total separate summing-current distributions',
                     '# total combined summing-current distributions'):
        raise ValueError('RHS clamping requires a total-current summary (separate or combined).')
    mode = 'separate' if 'separate' in title else 'combined'
    lines = [(n, line) for n, line in enumerate(text.splitlines(), 1)
             if line.startswith('|')]
    if len(lines) < 3:
        raise ValueError('Missing current summary table.')
    cells = lambda line: [value.strip() for value in line.strip().strip('|').split('|')]
    header = cells(lines[0][1])
    selected = []
    units = {'A': 1., 'mA': 1e-3, 'µA': 1e-6, 'μA': 1e-6, 'uA': 1e-6, 'nA': 1e-9}
    for side in ('Lower', 'Upper'):
        matches = []
        for index, column in enumerate(header):
            pattern = r'(Lower|Upper) \(([^)]+)\)' if explicit else r'(Lower|Upper) ([\d.]+)% \(([^)]+)\)'
            match = re.fullmatch(pattern, column)
            if match and match[1] == side and (explicit or float(match[2]) == percentile):
                unit = match[2] if explicit else match[3]
                if unit not in units:
                    raise ValueError('Unsupported current unit: ' + unit)
                matches.append((index, units[unit]))
        if len(matches) != 1:
            raise ValueError(f'Expected one {side} {percentile:g}% column in {path}.')
        selected.append(matches[0])
    rows = {}
    for line_number, line in lines[2:]:
        values = cells(line)
        if len(values) != len(header):
            raise ValueError(f'Malformed summary row at {path}:{line_number}')
        layer, branch = values[header.index('Layer')], values[header.index('Branch')]
        branch = branch.upper()
        if branch not in (('FF', 'FB') if mode == 'separate' else ('COMBINED',)):
            raise ValueError(f'Invalid branch {branch} for {mode} summary.')
        low, high = [float(values[index]) * scale for index, scale in selected]
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f'Invalid current bounds at {path}:{line_number}')
        if (layer, branch) in rows:
            raise ValueError(f'Duplicate current row: {layer} {branch}')
        rows[layer, branch] = dict(summary_row=line_number, summary_branch=branch,
                                  lower_A=low, upper_A=high,
                                  bounds_source='explicit' if explicit else 'fitted_percentile')
    return mode, rows, hashlib.sha256(text.encode()).hexdigest()


class CurrentLimit:
    def __init__(self, metadata):
        self.metadata = metadata
        self.counts = None
        self.calls = 0
        self.observer = None  # Diagnostics observe the actual production clamp.

    def apply(self, block, state, duration, rhs, summing_delta, coupler_delta):
        capacitance = block._stage_capacitance(self.metadata['stage'])
        current = rhs * capacitance
        for delta in (summing_delta, coupler_delta):
            if delta is not None:
                current = current + delta * capacitance / duration
        low, high = self.metadata['lower_A'], self.metadata['upper_A']
        limited = current.clamp(min=low, max=high)
        counts = torch.stack(((current < low).sum(), (current > high).sum(),
                              current.new_tensor(current.numel(), dtype=torch.int64)))
        self.counts = counts if self.counts is None else self.counts + counts
        self.calls += 1
        updated = state + (duration / capacitance) * limited
        if self.observer is not None:
            self.observer(self, block, state, duration, rhs, summing_delta,
                          coupler_delta, current, limited, updated)
        return updated


def install_rhs_current_clamp(model, args):
    """Use the same 1-based PcConvs enumeration as CurrentRecorder._blocks."""
    if not args.rhs_current_summary:
        return []
    from ode_pc import TogglePulseFFFB
    from final_linear import AnalogLinear
    if model.training:
        raise ValueError('RHS current summary clamping is evaluation-only.')
    mode, rows, digest = read_summary(args.rhs_current_summary,
                                     args.rhs_current_bound_percentile)
    blocks = [(block, f'layer_{index:02d}', f'PcConvs.{index-1}', ('z', 'y'))
              for index, block in enumerate(model.PcConvs, 1)]
    head = getattr(model, 'linear', None)
    if isinstance(head, AnalogLinear):
        # One-shot FF conv1 executes internal stage z; its summary branch is FF.
        blocks.append((head._circuit, 'final_linear', 'linear._circuit', ('z',)))
    assignments, consumed = [], set()
    for block, layer, module, stages in blocks:
        if not isinstance(block, TogglePulseFFFB) or not block.toggle_fast_path:
            raise ValueError(f'{module}: current clamp requires Level-3 toggle fast path.')
        for stage in stages:
            branch = 'FB' if stage == 'z' and layer != 'final_linear' else 'FF'
            key = (layer, branch if mode == 'separate' else 'COMBINED')
            if key not in rows:
                raise ValueError(f'Missing current bounds for {key} ({module}, {stage}).')
            consumed.add(key)
            metadata = dict(rows[key], layer=layer, module=module, stage=stage,
                            branch=branch, percentile=(None if rows[key]['bounds_source'] == 'explicit'
                                                       else args.rhs_current_bound_percentile),
                            summary=str(Path(args.rhs_current_summary).resolve()),
                            summary_sha256=digest)
            assignments.append((block, stage, CurrentLimit(metadata)))
    if consumed != set(rows):
        raise ValueError(f'Unused summary rows (model mismatch): {set(rows) - consumed}')
    # Validate everything before installing any limits.
    limits = []
    for block, stage, limit in assignments:
        if not hasattr(block, '_rhs_current_limits'):
            block._rhs_current_limits = {}
        block._rhs_current_limits[stage] = limit
        limits.append(limit)
        print('RHS_CURRENT_MAP ' + json.dumps(limit.metadata), flush=True)
    return limits


def report_rhs_current_clamp(limits, args, trial, accuracy):
    if not limits:
        return
    rows = []
    for limit in limits:
        if limit.counts is None:
            raise RuntimeError('Current limit was never applied: ' + str(limit.metadata))
        lower, upper, samples = limit.counts.cpu().tolist()
        rows.append(dict(limit.metadata, calls=limit.calls, samples=samples,
                         clipped_lower=lower, clipped_upper=upper))
    result = dict(case=getattr(args, 'ablation_case_name', ''), trial=trial,
                  model_name=getattr(args, 'model_name', None),
                  checkpoint=getattr(args, 'ckpt', None),
                  accuracy=accuracy, mappings=rows)
    print('RHS_CURRENT_AUDIT ' + json.dumps(result), flush=True)
    if args.rhs_current_audit_path:
        path = Path(args.rhs_current_audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(result) + '\n')
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
