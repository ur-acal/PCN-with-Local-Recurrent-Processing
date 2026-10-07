#!/usr/bin/env python3
"""Observe one real production eval batch; independently verify table-to-update mapping.

Accepts ode_inference.py arguments. Writes <rhs_current_audit_path>.proof.json.
The production evaluator installs the clamp before this observer is attached.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import ode_inference


class VerifiedBatch(Exception):
    pass


class Proof:
    def start(self, model, args, trial):
        self.args, self.rows = args, []
        # Independent parser: never use production read_summary or its mapping.
        lines = Path(args.rhs_current_summary).read_text().splitlines()
        table = [(i, [v.strip() for v in line.strip('|').split('|')])
                 for i, line in enumerate(lines, 1) if line.startswith('|')]
        header = table[0][1]
        percentile = f'{args.rhs_current_bound_percentile:g}'
        explicit = lines[0].startswith('# Custom ')
        unit = 'A' if explicit and 'Lower (A)' in header else 'µA'
        scale = 1. if unit == 'A' else 1e-6
        lo = header.index(f'Lower ({unit})' if explicit else f'Lower {percentile}% (µA)')
        hi = header.index(f'Upper ({unit})' if explicit else f'Upper {percentile}% (µA)')
        combined = 'combined' in lines[0]
        expected = {(row[0], row[1].upper()): (n, float(row[lo])*scale, float(row[hi])*scale)
                    for n, row in table[2:]}
        # CurrentRecorder is the code that assigned layer labels in the original run.
        from diagnostic_scripts.plot_toggle_current_distributions import CurrentRecorder
        blocks = CurrentRecorder(model, None)._blocks()
        for block, layer, forced_branch in blocks:
            for stage in (('z',) if forced_branch else ('z', 'y')):
                branch = forced_branch or ('FB' if stage == 'z' else 'FF')
                key = (layer, 'COMBINED' if combined else branch)
                line_number, low, high = expected[key]
                limit = block._rhs_current_limits[stage]
                assert limit.metadata['summary_row'] == line_number
                assert limit.metadata['layer'] == layer
                assert limit.metadata['branch'] == branch
                assert limit.metadata['lower_A'] == low
                assert limit.metadata['upper_A'] == high
                row = dict(layer=layer, branch=branch, stage=stage,
                           summary_row=line_number, lower_A=low, upper_A=high,
                           production_module=limit.metadata['module'], calls=0,
                           max_current_error_A=0., max_update_error_V=0.,
                           clipped_lower=0, clipped_upper=0)
                self.rows.append(row)

                def observe(actual_limit, actual_block, state, duration, rhs,
                            summing, coupler, current, limited, updated,
                            expected_block=block, expected_limit=limit,
                            expected_stage=stage, lower=low, upper=high, result=row):
                    assert actual_block is expected_block
                    assert actual_limit is expected_limit
                    cap = actual_block._stage_capacitance(expected_stage)
                    expected_current = rhs * cap
                    if summing is not None:
                        expected_current = expected_current + summing * cap / duration
                    if coupler is not None:
                        expected_current = expected_current + coupler * cap / duration
                    expected_limited = torch.minimum(torch.maximum(
                        expected_current, expected_current.new_tensor(lower)),
                        expected_current.new_tensor(upper))
                    expected_update = state + duration / cap * expected_limited
                    torch.testing.assert_close(current, expected_current, rtol=0, atol=0)
                    torch.testing.assert_close(limited, expected_limited, rtol=0, atol=0)
                    torch.testing.assert_close(updated, expected_update, rtol=0, atol=0)
                    result['calls'] += 1
                    result['clipped_lower'] += int((expected_current < lower).sum())
                    result['clipped_upper'] += int((expected_current > upper).sum())

                limit.observer = observe

    def record(self, batch_idx, targets, logits):
        assert all(row['calls'] > 0 for row in self.rows), [
            (row['layer'], row['branch']) for row in self.rows if not row['calls']]
        assert sum(row['clipped_lower'] + row['clipped_upper'] for row in self.rows) > 0
        path = Path(self.args.rhs_current_audit_path + '.proof.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(model=self.args.model_name,
            model_dir=self.args.model_dir, summary=self.args.rhs_current_summary,
            percentile=self.args.rhs_current_bound_percentile,
            batch_samples=len(targets), mappings=self.rows, passed=True), indent=2)+'\n')
        print(f'RHS_PROOF_PASSED {path}', flush=True)
        raise VerifiedBatch()


if __name__ == '__main__':
    try:
        ode_inference.run_ode_inference(linear_study=Proof())
    except VerifiedBatch:
        pass
