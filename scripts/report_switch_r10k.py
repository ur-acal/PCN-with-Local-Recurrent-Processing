"""Compact 324-row study table and complete layer/inversion CSV exports."""
import csv
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/switch_full_network_r10k_rmax150k_enobnone'
PAPER = ROOT.parent / 'papers/hardware-native-neural-ode/shared/pixel_switch_method_study.md'
BEGIN = '<!-- MATCHED_R10K_BEGIN -->'
END = '<!-- MATCHED_R10K_END -->'
CONFIGS = [(method, m, n, tol) for method in ('Jacobi', 'Lie', 'Strang')
           for m in (1, 2, 3, 4) for n in (5, 6, 7, 8, 9, 10, 12, 15, 20)
           for tol in (1e-6, 1e-7, 1e-8)]


def atomic_text(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def write_csv(path, fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def case_id(config):
    method, m, n, tol = config
    return f'{method}_m{m}_n{n}_tol{tol:.0e}'


def inversions(rows):
    """All completed pairs; retain whether only block size differs."""
    pairs = []
    for a in rows:
        for b in rows:
            if a['accuracy_gap_pp'] <= b['accuracy_gap_pp']:
                continue
            for metric in ('logit_relative_error', 'mean_layer_relative_error', 'first_layer_relative_error'):
                if a[metric] >= b[metric]:
                    continue
                pairs.append({'metric': metric, 'lower_error_id': a['id'],
                              'lower_error_case': a['case'], 'comparison_id': b['id'],
                              'comparison_case': b['case'], 'lower_error': a[metric],
                              'comparison_error': b[metric], 'larger_gap_pp': a['accuracy_gap_pp'],
                              'comparison_gap_pp': b['accuracy_gap_pp'],
                              'lower_error_accuracy_pct': 100*a['accuracy'],
                              'comparison_accuracy_pct': 100*b['accuracy'],
                              'lower_actual_accuracy': a['accuracy'] < b['accuracy'],
                              'only_block_size_differs': all(a[k] == b[k] for k in ('method', 'n_iters', 'tol'))})
    return pairs


def render(out=OUT, paper=PAPER):
    status = json.loads((out/'status.json').read_text())
    reference = None
    if (out/'reference_summary.json').exists():
        controls = json.loads((out/'reference_summary.json').read_text())
        reference = next((c['accuracy'] for c in controls if c['tol'] == 1e-6), None)
    rows, layers, complete = [], [], []
    for index, config in enumerate(CONFIGS, 1):
        name = case_id(config)
        folder = out/'cases'/name
        count = len(list(folder.glob('batch_*.pt')))
        state = 'failed' if name in status.get('failed_cases', {}) else ('running' if folder.exists() else 'pending')
        row = dict(id=f'C{index:03d}', case=name, method=config[0], m=config[1],
                   n_iters=config[2], tol=config[3], status=state, batches=count)
        if (folder/'summary.json').exists():
            result = json.loads((folder/'summary.json').read_text())
            assert result['n_samples'] == 896 and count == 7 and len(result['layers']) == 16, name
            row.update(result)
            row.update(status='complete', batches=7,
                       mean_layer_relative_error=sum(x['output_relative_error'] for x in result['layers'])/16,
                       first_layer_relative_error=result['layers'][0]['output_relative_error'],
                       # Accuracies count discrete correct predictions. Recover
                       # that integer difference to avoid false unequal gaps
                       # from subtraction rounding on opposite sides of reference.
                       accuracy_gap_pp=100*abs(round((result['accuracy']-result['reference_accuracy'])*896))/896)
            complete.append(row)
        rows.append(row)
        for layer in range(16):
            item = {k: row[k] for k in ('id', 'case', 'method', 'm', 'n_iters', 'tol', 'status', 'batches')}
            item.update(layer=layer+1, n_samples=row.get('n_samples', ''))
            if row['status'] == 'complete':
                item.update({k: v for k, v in row['layers'][layer].items() if k != 'index'})
            layers.append(item)
    pairs = inversions(complete)
    pair_fields = ['metric', 'lower_error_id', 'lower_error_case', 'comparison_id', 'comparison_case',
                   'lower_error', 'comparison_error', 'larger_gap_pp', 'comparison_gap_pp',
                   'lower_error_accuracy_pct', 'comparison_accuracy_pct', 'lower_actual_accuracy', 'only_block_size_differs']
    write_csv(out/'error_accuracy_inversions.csv', pair_fields, pairs)
    write_csv(out/'layer_metrics.csv',
              ['id', 'case', 'method', 'm', 'n_iters', 'tol', 'status', 'batches', 'n_samples', 'layer',
               'input_relative_error', 'pre_quant_relative_error', 'output_relative_error', 'rail_fraction'], layers)
    flags = {}
    for pair in pairs:
        flags.setdefault(pair['lower_error_id'], {}).setdefault(pair['metric'], []).append(pair['comparison_id'])

    def markdown(base):
        link = lambda name: Path(os.path.relpath(out/name, base)).as_posix()
        lines = ['## Matched 324-configuration study: R=10 kΩ, R_max=150 kΩ, ENOB=None', '',
                 'Purpose: investigate the earlier cases with smaller relative error but lower accuracy. '
                 'This PIXEL_SWITCHED software study measures both quantities on identical noisy inputs.', '',
                 f"Status: **{status['status']}**; **{len(complete)}/324 complete**, {len(status.get('failed_cases', {}))} recorded failures.", '',
                 'Same trained scanGFI CIFAR-100 16L96C checkpoint; five-bit weights, C=49 fF, V_DD=0.1 V. '
                 'Each configuration and reference receives the same seven saved batches of 128 images '
                 '(896 images), including identical sensor noise. No input noise is regenerated. '
                 'Full unsplit reference: Dopri5 rtol=atol=1e-6, rebuilt with R=10 kΩ, R_max=150 kΩ and ENOB=None. '
                 'Tighter references are controls. Accuracies are against dataset ground-truth labels, not agreement with reference predictions.', '',
                 '**Logit rel:** L2 relative error of all 100 pre-softmax scores, pooled over 896 images. '
                 '**Mean layer rel:** arithmetic mean of the 16 actual layer-output relative errors; '
                 'each layer error pools squared differences/reference values over the same 896 images. '
                 '**Gap:** absolute difference between case and reference ground-truth accuracies, in percentage points. '
                 'Only finished seven-batch cases receive final metrics; blanks mean pending/unfinished.', '',
                 f"[Layer details CSV: all 324 × 16 rows]({link('layer_metrics.csv')}) · "
                 f"[All error/accuracy inversion pairs CSV]({link('error_accuracy_inversions.csv')}) · "
                 f"[Manifest and input hashes]({link('manifest.json')}) · [Reference controls]({link('reference_summary.json')})", '',
                 '**Flags identify lower-error cases with a larger absolute accuracy gap than the listed peers:** '
                 'L = logit error; A = mean layer error; F = first-layer output error (the original motivation). '
                 'All completed configurations are compared; the pair CSV also marks comparisons where only m differs '
                 'and whether actual accuracy is lower. Up to two peer IDs per metric are shown; +N counts additional peers. '
                 'Flags are observations, not proof of a bug or an explanation of the cause.', '',
                 '| ID | Configuration (method / m / iterations / tolerance) | Logit rel | Mean layer rel | 7-batch acc (%) | Reference GT acc (%) | Gap (pp) | Lower-error / larger-gap peers | Status |',
                 '|---|---|---:|---:|---:|---:|---:|---|---|']
        for row in rows:
            cfg = f"{row['method']} / {row['m']} / {row['n_iters']} / {row['tol']:.0e}"
            parts = []
            for metric, tag in [('logit_relative_error', 'L'), ('mean_layer_relative_error', 'A'), ('first_layer_relative_error', 'F')]:
                peers = flags.get(row['id'], {}).get(metric, [])
                if peers:
                    parts.append(f"{tag}: {','.join(peers[:2])}" + (f' +{len(peers)-2}' if len(peers)>2 else ''))
            flag = '; '.join(parts) or '—'
            fmt = lambda key, scale=1: f'{scale*row[key]:.6f}' if key in row else '—'
            ref = row.get('reference_accuracy', reference)
            ref_text = f'{100*ref:.4f}' if ref is not None else '—'
            lines.append(f"| {row['id']} | {cfg} | {fmt('logit_relative_error')} | {fmt('mean_layer_relative_error')} | "
                         f"{fmt('accuracy',100)} | {ref_text} | {fmt('accuracy_gap_pp')} | {flag} | {row['status']} ({row['batches']}/7) |")
        return '\n'.join(lines)+'\n'

    atomic_text(out/'summary.md', markdown(out))
    if paper is not None:
        text = paper.read_text()
        section = BEGIN+'\n'+markdown(paper.parent)+END
        if BEGIN in text:
            before, rest = text.split(BEGIN, 1)
            _, after = rest.split(END, 1)
            text = before+section+after
        else:
            title, remainder = text.split('\n', 1)
            text = title+'\n\n'+section+'\n'+remainder
        atomic_text(paper, text)
    return len(complete), len(layers), len(pairs)


if __name__ == '__main__':
    print(render())
