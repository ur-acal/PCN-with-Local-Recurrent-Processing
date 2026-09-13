"""Publish saved per-layer study metrics without touching running numerical code."""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/switch_full_network_matched'


def render():
    status = json.loads((OUT / 'status.json').read_text())
    cases = [json.loads(p.read_text()) for p in sorted((OUT / 'cases').glob('*/summary.json'))]
    lines = [
        '# Live per-layer relative errors', '',
        f"Study status: {status['status']}. Completed configurations shown: {len(cases)}/324.",
        f"Reported failed configurations: {len(status.get('failed_cases', {}))}. Failed or unfinished cases are not shown as completed results.", '',
        '[Overall accuracy/logit table](summary.md) · [Study status and failures](status.json)', '',
        'All errors compare the current model trajectory with the full unsplit reference',
        '(Dopri5 rtol=atol=1e-6) on the same seven fixed batches, 896 images.',
        'Each entry is sqrt(sum squared differences / sum squared reference values),',
        'aggregated over all examples and tensor entries. Layer numbers below are',
        '1-based; the saved JSON indices are 0-based.', '',
        '- **Input error:** input to this layer versus its reference-model input.',
        '- **Pre-ENOB output error:** physical layer output before output quantization.',
        '- **Post-ENOB output error:** actual layer output after ENOB quantization and wrapper output scaling.', '',
        'These are end-to-end propagated trajectory errors, including upstream errors;',
        'they are not isolated local-flow errors obtained by forcing identical inputs at every layer.', '',
        '## Completed configurations', '',
    ]
    for i, case in enumerate(cases):
        lines.append(f"- [{case['case']}](#case-{i+1})")
    for i, case in enumerate(cases):
        assert len(case['layers']) == 16, case['case']
        lines += ['', f'<a id="case-{i+1}"></a>',
                  f"## {case['method']}: m={case['m']}, iterations={case['n_iters']}, tolerance={case['tol']:.0e}", '',
                  f"Subset accuracy: {100*case['accuracy']:.4f}%; reference: {100*case['reference_accuracy']:.4f}%; logit relative error: {case['logit_relative_error']:.8f}.", '',
                  '| Layer | Input relative error | Pre-ENOB output relative error | Post-ENOB output relative error |',
                  '|---:|---:|---:|---:|']
        for layer in case['layers']:
            lines.append(f"| {layer['index']+1} | {layer['input_relative_error']:.8f} | {layer['pre_quant_relative_error']:.8f} | {layer['output_relative_error']:.8f} |")
        lines += ['', f"[Raw metrics](cases/{case['case']}/summary.json)"]
    tmp = OUT / 'layer_errors.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(OUT / 'layer_errors.md')
    return status['status'], len(cases)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    last = None
    while True:
        result = render()
        if result != last:
            print(f'Published per-layer tables: {result}', flush=True)
            last = result
        if not args.watch or result[0] in {'complete', 'finished with failures', 'failed/interrupted'}:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
