#!/usr/bin/env python3
"""Verify actual MC45 worker wiring, then run the eight requested production trials."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_report(output, manifest):
    lines = ['# Total-current clamp verification and evaluation', '',
             'Same saved analog/no-bias C36→72 model as the source current summaries. '
             'Limits are fitted Gaussian intervals from FS_V2_T1 training data; '
             'all accuracy trials use the complete test set.', '',
             '| Configuration | Verified mappings | Verified slice updates | Top-1 (%) | Clipped currents (%) |',
             '|---|---:|---:|---:|---:|']
    for run in manifest['runs']:
        leaf = output/run['name']
        proof_path = leaf/'clamp_audit.jsonl.proof.json'
        if not proof_path.exists():
            continue
        proof = json.loads(proof_path.read_text())
        audit_path = leaf/'clamp_audit.jsonl'
        audit = json.loads(audit_path.read_text().splitlines()[-1]) if audit_path.exists() else None
        accuracy = f"{audit['accuracy']:.4f}" if audit else 'pending'
        clipped = 'pending'
        if audit:
            actual = audit['mappings']
            expected = {(r['layer'], r['stage']): r for r in proof['mappings']}
            assert len(actual) == len(expected)
            pooled_counts = {}
            for row in actual:
                reference = expected[row['layer'], row['stage']]
                for key in ('summary_row', 'lower_A', 'upper_A', 'branch'):
                    assert row[key] == reference[key], (run['name'], row['layer'], key)
                assert row['calls'] == reference['calls'] * 79  # 10,000 images, batch 128
                line = row['summary_row']
                pooled_counts[line] = pooled_counts.get(line, 0) + row['samples']
            summary_lines = Path(proof['summary']).read_text().splitlines()
            training_samples = json.loads((Path(manifest['source'])/'run_config.json').read_text())['accuracy']['samples']
            for line, samples in pooled_counts.items():
                source_count = int(summary_lines[line-1].split('|')[3].strip().replace(',', ''))
                assert samples * training_samples == source_count * 10000, (run['name'], line)
            clipped = f"{100*sum(r['clipped_lower']+r['clipped_upper'] for r in actual)/sum(r['samples'] for r in actual):.4f}"
        lines.append(f"| [{run['name']}]({run['name']}/mapping.md) | {len(proof['mappings'])} | "
                     f"{sum(row['calls'] for row in proof['mappings'])} | {accuracy} | {clipped} |")
        detail = ['# '+run['name'], '', 'Source: `'+proof['summary']+'`', '',
                  'Every observed production current and update matched the independent calculation '
                  'exactly in a two-image batch. Stage z in final_linear is its one-shot FF MVM.', '',
                  '| Layer | Branch | Module | Stage | Summary line | Lower (µA) | Upper (µA) | '
                  'Verified calls | Clipped below / above in proof |',
                  '|---|---|---|---|---:|---:|---:|---:|---:|']
        for row in proof['mappings']:
            detail.append(f"| {row['layer']} | {row['branch']} | {row['production_module']} | "
                f"{row['stage']} | {row['summary_row']} | {row['lower_A']*1e6:.6g} | "
                f"{row['upper_A']*1e6:.6g} | {row['calls']} | "
                f"{row['clipped_lower']} / {row['clipped_upper']} |")
        if audit:
            detail += ['', 'Full evaluation clipping counts: [clamp_audit.jsonl](clamp_audit.jsonl).',
                       'Full-run audit passed: all bounds match the proof; all stage call counts '
                       'match 79 batches; every summary row has exactly the expected number '
                       'of current samples for 10,000 test images.',
                       'Production command: [production_command.json](production_command.json).']
        (leaf/'mapping.md').write_text('\n'.join(detail)+'\n')
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def capture_worker(env, output):
    """Run the real shell worker with only its Python execution replaced by argv capture."""
    with tempfile.TemporaryDirectory() as temporary:
        shim = Path(temporary) / 'python'
        shim.write_text('#!' + sys.executable + '\nimport json, os, sys\n'
                        'open(os.environ["CAPTURE_ARGV"], "w").write(json.dumps(sys.argv[1:]))\n')
        shim.chmod(0o755)
        capture = output / 'worker_argv.json'
        local = dict(env, PATH=temporary+os.pathsep+env['PATH'], CAPTURE_ARGV=str(capture))
        subprocess.run(['bash', 'launch_scripts/run_mc45_toggle_ablation.sh'],
                       cwd=ROOT, env=local, check=True, stdout=subprocess.DEVNULL)
        return json.loads(capture.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=ROOT/'results/rhs_current_clamp_study')
    parser.add_argument('--verify_only', action='store_true')
    parser.add_argument('--run_verified', action='store_true',
                        help='Require saved passing proofs, then run the full trials.')
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source/'run_config.json').read_text())
    corners = [f'{p}_V{v}_T{t}' for p in ('TT','FF','SS','FS','SF')
               for v in range(3) for t in range(3)]
    other = random.Random(config['base_seed']).choice([c for c in corners if c != 'FS_V2_T1'])
    chosen = ['FS_V2_T1', other]
    manifest = dict(source=str(source), corners=chosen, seed=config['base_seed'],
                    model=config['model_name'], model_dir=config['model_dir'], runs=[])
    from scripts import run_toggle_nonideality_ablation as driver
    from data_utils import MC45CornerData
    for mode in ('separate', 'combined'):
        for percentile in (95, 99):
            for corner in chosen:
                name = f'{mode}_{percentile}_{corner}'
                leaf = output/name
                leaf.mkdir(exist_ok=True)
                env = dict(os.environ)
                env.update(MODEL_NAME=config['model_name'], MODEL_DIR=config['model_dir'],
                    N_TRIALS='1', BASE_SEED=str(config['base_seed']), CORNER_IDS=corner,
                    OUTPUT_DIR=str(leaf), WEIGHT_QUANT_FACTOR_BITS='1', FULL_45_CORNER_C='500e-15',
                    MC_COUPLER_NONLINEAR_VARIATION_SOURCE='coupler_full_range',
                    MC_COUPLER_NONLINEAR_VARIATION_QUANTITY='conductance', MC_COUPLER_NOMINAL_R='50e3',
                    NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                    MC_RELU_MONTE_CARLO_SOURCE='0906_RELU_Voltage', ACTIVATION_CURVE_SHARING='per_spin',
                    V_DD='0.5', ONE_OVER_Q='5', TOGGLE_TIMING_MODE='fixed', TOGGLE_Y_TIME='10e-9',
                    Z_OVER_Y_TIME='1', INPUT_QUANT_BITS='12', CENTER_STUDENT_INPUT='false',
                    ENABLE_MEASURED_POOLING='true', ENOB='none', IS_SLURM='0',
                    RHS_CURRENT_SUMMARY=str(source/'total'/mode/'summary.md'),
                    RHS_CURRENT_BOUND_PERCENTILE=str(percentile),
                    RHS_CURRENT_AUDIT_PATH=str(leaf/'clamp_audit.jsonl'),
                    PYTHONUNBUFFERED='1')
                argv = capture_worker(env, leaf)
                old = sys.argv
                try:
                    sys.argv = argv
                    parsed = driver.parse_args()
                finally:
                    sys.argv = old
                assert parsed.rhs_current_summary == env['RHS_CURRENT_SUMMARY']
                assert parsed.rhs_current_bound_percentile == percentile
                catalog = MC45CornerData(parsed.mc_45_corner_dir,
                    spin_variation_source=parsed.mc_spin_variation_source,
                    dtc_pulse_width_variation_source=parsed.mc_dtc_pulse_width_variation_source,
                    relu_monte_carlo_source=parsed.mc_relu_monte_carlo_source,
                    coupler_nonlinear_variation_source=parsed.mc_coupler_nonlinear_variation_source,
                    coupler_nonlinear_variation_quantity=parsed.mc_coupler_nonlinear_variation_quantity,
                    coupler_nominal_R=parsed.mc_coupler_nominal_R)
                index = next(i for i,c in enumerate(catalog.corners) if c['id']==corner)
                command, _ = driver.build_corner_command(parsed, catalog.corners[index], index)
                assert command[command.index('--rhs_current_summary')+1] == env['RHS_CURRENT_SUMMARY']
                assert float(command[command.index('--rhs_current_bound_percentile')+1]) == percentile
                (leaf/'production_command.json').write_text(json.dumps(command, indent=2)+'\n')
                manifest['runs'].append(dict(name=name, env={k:v for k,v in env.items()
                    if k not in os.environ or os.environ[k]!=v}, command=command))
                (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                proof = Path(env['RHS_CURRENT_AUDIT_PATH']+'.proof.json')
                if not args.run_verified:
                    diagnostic = command.copy()
                    diagnostic[diagnostic.index('ode_inference.py')] = 'diagnostic_scripts/verify_rhs_current_clamp.py'
                    diagnostic[diagnostic.index('--test_bs')+1] = '2'
                    print('VERIFY '+name, flush=True)
                    with (leaf/'verification.log').open('w') as log:
                        subprocess.run(diagnostic, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                if not proof.exists() or not json.loads(proof.read_text())['passed']:
                    raise RuntimeError('Missing successful production proof: '+str(proof))
    write_report(output, manifest)
    if args.verify_only:
        print('All eight wiring/mapping proofs passed.', flush=True)
        return
    # All verification must pass before any full inference trial is launched.
    for run in manifest['runs']:
        leaf = output/run['name']
        print('EVALUATE '+run['name'], flush=True)
        env = dict(os.environ, **run['env'])
        with (leaf/'worker.log').open('w') as log:
            subprocess.run(['bash', 'launch_scripts/run_mc45_toggle_ablation.sh'],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_report(output, manifest)
    print('All eight production trials finished: '+str(output), flush=True)


if __name__ == '__main__':
    main()
