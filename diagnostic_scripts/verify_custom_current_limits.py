#!/usr/bin/env python3
"""Convert both prior 95% tables, prove their mapping, rerun ONE production trial."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostic_scripts.make_current_limits import generate
from diagnostic_scripts.run_rhs_current_clamp_study import capture_worker
from rhs_current_clamp import read_summary


def report(output, study):
    leaf = output/'separate'
    before = json.loads((study/'separate_95_FS_V2_T1/clamp_audit.jsonl').read_text().splitlines()[-1])
    after = json.loads((leaf/'clamp_audit.jsonl').read_text().splitlines()[-1])
    assert before['accuracy']==after['accuracy'], (before['accuracy'],after['accuracy'])
    assert len(before['mappings'])==len(after['mappings'])
    differences = []
    for a,b in zip(before['mappings'],after['mappings']):
        for key in ('layer','stage','lower_A','upper_A','calls','samples'):
            assert a[key]==b[key], (key,a[key],b[key])
        differences.append(dict(layer=a['layer'],stage=a['stage'],
            lower_delta=b['clipped_lower']-a['clipped_lower'],
            upper_delta=b['clipped_upper']-a['clipped_upper']))
    discrepancy = sum(abs(r['lower_delta'])+abs(r['upper_delta']) for r in differences)
    samples = sum(r['samples'] for r in after['mappings'])
    lines = ['# Custom current limit verification', '',
        'Both separate and pooled 95% conversions preserve every bound exactly. '
        'Both passed independent two-image production mapping/update proofs.', '',
        f'One full test trial: separate95 FS_V2_T1. Original: {before["accuracy"]:.2f}%; '
        f'custom: {after["accuracy"]:.2f}%. All 41 mappings, bounds, call counts and sample counts match exactly.', '',
        f'Clipping counters are NOT bitwise identical: sum of absolute count differences = {discrepancy} '
        f'across {samples:,} sampled currents. This is consistent with numerical replay sensitivity near '
        'thresholds, but that cause has not been separately proven. No additional full trial was run.', '',
        'Proofs: [separate](separate/clamp_audit.jsonl.proof.json), '
        '[pooled](combined/clamp_audit.jsonl.proof.json). '
        'Full production audit: [JSONL](separate/clamp_audit.jsonl).', '',
        '| Layer | Stage | Lower count difference | Upper count difference |',
        '|---|---|---:|---:|']
    lines += [f"| {r['layer']} | {r['stage']} | {r['lower_delta']} | {r['upper_delta']} |" for r in differences]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Mapping/accuracy checks passed; clipping-counter differences reported: '+str(output/'README.md'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='results/rhs_current_clamp_study')
    parser.add_argument('--output_dir', default='results/custom_current_limits_verification')
    parser.add_argument('--report_only', action='store_true', help='Compare existing audits; no inference.')
    args = parser.parse_args()
    study, output = Path(args.study).resolve(), Path(args.output_dir).resolve()
    if args.report_only:
        report(output,study)
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((study/'manifest.json').read_text())
    runs = []
    for mode in ('separate','combined'):
        old = next(r for r in manifest['runs'] if r['name']==f'{mode}_95_FS_V2_T1')
        path = generate(SimpleNamespace(num_layers=20, pool_positions=[9], granularity='layer',
            branches='pooled' if mode=='combined' else mode, no_final_linear=False,
            summary=old['env']['RHS_CURRENT_SUMMARY'], percentile=95., bound=None,
            name=None, output_dir=ROOT/'hardware_data/summing_current_limit', overwrite=False))
        _, before, _ = read_summary(old['env']['RHS_CURRENT_SUMMARY'],95)
        _, after, _ = read_summary(path,95)
        assert before.keys()==after.keys()
        for key in before:
            for bound in ('lower_A','upper_A'):
                assert before[key][bound]==after[key][bound], (key,bound)
        leaf = output/mode
        leaf.mkdir()
        env = dict(os.environ, **old['env'])
        env.update(OUTPUT_DIR=str(leaf), RHS_CURRENT_SUMMARY=str(path),
                   RHS_CURRENT_AUDIT_PATH=str(leaf/'clamp_audit.jsonl'))
        captured = capture_worker(env,leaf)
        assert captured[captured.index('--rhs_current_summary')+1]==str(path)
        command = old['command'].copy()
        command[0] = sys.executable
        command[command.index('--rhs_current_summary')+1] = str(path)
        command[command.index('--rhs_current_audit_path')+1] = env['RHS_CURRENT_AUDIT_PATH']
        command[command.index('ode_inference.py')] = 'diagnostic_scripts/verify_rhs_current_clamp.py'
        command[command.index('--test_bs')+1] = '2'
        with (leaf/'verification.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        runs.append((old,leaf,env))
        print(f'{mode}: exact conversion and production batch proof passed',flush=True)
    old,leaf,env = runs[0]
    print('Running one full separate95 FS_V2_T1 production trial',flush=True)
    with (leaf/'worker.log').open('w') as log:
        subprocess.run(['bash','launch_scripts/run_mc45_toggle_ablation.sh'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    report(output,study)


if __name__ == '__main__':
    main()
