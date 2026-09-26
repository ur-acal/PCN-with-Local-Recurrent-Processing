"""Replay fixed-FF corner inputs with reference/lookup/full fusion; save logits.

Use the same --log, --batch-size and --batches for both invocations. This runs
bounded forwards, not a full accuracy trial. Existing TC profiler handles the
production entry point, seeds, timing and persistent JSON output.
"""
import argparse
import os
import json
from pathlib import Path
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validation import MVMConv

p = argparse.ArgumentParser()
p.add_argument('--log', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mode', choices=('reference','optimized','fused'), required=True,
               help='reference disables both; optimized is lookup-only; fused includes current/reduction.')
p.add_argument('--batch-size', type=int, default=128)
p.add_argument('--batches', type=int, default=3)
p.add_argument('--deterministic', action='store_true', help='Diagnostic-only deterministic PyTorch reduction; not a production default.')
p.add_argument('--verify-lookups', action='store_true', help='Compare every fused lookup against PyTorch on the actual model inputs (untimed audit).')
a = p.parse_args()
if a.verify_lookups and a.mode != 'optimized':
    p.error('--verify-lookups requires --mode optimized (lookup-only audit).')
if a.deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
lookup_checks = [0]
if a.verify_lookups:
    import torch
    import toggle_edge_inference as lookup
    original = lookup.empirical_resistance
    def checked(module, voltage, index):
        result = original(module, voltage, index)
        if result is not None:
            module._toggle_fused_lookup = False
            try:
                expected = module._get_curve_bank_R_eff(voltage, index)
            finally:
                del module._toggle_fused_lookup
            torch.testing.assert_close(result,expected,rtol=0,atol=0)
            lookup_checks[0] += 1
        return result
    lookup.empirical_resistance = checked
command = shlex.split(next(line for line in Path(a.log).read_text().splitlines()
                          if line.startswith('COMMAND:')).removeprefix('COMMAND:'))
def arg(name):
    return command[command.index(name)+1]
name = arg('--model_name')
checkpoint = ROOT / arg('--model_dir') / name / (name+'_'+arg('--ckpt')+'_ckpt.pth')
MVMConv._toggle_fused_lookup = a.mode == 'optimized'
MVMConv._toggle_fused_edges = a.mode == 'fused'
sys.argv = ['profile_tc_eval_cost.py', '--checkpoint', str(checkpoint),
            '--output', a.output, '--toggle-log', a.log,
            '--batch-size', str(a.batch_size), '--batches', str(a.batches),
            '--optimization', 'reference', '--timing-only']
runpy.run_path(str(ROOT/'diagnostic_scripts/profile_tc_eval_cost.py'),run_name='__main__')
report = json.loads(Path(a.output).read_text())
report.update(toggle_lookup_mode=a.mode, deterministic=a.deterministic,
              verified_exact_lookups=lookup_checks[0])
Path(a.output).write_text(json.dumps(report,indent=2))
