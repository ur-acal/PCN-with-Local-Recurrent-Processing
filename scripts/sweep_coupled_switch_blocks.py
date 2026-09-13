"""108-case coupled-block sweep; fixed full reference at Dopri5 1e-6."""
import functools
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'docs/switched_dynamics_diagnostic_snapshot'), str(ROOT)]
from debug_strang_real_block import build, CKPT
from ode_pc import ODEXInitFFFB
import switch

OUT = ROOT / 'results/coupled_switch_sweep_ref1e6'
METHODS = [('Jacobi', switch.ODEXInitFFFBPixelSwitchEfficient),
           ('Lie', switch.ODEXInitFFFBPixelSwitchExplicit),
           ('Strang', switch.ODEXInitFFFBPixelSwitchStrang)]


def save(rows, status):
    payload = {'status': status, 'expected_cases': 108, 'reference_tol': 1e-6,
               'error_domain': 'physical state before output quantization', 'rows': rows}
    tmp = OUT / 'results.json.tmp'
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(OUT / 'results.json')
    lines = ['# Coupled-block sweep: fixed full reference 1e-6', '',
             f'Status: {status}; {len(rows)}/108 cases complete.', '',
             'Same saved real-block input; float32, batch 1. Both rtol and atol',
             'equal the listed tolerance. Full unsplit ODEXInitFFFB reference',
             'is fixed at 1e-6. Errors are pre-output-quantization relative L2.', '']
    for m in (2, 3, 4):
        lines += [f'## m={m}', '',
                  '| Iterations | Mini tolerance | Jacobi | Lie | Strang |',
                  '|---:|---:|---:|---:|---:|']
        for n in (5, 8, 10, 20):
            for tol in (1e-6, 1e-7, 1e-8):
                cells = []
                for name, _ in METHODS:
                    found = [r for r in rows if (r['method'], r['m'], r['n_iters'], r['tol']) == (name, m, n, tol)]
                    cells.append(f"{found[0]['relative_error']:.8f}" if found else 'pending')
                lines.append(f'| {n} | {tol:.0e} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Raw diagnostics', '', '| Method | m | Iterations | Tolerance | Solves | NFE | Seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['m']} | {r['n_iters']} | {r['tol']:.0e} | {r['solves']} | {r['nfe']} | {r['seconds']:.3f} |")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Results already exist; refusing to overwrite a previous sweep.')
    rows = []
    save(rows, 'running')
    try:
        torch.manual_seed(4096)
        source = ROOT / 'results/coupled_switch_legacy.pt'
        x = torch.load(source, weights_only=True)['input'].cuda()
        manifest = {'checkpoint': CKPT, 'input_file': str(source),
                    'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'switch_sha256': hashlib.sha256((ROOT / 'switch.py').read_bytes()).hexdigest(),
                    'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                    'reference_tol': 1e-6, 'mini_tols': [1e-6, 1e-7, 1e-8]}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        net, wraps = build(ODEXInitFFFB, 1)
        b = net.PcConvs[0]
        b.option_aca.update(rtol=1e-6, atol=1e-6)
        xp = wraps[0].proj_fn(wraps[0].inp_scale*x)
        reference = switch.aca_ode_solve(b._make_ode_fn(xp), b.init_y(xp), b.option_aca)[-1]
        torch.save(reference.cpu(), OUT / 'reference.pt')
        print('REFERENCE READY: full ODEXInitFFFB, rtol=atol=1e-6', flush=True)
        del net, wraps, b
        original_solve = switch.aca_ode_solve
        for m in (2, 3, 4):
            for name, cls in METHODS:
                class Configured(cls):
                    def __init__(self, **kw):
                        super().__init__(block_size=m, **kw)
                net, wraps = build(Configured, 1)
                b = net.PcConvs[0]
                xp = wraps[0].proj_fn(wraps[0].inp_scale*x)
                for n in (5, 8, 10, 20):
                    for tol in (1e-6, 1e-7, 1e-8):
                        b.n_iters = n
                        b.option_aca.update(rtol=tol, atol=tol)
                        counts = {'solves': 0, 'nfe': 0}
                        def counted_solve(fn, *args, **kw):
                            counts['solves'] += 1
                            @functools.wraps(fn)
                            def rhs(*a, **k):
                                counts['nfe'] += 1
                                return fn(*a, **k)
                            return original_solve(rhs, *args, **kw)
                        switch.aca_ode_solve = counted_solve
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        try:
                            y = b._run_explicit_pixel_switch(xp)
                            torch.cuda.synchronize()
                        finally:
                            switch.aca_ode_solve = original_solve
                        expected = n * len(b.iter_spatial_blocks(*y.shape[-2:])) * (2 if name == 'Strang' else 1)
                        assert counts['solves'] == expected, (counts, expected)
                        assert torch.isfinite(y).all()
                        row = dict(method=name, m=m, n_iters=n, tol=tol,
                                   relative_error=float((y-reference).norm()/reference.norm()),
                                   seconds=time.perf_counter()-start, **counts)
                        rows.append(row)
                        save(rows, 'running')
                        print(f'CASE {len(rows)}/108 ' + json.dumps(row), flush=True)
                del net, wraps, b
        save(rows, 'complete')
        print('COMPLETE: 108/108 cases', flush=True)
    except BaseException:
        save(rows, 'failed')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
