"""Record operator parity and exact pinned-checkpoint availability, not FT speed.

This never launches training or substitutes a checkpoint. Complete-model parity
and timing remain blocked until the pinned pipeline's PT artifact is available.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pinned_pretrain_root():
    """Resolve the literal PT source of the current pinned toggle FT command."""
    text = (ROOT/'docs/current_pcn_reference.md').read_text()
    section = text.split('## Main-model SLURM training:', 1)[1].split('### Provisional', 1)[0]
    for line in section.splitlines():
        assignment = line.strip().rstrip('\\').strip()
        if assignment.startswith('PRETRAIN_SAVE_PATH='):
            values = shlex.split(assignment.split('=', 1)[1])
            if len(values) != 1 or '$' in values[0]:
                raise ValueError('Pinned PT source must be a literal path.')
            return ROOT / values[0]
    raise ValueError('Pinned toggle command does not specify PRETRAIN_SAVE_PATH.')


def difference(actual, expected):
    import torch
    error = (actual-expected).double()
    return dict(max_abs=error.abs().max().item(),
        relative_l2=(error.square().sum()/expected.double().square().sum().clamp_min(1e-30)).sqrt().item(),
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        elementwise_pass=bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)))


def main():
    import torch
    from torch import nn
    from tc_shared_correction import toggle_empirical_correction, triton
    from utils import load_mc_res_training_curve_bank
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    source = pinned_pretrain_root()
    candidates = sorted(str(p) for p in source.glob('TIMMPCNet*/*_best_ckpt.pth'))
    result = dict(
        thresholds=dict(output_atol=1e-5, output_rtol=1e-4,
            operator_gradient_atol=1e-5, operator_gradient_rtol=1e-4,
            full_gradient_max_abs=1e-3, full_gradient_relative_l2=1e-3),
        control='FUSE_TOGGLE_NONLINEAR_R_INPUT', default=False,
        pinned_checkpoint=dict(root=str(source), candidates=candidates,
            provenance='docs/current_pcn_reference.md: explicit PRETRAIN_SAVE_PATH; '
            'slurm_search_config.sh -> run_kdcrd_then_ft.sbatch: ft_and_eval, best PT -> FT'),
        full_model_parity=dict(status='not_run', reason='Exact pinned pretrained checkpoint absent'
            if not candidates else 'Run real FT-loop validation separately; operator check is not model validation'),
        training_benchmark=dict(status='not_run', median_seconds=None,
            peak_allocated_bytes=None, peak_reserved_bytes=None,
            reason='No exact pinned pretrained checkpoint' if not candidates else 'Not an operator benchmark'),
        operator=[])
    if torch.cuda.is_available() and triton is not None:
        result['gpu'] = torch.cuda.get_device_name()
        bank = load_mc_res_training_curve_bank(
            ROOT/'hardware_data/mc_45_corners/coupler_full_range',
            mode='exact_curve', quantity='conductance', dtype=torch.float32, device='cuda')
        package = dict(bank, R=5e4, proj_fn=nn.Hardtanh(-.5, .5))
        for index in (0, 99, 100, 1800, 4499):
            length = int(bank['lengths'][index])
            grid = bank['v_grid'][index, :length]
            left = bank['R_left'][index, :length-1]
            slope = bank['R_slope'][index, :length-1]
            x = torch.cat((grid, grid[:-1]+grid.diff()/2,
                grid.new_tensor([-1., -.50001, -.5, -.49999, .49999, .5, .50001, 1.]))).requires_grad_()
            query = package['proj_fn'](x).clamp(min=grid[0], max=grid[-1])
            interval = (torch.bucketize(query.contiguous(), grid.contiguous())-1).clamp(0, length-2)
            R_eff = left[interval]+slope[interval]*(query-grid[interval])
            reference = x*x.new_tensor(package['R'])/R_eff
            actual = toggle_empirical_correction(x, package, index, {})
            if actual is None:
                raise RuntimeError('CUDA case unexpectedly fell back')
            gradient = difference(torch.autograd.grad(actual.sum(), x)[0],
                                  torch.autograd.grad(reference.sum(), x)[0])
            row = dict(curve_index=index, outputs=difference(actual, reference), input_gradients=gradient)
            result['operator'].append(row)
    else:
        result['operator_status'] = 'CUDA/Triton unavailable'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if any(not row[category]['elementwise_pass'] or not row[category]['finite']
           for row in result['operator'] for category in ('outputs', 'input_gradients')):
        raise SystemExit('Operator parity failed; see report')


if __name__ == '__main__':
    main()
