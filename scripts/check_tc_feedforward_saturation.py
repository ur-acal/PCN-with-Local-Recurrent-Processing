#!/usr/bin/env python3
"""Compare actual TC integration paths at the rails; no production changes.

Uses programmed 5-bit weights +/-1 and physical inputs within 0.1 V.
Static imperfections are disabled to isolate dynamic noise and projection.
The coupled comparison supplies one-shot with the sum of Dopri5's actual
noise increments, eliminating unrelated random-draw differences.
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from physical_feedforward_tc import TCPhysicalBasicBlock
from tc_nonidealities import TCNoiseLifecycle
from TorchDiffEqPack.odesolver.base import ODESolver


def main():
    torch.set_num_threads(1)
    torch.manual_seed(141)
    rows = []
    for target in (.08, .10, .12, .80):
        conv = nn.Conv2d(16, 2, 1, bias=False).double()
        with torch.no_grad():
            conv.weight[0].fill_(1.)
            conv.weight[1].fill_(-1.)
        a = TCPhysicalBasicBlock(
            conv, R=1e4, C=49e-15, v_dd=.1, one_shot_conv=True,
            toggle_timing_mode='derived', enable_coupler_noise=True,
            enable_summing_current_noise=True, coupler_noise_p=.6e-12,
            summing_current_p=.6e-12, tc_tol=1e-6).double().eval()
        b = copy.deepcopy(a)
        b.one_shot_conv = False
        x = torch.full((50000, 16, 1, 1), target/16, dtype=torch.float64)
        state = b._output_zeros(b.conv1, x)
        duration = a._stage_duration(x, 'z')
        _, eps = a._noise_context(a.conv1, x, state, 'z')
        increments, steps, step_rails = [], [], []
        original_update = ODESolver.addi_noisy_update_and_proj
        original_normal = ODESolver._randn_like
        current_h = [None]

        def update(solver, h, y_current):
            current_h[0] = h
            steps.append(float(h))
            out = original_update(solver, h, y_current)
            step_rails.append(float((out[0].abs() >= .1).double().mean()))
            return out

        def normal(solver, ref, branch=0):
            z = original_normal(solver, ref, branch)
            increments.append(current_h[0]**.5 * solver.eps * z)
            return z

        with torch.no_grad():
            one = a._run_stage(a.conv1, x, 'z')
            with patch.object(ODESolver, 'addi_noisy_update_and_proj', update), \
                 patch.object(ODESolver, '_randn_like', normal):
                dopri = b._run_stage(b.conv1, x, 'z')
            total_noise = torch.stack(increments).sum(0)
            with patch.object(TCNoiseLifecycle, 'normal',
                              lambda self, ref, branch: total_noise/(duration.sqrt()*eps)):
                coupled = a._run_stage(a.conv1, x, 'z')

        def stats(y):
            return dict(mean_V=y.mean((0, 2, 3)).tolist(),
                        std_V=y.std((0, 2, 3)).tolist(),
                        rail_fraction=(y.abs() >= .1).double().mean((0, 2, 3)).tolist())

        rows.append(dict(unclamped_target_V=target, input_V=target/16,
                         duration_s=float(duration), accepted_steps_s=steps,
                         rail_fraction_after_each_step=step_rails,
                         unbounded_noise_std_V=(duration.sqrt()*eps).flatten().tolist(),
                         one_shot=stats(one), dopri5=stats(dopri),
                         coupled_one_shot=stats(coupled),
                         coupled_max_abs_V=float((coupled-dopri).abs().max()),
                         coupled_rms_V=float((coupled-dopri).square().mean().sqrt())))
    result = dict(samples_per_sign=50000, dtype='float64', seed=4096,
                  reference_ASD_A_per_sqrt_Hz=.6e-12, reference_R_ohms=50000,
                  static_imperfections='off: isolate noise and rail integration', results=rows)
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
