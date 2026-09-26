"""Run production TC evaluation with fail-fast expansion checks; no math overrides."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
import ode_inference as entry
from validation import MVMConv, cached_unrolled_weight_matches
from mnist_train_eval.mnist_evaluate import cpu_unroll_convolution

p = argparse.ArgumentParser(__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--max-batches', type=int, default=0)
a = p.parse_args()
checkpoint = Path(a.checkpoint).resolve()
if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
name = checkpoint.parent.name
if checkpoint.name != name+'_last_ckpt.pth':
    raise ValueError('This audit expects a flattened last checkpoint, not full_param.')
out = Path(a.output_dir).resolve()
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, MODEL_NAME=name, MODEL_DIR=str(checkpoint.parent.parent),
    CKPT='last', TC_STATE='1', TC_DRY_RUN='true', TASK='cifar100', IMG_TYPE='rgb',
    TC_EMPIRICAL_CURVE_BANK='', TEST_BS=str(a.batch_size), N_TRIALS='1',
    TC_MAX_EVAL_BATCHES=str(a.max_batches), TC_METADATA_PATH=str(out/'trial.jsonl'))
command = shlex.split(subprocess.check_output(
    ['bash','launch_scripts/run_tc_nonidealities.sh','eval'], cwd=ROOT, env=env, text=True))
argv = command[command.index('ode_inference.py')+1:]
argv += ['--ode_wrapper','QATTester1State']
(out/'resolved.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),argv=argv),indent=2))
original = entry.Validator

def validator(*args, **kwargs):
    model = kwargs['model']
    dense, shapes, handles = {}, {}, []
    for idx, block in enumerate(model.PcConvs):
        block.unroll_convolution = cpu_unroll_convolution  # Initialization only.
        for role in ('FFconv','FBconv'):
            conv = getattr(block, role)
            if not isinstance(conv,nn.Conv2d): raise TypeError((idx,role,type(conv)))
            key = (idx,role)
            dense[key] = SimpleNamespace(weight=conv.weight.detach().cpu(),
                groups=conv.groups,dilation=conv.dilation,stride=conv.stride,
                padding=conv.padding,kernel_size=conv.kernel_size,
                in_channels=conv.in_channels,out_channels=conv.out_channels)
            def capture(module, inputs, key=key): shapes[key] = tuple(inputs[0].shape[1:])
            handles.append(conv.register_forward_pre_hook(capture))
    try:
        result = original(*args, **kwargs)
    finally:
        for handle in handles: handle.remove()
    rows = []
    with torch.no_grad():
        for idx, block in enumerate(result.model.PcConvs):
            for role in ('FFconv','FBconv'):
                m = getattr(block,role)
                if not isinstance(m,MVMConv): raise RuntimeError(f'Unexpanded {idx}:{role}')
                if m.nonlinear_R_curve_sampling != 'multivariate_gaussian':
                    raise RuntimeError('Expected Gaussian curves')
                curves = m.nonlinear_R_curve_gaussian_R_normalized
                if (curves is None or curves.shape[0] != int((m.mat.values()!=0).sum())
                        or m.nonlinear_R_curve_sharing != 'per_coupler'):
                    raise RuntimeError('Per-physical-coupler curves missing')
                if not cached_unrolled_weight_matches(
                        dict(weight=m.mat.cpu(),meta=m.meta),dense[(idx,role)],shapes[(idx,role)]):
                    raise RuntimeError(f'Expanded weights mismatch at {idx}:{role}')
                torch.testing.assert_close(m._tc_signed_mat.values(),m.mat.values().sign(),rtol=0,atol=0)
                rows.append(dict(layer=idx,role=role,sites=m.mat.values().numel(),
                    seed=m.nonlinear_R_curve_seed,curve_shape=list(curves.shape)))
    if len({r['seed'] for r in rows}) != len(rows):
        raise RuntimeError('FF/FB curve seeds are not independent')
    (out/'expansion_audit.json').write_text(json.dumps(rows,indent=2))
    print(f'AUDIT PASSED: {len(rows)} FF/FB matrices expanded; weights match; '
          'independent Gaussian curves per physical coupler.',flush=True)
    return result

entry.Validator = validator
sys.argv = ['ode_inference.py', *argv]
entry.run_ode_inference()
print('TC evaluation completed successfully.',flush=True)
