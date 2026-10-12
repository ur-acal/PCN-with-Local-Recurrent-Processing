"""Instrumentation around the existing pipeline entry, not a replacement evaluator.

Usage: audit_pipeline_logits.py --output FILE [--reference] -- -m MODULE ARGS
Reference disables RK reuse/fusion and restores the old full Gaussian table.
All model/data/solver/noise construction remains in the selected entry point.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from validation import MVMConv, Validator
from feedforward_validation import FeedForwardCNNValidator
from physical_feedforward import iter_physical_blocks
from measured_activation import PiecewiseLinearActivation
from measured_pooling import MeasuredAvgPool2d
from tc_nonidealities import TCNoiseLifecycle
from TorchDiffEqPack.odesolver.adaptive_grid_solver import Dopri5
import tc_edge_inference

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', required=True)
p.add_argument('--reference', action='store_true')
p.add_argument('--synthetic-input-size', type=int, default=0,
               help='Use deterministic square RGB inputs of this size.')
p.add_argument('entry', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.entry
if command[0] == '--': command = command[1:]
if command[0] == '-u': command = command[1:]
assert command[0] == '-m'
torch.set_num_threads(2)
if a.reference:
    Dopri5.reuse_accepted_step_inference = False
    MVMConv._tc_fused_edges = False
report = dict(reference=a.reference, command=command, layers={}, batches=[], calls={})
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
active = False
counts = collections.Counter()
forward_timer = {}

def save():
    report['calls'] = dict(counts)
    out.write_text(json.dumps(report, indent=2, default=str))

original_sample = MVMConv.sample_nonlinear_R_gaussian_curves
def sample(m):
    value = original_sample(m)
    if a.reference and getattr(m, '_tc_curve_row_index', None) is not None:
        rows = m._tc_curve_row_index
        full = value.new_full((rows.numel(), value.shape[1]), float('inf'))
        for start in range(0, rows.numel(), 65536):
            r = rows[start:start+65536]
            valid = r >= 0
            full[start:start+65536][valid] = value[r[valid]]
        m._tc_curve_row_index = None
        m.nonlinear_R_curve_gaussian_R_normalized = full
        return full
    return value
MVMConv.sample_nonlinear_R_gaussian_curves = sample

original_fused = tc_edge_inference.gaussian_edge_forward
def fused(m, *args):
    result = original_fused(m, *args)
    if active:
        counts['fused:'+m._audit_name] += int(result is not None)
        counts['fallback:'+m._audit_name] += int(result is None)
    return result
tc_edge_inference.gaussian_edge_forward = fused

def count_method(cls, name):
    original = getattr(cls, name)
    def call(self, *args, **kwargs):
        if active: counts[cls.__name__+'.'+name] += 1
        return original(self, *args, **kwargs)
    setattr(cls, name, call)
for name in ('normal', 'fb_current', 'accepted'):
    count_method(TCNoiseLifecycle, name)
count_method(Dopri5, 'step')
count_method(nn.Hardtanh, 'forward')

original_noise = TCNoiseLifecycle._sample
noise_hash = hashlib.sha256()
def noise(self, key, ref, coefficients):
    fresh = (self.index, key) not in self.tape
    result = original_noise(self, key, ref, coefficients)
    if active and fresh:
        assert all(torch.isfinite(c).all() for c in coefficients)
        assert all(torch.count_nonzero(c) for c in coefficients), 'Missing noise source'
        noise_hash.update(result.detach().cpu().contiguous().numpy().tobytes())
        counts['noise_draw:'+str(key)] += 1
    return result
TCNoiseLifecycle._sample = noise

def attach(model, expected):
    modules = dict(model.named_modules())
    private_expected = set()
    head_name = 'linear' if hasattr(model, 'linear') else 'fc'
    head = getattr(model, head_name, None)
    private_head = getattr(head, '_expanded_module', None)
    if isinstance(private_head, MVMConv):
        private_name = head_name + '._expanded_module'
        modules[private_name] = private_head
        expected = [*expected, private_name]
        private_expected.add(private_name)
    assert expected
    blocks = list(model.PcConvs) if hasattr(model, 'PcConvs') else list(iter_physical_blocks(model))
    for b in blocks:
        cfg = getattr(b, '_tc_noise_cfg', vars(b))
        for flag in ('enable_spin_variation', 'enable_summing_current_noise', 'enable_coupler_noise'):
            assert cfg[flag], ('Missing nonideality', flag)
        assert b.q_hi == 15, ('Not signed 5-bit', b.q_hi)
        assert math.isfinite(float(b.v_dd)) and float(b.v_dd) > 0, (
            'Invalid supply voltage', b.v_dd)
    report['hardware'] = [dict(noise=getattr(b, '_tc_noise_cfg', None),
        v_dd=b.v_dd, q_hi=b.q_hi, R=b.R, C=getattr(b, 'C', None)) for b in blocks]
    for name in expected:
        m = modules[name]
        assert isinstance(m, MVMConv), ('Not expanded', name, type(m))
        m._audit_name = name
        curves = m.nonlinear_R_curve_gaussian_R_normalized
        if curves is None:
            curves = m._gaussian_curve_cpu
        assert curves is not None and m.nonlinear_R_curve_sharing == 'per_coupler'
        nz = m.mat.values() != 0
        compact = getattr(m, '_tc_curve_row_index', None) is not None
        assert curves.shape[0] == (int(nz.sum()) if compact else nz.numel())
        digest = hashlib.sha256()
        # Hash active curves in physical order in both storage layouts.
        if not compact:
            for i in range(0, nz.numel(), 65536):
                mask = nz[i:i+65536].to(curves.device)
                chunk = curves[i:i+65536][mask].detach().cpu().contiguous()
                digest.update(memoryview(chunk.numpy()))
        else:
            digest.update(memoryview(curves.detach().cpu().contiguous().numpy()))
        report['layers'][name] = dict(sites=nz.numel(), active=int(nz.sum()),
            curve_shape=list(curves.shape), curves_sha256=digest.hexdigest(),
            seed=m.nonlinear_R_curve_seed)
    for name, m in modules.items():
        if isinstance(m, (MVMConv, PiecewiseLinearActivation, MeasuredAvgPool2d, nn.Hardtanh)):
            def called(module, inputs, name=name):
                if active:
                    if name not in private_expected:
                        assert not module.training, ('Training mode during eval', name)
                    counts['module:'+name] += 1
            m.register_forward_pre_hook(called)
    def before(module, inputs):
        global active
        active = True
        if inputs[0].is_cuda:
            torch.cuda.reset_peak_memory_stats(inputs[0].device)
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            forward_timer['start'] = start
        else:
            forward_timer['start'] = time.perf_counter()
    def after(module, inputs, output):
        global active
        active = False
        assert torch.isfinite(output).all()
        assert all(
            counts['module:'+n] + counts['fused:'+n] + counts['fallback:'+n] > 0
            for n in expected)
        if output.is_cuda and not a.reference:
            for name in expected:
                module = modules[name]
                fused = counts['fused:'+name]
                fallback = counts['fallback:'+name]
                if (getattr(module, '_tc_curve_package', None) is not None
                        and name not in private_expected):
                    assert fused > 0, ('TC fused path not adopted', name)
                    assert fallback == 0, ('Unexpected TC CUDA fallback', name)
                else:
                    assert fused + fallback > 0, (
                        'Expanded nonlinear-R path not adopted', name)
        static = hashlib.sha256()
        for b in blocks:
            spin = getattr(b, '_tc_spin_state', b)
            for stage in ('y', 'z'):
                factor = getattr(spin, '_spin_factor_'+stage, None)
                # CNN stem has only one destination state.
                if stage == 'y' and hasattr(b, 'conv2') and b.conv2 is None:
                    continue
                assert factor is not None, ('Spin not applied', stage)
                static.update(factor.detach().cpu().contiguous().numpy().tobytes())
        for n,m in modules.items():
            if isinstance(m, PiecewiseLinearActivation):
                assert counts['module:'+n] > 0
                assert m.curve_sharing == 'per_spin' and m._sampled_curve_indices is not None
                static.update(m._sampled_curve_indices.cpu().contiguous().numpy().tobytes())
            if isinstance(m, MeasuredAvgPool2d):
                assert counts['module:'+n] > 0
                assert m.enable_nonideality and m.curve_gaussian
                assert m._gaussian_curve_samples
                for v in m._gaussian_curve_samples.values():
                    if torch.is_tensor(v): static.update(v.cpu().contiguous().numpy().tobytes())
        if output.is_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            seconds = forward_timer['start'].elapsed_time(end) / 1000.
            peak = torch.cuda.max_memory_allocated(output.device)
        else:
            seconds = time.perf_counter() - forward_timer['start']
            peak = 0
        row = dict(device=str(output.device), logits=output.detach().cpu().tolist(),
                   seconds=seconds, peak_allocated=peak,
                   noise_sha256=noise_hash.hexdigest(), static_sha256=static.hexdigest())
        if report['batches']:
            assert row['static_sha256'] == report['batches'][0]['static_sha256'], 'Static defects resampled'
        report['batches'].append(row)
        save()
        print('FULL_MODEL_AUDIT '+json.dumps(row), flush=True)
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    report['components'] = {n:dict(type=type(m).__name__,
        sharing=getattr(m, 'curve_sharing', None),
        nonideal=getattr(m, 'enable_nonideality', None)) for n,m in modules.items()
        if isinstance(m, (PiecewiseLinearActivation, MeasuredAvgPool2d, nn.Hardtanh))}
    save()

def instrument_validator(cls):
    original = cls.__init__
    def init(self, *args, **kwargs):
        model = kwargs.get('model', args[0] if args else None)
        if hasattr(model, 'PcConvs'):
            expected = [f'PcConvs.{i}.{role}' for i,b in enumerate(model.PcConvs)
                        for role in ('FFconv', 'FBconv')]
        else:
            blocks = {id(b) for b in iter_physical_blocks(model)}
            expected = [f'{n}.{role}' for n,b in model.named_modules() if id(b) in blocks
                        for role in ('conv1','conv2') if getattr(b, role) is not None]
        original(self, *args, **kwargs)
        attach(model, expected)
    cls.__init__ = init
instrument_validator(Validator)
instrument_validator(FeedForwardCNNValidator)
if a.synthetic_input_size:
    from torch.utils.data import DataLoader, TensorDataset
    import inference_utils

    def deterministic_test_data(test_bs=1, task='cifar10', **_kwargs):
        classes = 100 if task == 'cifar100' else 10
        count = max(3, int(test_bs) * 3)
        values = torch.linspace(
            -1., 1., count * 3 * a.synthetic_input_size ** 2,
            dtype=torch.float32).reshape(
                count, 3, a.synthetic_input_size, a.synthetic_input_size)
        labels = torch.arange(count, dtype=torch.long) % classes
        return DataLoader(TensorDataset(values, labels), batch_size=test_bs,
                          shuffle=False, num_workers=0)

    inference_utils.get_test_data = deterministic_test_data
sys.argv = [command[1], *command[2:]]
try:
    runpy.run_module(command[1], run_name='__main__')
finally:
    save()
