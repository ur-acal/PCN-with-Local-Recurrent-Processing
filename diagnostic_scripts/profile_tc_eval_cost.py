"""Bounded, process-local profiling of the production TC evaluation entry.

The dense control skips expansion and selects the FT shared-curve approximation;
it is a forward-cost control, not an accuracy-equivalent evaluation or FT run.
No production implementation or checkpoint is changed.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import ode_inference as entry
from validation import MVMConv
from ode_pc import TogglePulseFFFB
from TorchDiffEqPack.odesolver.adaptive_grid_solver import Dopri5

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--batches', type=int, default=2)
p.add_argument('--optimization', choices=('reference','reuse','fused','both'), default='both')
p.add_argument('--timing-only', action='store_true', help='Disable nested profiling events for end-to-end timing.')
p.add_argument('--dense', action='store_true')
p.add_argument('--toggle-log', help='Replay the COMMAND line from a recorded corner log.')
p.add_argument('--cpu-unroll', action='store_true', help='Use the existing MNIST CPU cache-construction helper; timing excludes initialization.')
p.add_argument('--force-toggle-ff-capture', action='store_true', help='Diagnostic only: trigger the missing FF module hook during shape capture.')
a = p.parse_args()
Dopri5.tc_reuse_accepted_step = a.optimization in ('reuse','both')
MVMConv._tc_fused_edges = a.optimization in ('fused','both')
checkpoint = Path(a.checkpoint).resolve()
assert checkpoint.is_file(), checkpoint
name = checkpoint.parent.name
suffix = checkpoint.name[len(name)+1:-len('_ckpt.pth')]
out = Path(a.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, MODEL_NAME=name, MODEL_DIR=str(checkpoint.parent.parent),
           CKPT=suffix, TC_DRY_RUN='true', TEST_BS=str(a.batch_size),
           TC_MAX_EVAL_BATCHES=str(a.batches), N_TRIALS='1', TASK='cifar100',
           IMG_TYPE='rgb', TC_CONV_METHOD='shared' if a.dense else 'loop')
command = shlex.split(subprocess.check_output(
    ['bash', 'launch_scripts/run_tc_nonidealities.sh', 'eval'], env=env, text=True))
argv = command[command.index('ode_inference.py')+1:]
wrapper = 'QATTester1State' if suffix == 'last' else 'ODEWrapper1State'
argv += ['--ode_wrapper', wrapper,
         '--tc_metadata_path', str(out.with_suffix('.trial.jsonl'))]
if a.toggle_log:
    command = shlex.split(Path(a.toggle_log).read_text().splitlines()[0].removeprefix('COMMAND: '))
    argv = command[command.index('ode_inference.py')+1:]
    argv = [str(ROOT / v.split('PCN-with-Local-Recurrent-Processing/')[1])
            if 'PCN-with-Local-Recurrent-Processing/' in v else v for v in argv]
    if '--patched_nonlinearity_data' in argv:
        i=argv.index('--patched_nonlinearity_data')
        assert argv[i+1]=='false', 'Cannot drop an enabled historical feature.'
        del argv[i:i+2]  # Historical disabled flag removed from current parser.
    argv += ['--test_bs', str(a.batch_size), '--noisy_trials', '1']
report = dict(checkpoint=str(checkpoint), dense_control=a.dense, optimization=a.optimization,
              timing_only=a.timing_only, argv=argv, batches=[])
# Recovery checkpoints also contain CPU RNG state; CUDA map_location would
# incorrectly move that state before Generator.__setstate__. We only need the
# model payload; load this diagnostic source on CPU before normal model.to().
original_load = torch.load
def load(path, *args, **kwargs):
    if isinstance(path, (str, Path)) and Path(path).resolve() == checkpoint:
        kwargs['map_location'] = 'cpu'
    return original_load(path, *args, **kwargs)
torch.load = load
active = False
events = collections.defaultdict(list)
counts = collections.Counter()

def instrument(cls, name, label):
    if a.timing_only:
        return
    original = getattr(cls, name)
    def wrapped(self, *args, **kwargs):
        if not active:
            return original(self, *args, **kwargs)
        counts[label] += 1
        if label.endswith('lookup'):
            counts[label+'_edges'] += args[0].shape[0]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = original(self, *args, **kwargs)
        end.record()
        events[label].append((start, end))
        return result
    setattr(cls, name, wrapped)

instrument(MVMConv, '_get_gaussian_curve_R_eff', 'gaussian_lookup')
instrument(MVMConv, '_get_curve_bank_R_eff', 'empirical_lookup')
instrument(MVMConv, '_forward_pulse_per_edge', 'edge_convolution')
instrument(Dopri5, 'step', 'rk_attempt_or_replay')
original_pulse = TogglePulseFFFB._apply_pulse_module
def pulse(self, module, *args, **kwargs):
    if active:
        counts['pulse_'+('expanded' if isinstance(module,MVMConv) else 'dense')] += 1
    return original_pulse(self,module,*args,**kwargs)
TogglePulseFFFB._apply_pulse_module = pulse
if a.force_toggle_ff_capture:
    assert a.toggle_log
    original_init_z = TogglePulseFFFB.init_z
    def init_z(self,y):
        z = original_init_z(self,y)
        if isinstance(self.FFconv,torch.nn.Conv2d) and self.FFconv._forward_hooks:
            self.FFconv(z)  # Shape-only call triggers the already-installed capture hook.
        return z
    TogglePulseFFFB.init_z = init_z
    report['diagnostic_force_toggle_ff_capture'] = True

original_validator = entry.Validator
def validator(*args, **kwargs):
    if a.cpu_unroll:
        from mnist_train_eval.mnist_evaluate import cpu_unroll_convolution
        for block in kwargs['model'].PcConvs:
            block.unroll_convolution = cpu_unroll_convolution
    if a.dense:
        model = kwargs['model']
        class Holder: pass
        result = Holder()
        result.model = model
    else:
        begin = time.perf_counter()
        result = original_validator(*args, **kwargs)
        torch.cuda.synchronize()
        report['expansion_seconds'] = time.perf_counter()-begin
        model = result.model
    report['modules'] = [dict(name=n, edges=m.mat.values().numel(),
        grid_points=(m.nonlinear_R_curve_gaussian_v_grid.numel()
                     if m.nonlinear_R_curve_gaussian_v_grid is not None else
                     m.nonlinear_R_curve_bank_v_grid.shape[-1]),
        curve_bytes=(m.nonlinear_R_curve_gaussian_R_normalized.numel()*4
                     if m.nonlinear_R_curve_gaussian_R_normalized is not None else 0))
        for n,m in model.named_modules() if isinstance(m,MVMConv)]
    report['convolution_types'] = [dict(layer=i,FF=type(b.FFconv).__name__,FB=type(b.FBconv).__name__)
                                   for i,b in enumerate(model.PcConvs)]
    out.write_text(json.dumps(report, indent=2))
    from measured_activation import PiecewiseLinearActivation
    instrument(PiecewiseLinearActivation, 'forward', 'measured_relu')
    def before(module, inputs):
        global active, begun
        events.clear(); counts.clear()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begun = time.perf_counter(); active = True
    def after(module, inputs, output):
        global active
        torch.cuda.synchronize(); active = False
        row = dict(seconds=time.perf_counter()-begun, shape=list(inputs[0].shape),
            calls=dict(counts), inclusive_cuda_ms={k:sum(s.elapsed_time(e) for s,e in v)
                                                 for k,v in events.items()},
            allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated())
        row['logits'] = output.detach().cpu().tolist()
        report['batches'].append(row)
        out.write_text(json.dumps(report, indent=2))
        print('COST_PROFILE '+json.dumps(row), flush=True)
        if a.toggle_log and len(report['batches']) >= a.batches:
            raise ProfileDone()
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    return result
entry.Validator = validator
sys.argv = ['ode_inference.py', *argv]
class ProfileDone(Exception): pass
try:
    entry.run_ode_inference()
except ProfileDone:
    print('Bounded toggle profile finished; no full-dataset accuracy recorded.', flush=True)
