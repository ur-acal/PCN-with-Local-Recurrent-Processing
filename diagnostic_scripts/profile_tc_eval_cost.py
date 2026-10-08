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
p.add_argument('--toggle-nonlinear-r-table',
               help='Diagnostic override for only the replayed nonlinear-R curve bank.')
p.add_argument('--toggle-activation-curve-path',
               help='Diagnostic override for only the replayed measured-ReLU curve bank.')
p.add_argument('--toggle-pooling-curve-path',
               help='Hold measured pooling on this curve bank during a toggle replay.')
p.add_argument('--omit-logits', action='store_true',
               help='Keep bounded accuracy data without saving or printing full logits.')
p.add_argument('--cpu-unroll', action='store_true', help='Use the existing MNIST CPU cache-construction helper; timing excludes initialization.')
p.add_argument('--force-toggle-ff-capture', action='store_true', help='Diagnostic only: trigger the missing FF module hook during shape capture.')
a = p.parse_args()
Dopri5.reuse_accepted_step_inference = a.optimization in ('reuse','both')
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
    if a.toggle_nonlinear_r_table:
        i = argv.index('--nonlinear_R_table')
        argv[i+1] = str(Path(a.toggle_nonlinear_r_table).resolve())
    if a.toggle_activation_curve_path:
        i = argv.index('--activation_curve_path')
        argv[i+1] = str(Path(a.toggle_activation_curve_path).resolve())
    argv += ['--test_bs', str(a.batch_size), '--noisy_trials', '1']
report = dict(checkpoint=str(checkpoint), dense_control=a.dense, optimization=a.optimization,
              timing_only=a.timing_only, argv=argv, batches=[],
              toggle_nonlinear_r_table=a.toggle_nonlinear_r_table,
              toggle_activation_curve_path=a.toggle_activation_curve_path,
              toggle_pooling_curve_path=a.toggle_pooling_curve_path)

if a.toggle_pooling_curve_path:
    original_configure_measured_pooling = entry.configure_measured_pooling
    def configure_measured_pooling(*args, **kwargs):
        kwargs['curve_path'] = str(Path(a.toggle_pooling_curve_path).resolve())
        kwargs['curve_gaussian'] = None
        return original_configure_measured_pooling(*args, **kwargs)
    entry.configure_measured_pooling = configure_measured_pooling
current_targets = None
total_correct = 0
total_examples = 0

# Keep the production data loader and ordering, but retain the current targets so
# the bounded toggle replay can report an exact cumulative accuracy rather than
# only timing and logits.
original_get_test_data = entry.get_test_data
def get_test_data(*args, **kwargs):
    loader = original_get_test_data(*args, **kwargs)
    class TrackingLoader:
        def __len__(self):
            return len(loader)
        def __iter__(self):
            global current_targets
            for inputs, targets in loader:
                current_targets = targets
                yield inputs, targets
        def __getattr__(self, name):
            return getattr(loader, name)
    return TrackingLoader()
entry.get_test_data = get_test_data
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
pulse_labels = {}

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
        label = pulse_labels.get(id(module))
        if label is not None:
            counts['pulse_'+label] += 1
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
    remaining_dense = []
    unroll_audit = []
    for layer_idx, block in enumerate(model.PcConvs):
        for module_name, module in block.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                remaining_dense.append(
                    'PcConvs.{}.{}'.format(layer_idx, module_name))
        for stage in ('FFconv', 'FBconv'):
            module = getattr(block, stage)
            label = 'layer_{:02d}_{}'.format(layer_idx, stage[:2])
            pulse_labels[id(module)] = label
            values = module.mat.values() if isinstance(module, MVMConv) else None
            assignment = getattr(module, 'nonlinear_R_curve_assignment', None)
            unroll_audit.append(dict(
                layer=layer_idx, stage=stage, module_type=type(module).__name__,
                matrix_layout=(str(module.mat.layout) if isinstance(module, MVMConv)
                               else None),
                matrix_shape=(list(module.mat.shape) if isinstance(module, MVMConv)
                              else None),
                stored_edges=(values.numel() if values is not None else None),
                clean_values=(getattr(module, 'clean_mat_values', None).numel()
                              if getattr(module, 'clean_mat_values', None) is not None
                              else None),
                curve_assignments=(assignment.numel()
                                   if assignment is not None else None),
                dtc_block_ids=(getattr(module, 'dtc_block_ids', None).numel()
                               if getattr(module, 'dtc_block_ids', None) is not None
                               else None),
                dtc_output_ids=(getattr(module, 'dtc_output_ids', None).numel()
                                if getattr(module, 'dtc_output_ids', None) is not None
                                else None),
                toggle_pulse_edges=bool(
                    getattr(module, '_toggle_pulse_edges', False))))
    report['pc_unroll_audit'] = unroll_audit
    report['remaining_dense_pc_convs'] = remaining_dense
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
        global active, total_correct, total_examples
        torch.cuda.synchronize(); active = False
        assert current_targets is not None
        predicted = output.detach().argmax(dim=1).cpu()
        batch_targets = current_targets.detach().cpu()
        batch_correct = int((predicted == batch_targets).sum().item())
        total_correct += batch_correct
        total_examples += int(batch_targets.numel())
        row = dict(seconds=time.perf_counter()-begun, shape=list(inputs[0].shape),
            calls=dict(counts), inclusive_cuda_ms={k:sum(s.elapsed_time(e) for s,e in v)
                                                 for k,v in events.items()},
            allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated(),
            batch_correct=batch_correct, batch_examples=int(batch_targets.numel()),
            cumulative_correct=total_correct, cumulative_examples=total_examples,
            cumulative_accuracy=100.0*total_correct/total_examples)
        if not a.omit_logits:
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
