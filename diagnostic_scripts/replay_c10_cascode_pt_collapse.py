"""Instrument the real PT entry point without changing its optimization recipe.

The real worker supplies every training argument. Keep num_epochs=300 for the
original scheduler, but stop after an epoch-boundary recovery save. Diagnostics
do not change losses, activations, RNG seeds, clipping, or optimizer updates.
Extra backward probes can change execution timing/floating-point ordering;
bitwise reproduction is checked against the original log, not assumed.
"""
import argparse
import collections
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import train_ode_cifar as training
import trainer as trainer_base
import training_recovery
from trainer_timm import TrainerCiFarTimmStyleFeatureKD
from measured_activation import PiecewiseLinearActivation
from training_recovery import latest_path


class DiagnosticStop(Exception):
    pass


def scalar(x):
    x = float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
    return x if math.isfinite(x) else str(x)


def tensor_summary(x):
    x = x.detach().float().reshape(-1)
    return dict(norm=scalar(x.norm()), max_abs=scalar(x.abs().max()),
                finite_fraction=scalar(torch.isfinite(x).float().mean()))


def group_name(name):
    if name.startswith('PcConvs.'):
        return '.'.join(name.split('.')[:3])
    if name.startswith('aux.'):
        return 'srrl_connector'
    return name.split('.')[0]


def gradient_report(names, values):
    groups = collections.defaultdict(list)
    for name, value in zip(names, values):
        if value is not None:
            groups[group_name(name)].append(value.detach().float().square().sum())
    result = {key: scalar(torch.stack(v).sum().sqrt()) for key, v in groups.items()}
    if groups:
        result['global'] = scalar(torch.stack([x for v in groups.values() for x in v]).sum().sqrt())
    return result


class Recorder:
    def __init__(self, out, interval=10):
        self.out, self.interval = out, interval
        self.epoch = self.batch = 0
        self.active = False
        self.phase = 'total'
        self.owner = None
        self.layers = collections.defaultdict(list)
        self.activation_grads = collections.defaultdict(list)
        self.stat = self.pred = None
        self.step_before = None
        self.pending = None
        self.handles = []
        self.rows = (out / 'steps.jsonl').open('x', buffering=1)

    def selected(self):
        # Measure all minibatches through the expected first instability.
        return self.active and self.batch > 0 and (
            18 <= self.epoch <= 25 or (self.batch - 1) % self.interval == 0)

    def activation_hook(self, name, mod, inputs, output):
        if not self.selected():
            return
        x = inputs[0].detach().reshape(-1)
        stride = max(1, x.numel() // 2048)
        x = x[::stride][:2048].float()
        y = output.detach().reshape(-1)[::stride][:2048].float()
        q = mod._coordinate_pullback_scale
        if q is None:
            return
        z = x * q / mod.voltage_scale
        vin, curve = mod.vin, mod.curves[mod.active_corner]
        ii = torch.searchsorted(vin, z.contiguous(), right=True).clamp(1, len(vin)-1)
        slope = (curve[ii] - curve[ii-1]) / (vin[ii] - vin[ii-1])
        slope = torch.where((z < vin[0]) | (z > vin[-1]), 0., slope)
        self.layers[name].append(torch.stack([
            x.square().mean().sqrt(), x.abs().max(), y.square().mean().sqrt(),
            (y == 0).float().mean(), (slope.abs() < .1).float().mean(),
            ((z > 0) & (slope.abs() < .1)).float().mean(),
            ((z < vin[0]) | (z > vin[-1])).float().mean(),
            slope.abs().mean(), torch.isfinite(x).float().mean()]))
        if output.requires_grad:
            def grad_hook(g):
                v = g.detach().reshape(-1)[::stride][:2048].float()
                self.activation_grads[(name, self.phase)].append(torch.stack([
                    v.square().mean().sqrt(), v.abs().max(), (v == 0).float().mean()]))
            output.register_hook(grad_hook)

    def attach(self, owner):
        self.owner = owner
        for name, mod in owner.model.named_modules():
            if isinstance(mod, PiecewiseLinearActivation):
                self.handles.append(mod.register_forward_hook(
                    lambda m, a, y, name=name: self.activation_hook(name, m, a, y)))
        kd = owner._feature_kd_loss
        self.handles.append(kd.stat_loss_fn.register_forward_hook(
            lambda m, a, y: setattr(self, 'stat', y) if self.selected() else None))
        def kd_hook(m, args, kwargs, result):
            if self.selected():
                # New node over the SAME forward tensors; no second forward,
                # connector BN update, dropout realization, or teacher pass.
                self.pred = F.mse_loss(result[1]['srrl_logits'], kwargs['teacher_logits'].detach())
        self.handles.append(kd.register_forward_hook(kd_hook, with_kwargs=True))
        self.handles.append(owner.optimizer.register_step_pre_hook(self.before_step))
        self.handles.append(owner.optimizer.register_step_post_hook(self.after_step))

    def parameters(self):
        values = list(self.owner.model.named_parameters())
        values += [('aux.'+n, p) for n, p in self.owner._feature_kd_loss.named_parameters()]
        return [(n, p) for n, p in values if p.requires_grad]

    def before_backward(self, frame):
        if not self.selected():
            return
        loc = frame.f_locals
        names, params = zip(*self.parameters())
        self.pending = dict(epoch=self.epoch, batch=self.batch,
            ce=scalar(loc['ce_loss']), stat=scalar(self.stat), pred=scalar(self.pred),
            objective=scalar(loc['loss']),
            lr=[g['lr'] for g in self.owner.optimizer.param_groups],
            logits=tensor_summary(loc['outputs']),
            student_distillation_feature=tensor_summary(loc['student_features'][-1]))
        # Per-term gradients are read-only: autograd.grad does not populate
        # parameter.grad. The original loss.backward still executes unchanged.
        terms = {'ce': loc['ce_loss'], 'stat': self.stat, 'pred': self.pred}
        gradients = {}
        self.pending['term_gradient_norms'] = {}
        for name, term in terms.items():
            self.phase = name
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            self.pending['term_gradient_norms'][name] = gradient_report(names, grads)
            if name in ('ce', 'pred'):
                gradients[name] = grads
        pairs = [(a.detach().float(), b.detach().float()) for a,b in
                 zip(gradients['ce'], gradients['pred']) if a is not None and b is not None]
        if pairs:
            dot = torch.stack([(a*b).sum() for a,b in pairs]).sum()
            na = torch.stack([a.square().sum() for a,b in pairs]).sum().sqrt()
            nb = torch.stack([b.square().sum() for a,b in pairs]).sum().sqrt()
            self.pending['ce_pred_gradient_cosine'] = scalar(dot/(na*nb).clamp_min(1e-30))
        self.phase = 'total'

    def after_backward(self):
        if self.pending is not None:
            names, params = zip(*self.parameters())
            self.pending['total_gradient_norms_before_clip'] = gradient_report(names, [p.grad for p in params])

    def before_step(self, optimizer, args, kwargs):
        if self.pending is not None:
            self.step_before = {n: p.detach().clone() for n,p in self.parameters()}
            names, params = zip(*self.parameters())
            self.pending['total_gradient_norms_after_clip'] = gradient_report(names, [p.grad for p in params])

    def after_step(self, optimizer, args, kwargs):
        if self.pending is None:
            return
        names, params = zip(*self.parameters())
        updates = [p.detach() - self.step_before[n] for n,p in zip(names, params)]
        self.pending['update_norms'] = gradient_report(names, updates)
        self.pending['parameter_norms_before_step'] = gradient_report(names, [self.step_before[n] for n in names])
        fields = ['input_rms','input_absmax','output_rms','output_zero_fraction',
                  'low_slope_fraction','positive_low_slope_fraction','outside_domain_fraction',
                  'mean_abs_slope','input_finite_fraction']
        self.pending['activations'] = {
            n: {k: scalar(v) for k,v in zip(fields, torch.stack(vs).mean(0))}
            for n,vs in self.layers.items()}
        self.pending['activation_gradients'] = {
            n+'|'+phase: dict(zip(['rms','absmax','zero_fraction'],
                [scalar(x) for x in torch.stack(vs).mean(0)]))
            for (n,phase),vs in self.activation_grads.items()}
        self.rows.write(json.dumps(self.pending)+'\n')
        self.pending = self.step_before = self.stat = self.pred = None
        self.layers.clear()
        self.activation_grads.clear()


def install(recorder, stop_after):
    original_epoch = TrainerCiFarTimmStyleFeatureKD.train_one_epoch
    original_unpack = TrainerCiFarTimmStyleFeatureKD._unpack_train_batch
    original_backward = torch.Tensor.backward
    original_save = trainer_base.save_latest

    def epoch(owner, epoch):
        if recorder.owner is None:
            recorder.attach(owner)
        recorder.epoch, recorder.batch, recorder.active = epoch+1, 0, True
        try:
            return original_epoch(owner, epoch)
        finally:
            recorder.active = False

    def unpack(owner, data):
        if recorder.active and owner is recorder.owner:
            recorder.batch += 1
        return original_unpack(owner, data)

    def backward(tensor, *args, **kwargs):
        frame = inspect.currentframe().f_back
        loss_tensor = frame.f_locals.get('loss')
        # Accumulation moved backward into a helper. Inspect the training
        # caller's unscaled loss terms, but match the actual backward tensor
        # (which is divided for accumulation). No optimization is replaced.
        if frame.f_code is TrainerCiFarTimmStyleFeatureKD._backward_step_accum.__code__:
            loss_tensor = frame.f_locals.get('loss_for_backward')
            frame = frame.f_back
        is_training_loss = (recorder.active and frame.f_locals.get('self') is recorder.owner
                            and loss_tensor is tensor)
        if is_training_loss:
            recorder.before_backward(frame)
        result = original_backward(tensor, *args, **kwargs)
        if is_training_loss:
            recorder.after_backward()
        return result

    def save(owner, epoch, history):
        original_save(owner, epoch, history)
        if epoch in {1,5,10,15} or 17 <= epoch <= stop_after:
            epoch_root = recorder.out/'epoch_checkpoints'/f'epoch_{epoch:03d}'
            dst = epoch_root/owner.model_name/(owner.model_name+'_latest_ckpt.pth')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_path(owner), dst)
            config_path = Path(owner.save_path)/'run_config.json'
            if config_path.exists():
                shutil.copy2(config_path, epoch_root/'run_config.json')
            print(f'DIAGNOSTIC_RECOVERY epoch={epoch} path={dst}', flush=True)
        if epoch >= stop_after:
            (recorder.out/'DIAGNOSTIC_COMPLETE').write_text(
                f'Stopped after {epoch} completed epochs; original 300-epoch schedule retained.\n')
            raise DiagnosticStop

    TrainerCiFarTimmStyleFeatureKD.train_one_epoch = epoch
    TrainerCiFarTimmStyleFeatureKD._unpack_train_batch = unpack
    torch.Tensor.backward = backward
    trainer_base.save_latest = save
    # The timm trainer imports this function locally inside train(), unlike
    # TrainerCiFar's module-global reference. Cover BOTH real training branches.
    training_recovery.save_latest = save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=30)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('training_argv', nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    argv = opts.training_argv
    if argv and argv[0] == '--': argv = argv[1:]
    if argv and argv[0] == '-u': argv = argv[1:]
    assert argv[0] == 'train_ode_cifar.py', argv
    sys.argv = argv
    args = training.get_args()
    out = opts.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for key in ('save_path', 'output_save_path'):
        assert Path(getattr(args,key)).resolve().is_relative_to(out), (key, getattr(args,key))
    assert args.model_name is None and args.num_epochs == 300
    assert args.dataset == 'cifar10' and args.seed == 4096
    assert args.learning_rate == .005 and args.warmup_epoch == 5
    assert args.R == 67e3 and args.C == 500e-15 and args.v_dd == .6 and args.one_over_q == 6
    assert args.final_head_type == 'digital' and args.toggle_timing_mode == 'fixed'
    assert args.toggle_y_time == 10e-9 and args.z_over_y_time == 1
    assert args.scale_train_recipe == 1 and args.distill_method == 'srrl'
    assert args.activation_corner_mode == 'fixed' and args.activation_corner == 'TT_25_1_MC18'
    assert '1022_RELU_Voltage' in args.activation_curve_path
    manifest = dict(argv=argv, args=vars(args), stop_after=opts.stop_after,
        sampled_activation_elements_per_call=2048,
        diagnostic_steps='every batch in epochs18-25; every10 batches otherwise',
        source_log=str(ROOT/'logs/C10_C96L16_cc_1022_cascode_R67k_vdd0p6_q6_digital/N16_C24_n04_n15_n24/train_ToggleODEXInitFFFB.log'))
    sources = ['train_ode_cifar.py','trainer.py','trainer_timm.py','ode_pc.py',
               'final_linear.py','measured_activation.py','distillation/srrl.py',
               'diagnostic_scripts/replay_c10_cascode_pt_collapse.py',
               'launch_scripts/run_kdcrd_then_ft.sbatch',
               'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv']
    manifest['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    with (out/'run_manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2, default=str)
    if opts.capture_only:
        print('Captured exact production PT parser arguments; no training executed.', flush=True)
        return
    rec = Recorder(out)
    install(rec, opts.stop_after)
    try:
        training.main()
    except DiagnosticStop:
        print('DIAGNOSTIC STOP: epoch-boundary recovery saved; no FT/eval launched.', flush=True)
    finally:
        rec.rows.close()


if __name__ == '__main__':
    main()
