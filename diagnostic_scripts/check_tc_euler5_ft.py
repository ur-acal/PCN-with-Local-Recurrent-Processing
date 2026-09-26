"""Exercise the local Euler FT command without saving a trained checkpoint."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import ode_pc
import train_ode_cifar as entry
from trainer_timm import TrainerCiFarTimmStyle
from TorchDiffEqPack.odesolver.fixed_grid_solver import Euler

p = argparse.ArgumentParser()
p.add_argument('--state', type=int, choices=(1, 2), required=True)
p.add_argument('--mode', choices=('train', 'eval'), default='train')
p.add_argument('--method', choices=('euler', 'dopri5'), default='euler')
p.add_argument('--eval-batches', type=int, default=2)
p.add_argument('--checkpoint', choices=('pretrain', 'ft'), default='pretrain')
p.add_argument('--ideal', action='store_true', help='Inspect clean pretrained state ranges.')
p.add_argument('--unitless-bound', type=float, help='Diagnostic state clamp, with --ideal only.')
args = p.parse_args()
if args.unitless_bound is not None and (not args.ideal or args.unitless_bound <= 0):
    p.error('--unitless-bound requires --ideal and a positive bound')
command = subprocess.check_output(
    ['bash', 'launch_scripts/local_tc_rgb_ft_euler5.sh', str(args.state), '--dry-run'],
    cwd=ROOT, text=True)
sys.argv = shlex.split(command)[2:]
config = entry.get_args()
config.method = args.method
if args.checkpoint == 'ft':
    paths = list(Path(config.save_path).glob('TIMMQAT*22Layers6l7l6*/*_full_param_last_ckpt.pth'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one final FT checkpoint, found {paths}')
    config.model_name = paths[0].parent.name
    config.ckpt = 'full_param_last'
if args.mode == 'eval':
    config.distill_method = 'none'
    config.teacher_ckpt = None
if args.ideal:
    if args.mode != 'eval' or args.checkpoint != 'pretrain':
        raise ValueError('--ideal requires --mode eval --checkpoint pretrain')
    config.ode_wrapper = None
    config.tc_nonidealities = False
    for name in ('nonlinear_R', 'enable_measured_activation', 'enable_measured_pooling',
                 'enable_spin_variation', 'enable_summing_current_noise', 'enable_coupler_noise'):
        setattr(config, name, False)
config.output_save_path = str(ROOT / 'results/tc_euler5_check' / f'state{args.state}')
entry.get_args = lambda: config
# Skip only the standalone teacher-accuracy check; retain actual SRRL training.
entry.evaluate_teacher = lambda *a, **kw: None
original_step = Euler.step
solvers = {}

def counted_step(self, *a, **kw):
    key = id(self)
    if key not in solvers:
        solvers[key] = [self, 0, float(self.t1-self.t0), float(self.h)]
    solvers[key][1] += 1
    return original_step(self, *a, **kw)

Euler.step = counted_step

def check(self):
    solvers.clear()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    report = {'state': args.state, 'mode': args.mode, 'method': args.method,
              'checkpoint': args.checkpoint, 'ideal': args.ideal}
    report['unitless_bound'] = args.unitless_bound
    ranges = []
    handles = []
    if args.ideal:
        if args.unitless_bound is not None:
            bound = args.unitless_bound
            for module in self.model.modules():
                if isinstance(module, ode_pc.ODEBlockPC):
                    module.option_aca['proj_fn'] = torch.nn.Hardtanh(-bound, bound)
                    handles.append(module.register_forward_pre_hook(
                        lambda m, a: (a[0].clamp(-bound, bound), *a[1:])))
        def observe(module, inputs, output):
            value = output.detach().abs()
            ranges.append({'layer': module.layer_idx,
                           'fraction_above_1': float((value > 1).float().mean()),
                           'fraction_above_5': float((value > 5).float().mean()),
                           'max_abs': float(value.max())})
        handles += [m.register_forward_hook(observe) for m in self.model.modules()
                    if isinstance(m, ode_pc.ODEBlockPC)]
    try:
        if args.mode == 'train':
            batch = next(iter(self.train_dataloader))
            self.train_dataloader = [batch]
            loss = self.train_one_epoch(0)
            grads = [v.grad for v in self.model.parameters() if v.grad is not None]
            report.update(loss=float(loss), gradient_tensors=len(grads),
                          finite_gradients=all(bool(torch.isfinite(g).all()) for g in grads))
        else:
            acc, top5, _, _ = self.evaluate(self.val_dataloader, max_batches=args.eval_batches)
            report.update(top1=acc, top5=top5, batches=args.eval_batches)
        report['status'] = 'passed'
    except Exception as e:
        report.update(status='failed', error=repr(e))
        raise
    finally:
        for handle in handles:
            handle.remove()
        torch.cuda.synchronize()
        report.update(seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      steps=[r[1] for r in solvers.values()],
                      duration_over_h=[r[2]/r[3] for r in solvers.values()])
        if ranges:
            report['clean_layer_output_ranges'] = ranges
        out = ROOT / 'results/tc_euler5_check'
        out.mkdir(parents=True, exist_ok=True)
        suffix = '_ideal' if args.ideal else ''
        if args.unitless_bound is not None:
            suffix += f'_bound{args.unitless_bound:g}'
        (out / f'state{args.state}_{args.method}_{args.mode}_{args.checkpoint}{suffix}.json').write_text(json.dumps(report, indent=2))
        print('EULER_CHECK '+json.dumps(report), flush=True)

TrainerCiFarTimmStyle.train = check
entry.main()
