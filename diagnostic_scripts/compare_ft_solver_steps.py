"""One actual SRRL training batch, with process-local solver instrumentation.

Run separately with plain or physical. No training/solver source is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import ode_pc
import train_ode_cifar as training
import trainer_timm

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['plain', 'physical'])
p.add_argument('--task', choices=['cifar10', 'cifar100'], default='cifar10')
p.add_argument('--profile-saved-tensors', action='store_true')
p.add_argument('--batch-from', type=Path)
p.add_argument('--tag', default='')
p.add_argument('--no-solver-counting', action='store_true',
               help='Control run using the completely unchanged solver entry point.')
a = p.parse_args()
out = ROOT / 'results/ft_solver_diagnostic'
out.mkdir(parents=True, exist_ok=True)
script = (ROOT / 'launch_scripts/local_tc_state1_ft_warmup.sh').read_text()
model = script.split('MODEL_NAME="', 1)[1].split('"', 1)[0]
tokens = shlex.split(script.split('python -u train_ode_cifar.py', 1)[1].split('status=$?', 1)[0].replace('\\\n', ''))
tc = subprocess.check_output(['bash', '-c',
    'export TC_NONIDEALITIES=true TC_STATE=1 TOGGLE_MODE=none SWITCH_INF=false; '
    'source launch_scripts/tc_nonideality_args.sh ft; printf "%s\\0" "${TC_ARGS[@]}"'], cwd=ROOT)
argv = [model if t == '$MODEL_NAME' else t for t in tokens[:-1]]
argv += tc.decode().rstrip('\0').split('\0')
sys.argv = ['train_ode_cifar.py'] + argv
config = training.get_args()
if a.task == 'cifar100':
    root = ROOT / 'saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study'
    path, = root.glob('TIMMQAT*22Layers6l7l6*/*_latest_ckpt.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = argparse.Namespace(**checkpoint['training_recovery']['config'])
    config.num_workers = 0
    del checkpoint
config.output_save_path = str(out / a.mode)
if a.mode == 'plain':
    config.ode_wrapper = None
    config.tc_nonidealities = False
    config.nonlinear_R = False
    config.enable_measured_activation = False
    config.enable_measured_pooling = False
    config.enable_spin_variation = False
    config.enable_summing_current_noise = False
    config.enable_coupler_noise = False
    config.enable_slow_summing_current = False
    config.enable_slow_coupler_noise = False
    config.noise_level = None
training.get_args = lambda: config
report = {'mode': a.mode, 'config': vars(config), 'layers': [], 'status': 'starting'}
active = None
saved = {}
saved_by_site = Counter()
step_records = []

class SavedTensor:
    """Same tensor storage, no extra GPU copy; track its backward lifetime."""
    def __init__(self, tensor):
        self.tensor = tensor.detach()
        storage = tensor.untyped_storage()
        self.key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if self.key not in saved:
            frame = sys._getframe(2)
            site = '%s:%d' % (Path(frame.f_code.co_filename).name, frame.f_lineno)
            saved[self.key] = [0, storage.nbytes(), site]
            saved_by_site[site] += storage.nbytes()
        saved[self.key][0] += 1

    def __del__(self):
        entry = saved.get(self.key)
        if entry is not None:
            entry[0] -= 1
            if entry[0] == 0:
                saved_by_site[entry[2]] -= entry[1]
                del saved[self.key]

def pack(tensor):
    return SavedTensor(tensor)

def snapshot():
    return {k: round(v / 2**20, 4) for k, v in saved_by_site.items() if v}

def save():
    suffix = '_uncounted' if a.no_solver_counting else ''
    if a.task == 'cifar100': suffix += '_cifar100'
    if a.profile_saved_tensors: suffix += '_profile'
    if a.tag: suffix += '_' + a.tag
    (out / (a.mode + suffix + '.json')).write_text(json.dumps(report, indent=2, default=str))

original_solve = ode_pc.aca_ode_solve
def counted_solve(func, y0, options, **kwargs):
    row = active
    if row is None:
        return original_solve(func, y0, options, **kwargs)
    row['solver_options'] = {k: float(v) if torch.is_tensor(v) and v.numel()==1 else v
                             for k,v in options.items() if k in ('t0','t1','rtol','atol','method')}
    solver = original_solve(func, y0, options, return_solver=True, **kwargs)
    forward = solver.func.forward
    def rhs(*args, **kw):
        row['rhs_calls'] += 1
        return forward(*args, **kw)
    solver.func.forward = rhs
    step = solver.step
    def counted_step(*args, **kw):
        grad = torch.is_grad_enabled()
        before_mem = torch.cuda.memory_allocated()
        before_saved = sum(x[1] for x in saved.values())
        row['graph_attempts' if grad else 'search_attempts'] += 1
        result = step(*args, **kw)
        row['graph_steps' if grad else 'search_steps'] += 1
        if grad:
            row['last_t'] = float(args[1] + args[2])
            row['last_dt'] = float(args[2])
            step_records.append(dict(layer=row['layer'], step=row['graph_steps'],
                dt=float(args[2]), allocated_delta_mib=(torch.cuda.memory_allocated()-before_mem)/2**20,
                saved_delta_mib=(sum(x[1] for x in saved.values())-before_saved)/2**20))
        return result
    solver.step = counted_step
    try:
        return solver.integrate(y0=y0, t0=options['t0'],
                                t_eval=options.get('t_eval', [options['t1']]))
    finally:
        # Do not retain a solver->closure->bound-method->solver cycle and its
        # dense graph cache after the solve. Restore class method dispatch.
        del solver.step
        del solver.func.forward
if not a.no_solver_counting:
    ode_pc.aca_ode_solve = counted_solve

class OneBatch:
    def __init__(self, batch): self.batch = batch
    def __len__(self): return 1
    def __iter__(self): yield self.batch

def diagnostic_train(self):
    global active
    batch = next(iter(self.train_dataloader))
    batch_path = out / ('comparison_batch.pt' if a.task == 'cifar10' else 'comparison_batch_cifar100.pt')
    if a.batch_from is not None: batch_path = a.batch_from
    if batch_path.exists():
        batch = torch.load(batch_path, weights_only=False)
    else:
        torch.save(batch, batch_path)
    h = hashlib.sha256()
    for x in batch:
        if torch.is_tensor(x): h.update(x.numpy().tobytes())
    report['batch_sha256'] = h.hexdigest()
    report['lr'] = self.optimizer.param_groups[0]['lr']
    report['model_parameters'] = sum(p.numel() for p in self.model.parameters())
    report['batch_shapes'] = [list(x.shape) for x in batch if torch.is_tensor(x)]
    self.train_dataloader = OneBatch(batch)
    def before(idx):
        def hook(module, inputs):
            global active
            active = dict(layer=idx, rhs_calls=0, graph_attempts=0, graph_steps=0,
                          search_attempts=0, search_steps=0, status='running')
            active['input_shape'] = list(inputs[0].shape)
            active['entry_allocated_mib'] = torch.cuda.memory_allocated()/2**20
            active['act_fn'] = str(getattr(module, 'act_fn', None))
            report['layers'].append(active)
            print('ENTER LAYER', idx, flush=True)
        return hook
    def after(module, inputs, output):
        active['status'] = 'completed'
        active['allocated_mib'] = torch.cuda.memory_allocated() / 2**20
        active['saved_by_site_mib'] = snapshot()
        active['output_shape'] = list(output.shape)
        print(json.dumps(active), flush=True)
        save()
    for idx, block in enumerate(self.model.PcConvs, 1):
        block.register_forward_pre_hook(before(idx))
        block.register_forward_hook(after)
    torch.cuda.reset_peak_memory_stats()
    try:
        if a.profile_saved_tensors:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x.tensor):
                report['loss'] = self.train_one_epoch(0)
        else:
            report['loss'] = self.train_one_epoch(0)
        report['status'] = 'forward_backward_optimizer_completed'
    except Exception as exc:
        report['status'] = type(exc).__name__
        report['error'] = str(exc)
        if active is not None and active['status'] == 'running':
            active['status'] = 'failed'
        traceback.print_exc()
        raise
    finally:
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['step_records'] = step_records
        report['saved_by_site_mib_at_end'] = snapshot()
        save()
        print('DIAGNOSTIC_RESULT', report['status'], flush=True)

trainer_timm.TrainerCiFarTimmStyle.train = diagnostic_train
save()
training.main()
