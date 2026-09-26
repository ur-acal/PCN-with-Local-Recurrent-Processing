"""Evaluate old scanGFI pretraining and QAT checkpoints through ode_inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import ode_inference as inference
from pc_conv import PCConvReLU6Noisy

p = argparse.ArgumentParser()
p.add_argument('mode', choices=('fp', 'scaled_fp', 'mapped', 'ft', 'script'))
p.add_argument('--rep', type=int, default=6)
p.add_argument('--enob', default='8')
p.add_argument('--max-batches', type=int, default=0)
a = p.parse_args()
base = ('TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_'
        '1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_'
        '16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI')
name = (f'TIMMQAT5b8aNT0p25mul{base}_1REP' if a.mode in ('ft', 'script')
        else f'{base}_{a.rep}REP')
wrapper = {'fp': 'none', 'scaled_fp': 'ODEWrapperRC', 'mapped': 'ODEWrapper1State',
           'ft': 'QATTester1State', 'script': 'QATTester1State'}[a.mode]
sys.argv = ['ode_inference.py', '--model_name', name, '--model_dir', './saved_ckpt',
            '--ckpt', 'best', '--task', 'cifar100', '--img_type', 'scanGFI',
            '--pc_conv', 'PCConvReLU6Noisy', '--ode_block', 'ODEXInitFFFB',
            '--ode_wrapper', wrapper, '--method', 'dopri5', '--tol', '1e-6',
            '--n_steps', '100', '--test_bs', '128', '--test_only', 'true',
            '--test_only_nl', '0', '--diff_mismatch', str(a.mode == 'script'),
            '--thermal_noise', str(a.mode == 'script'), '--sde_noise_type', 'add',
            '--conv_only', 'true', '--test_expanded', 'true',
            '--R', '10e3', '--R_max', '150e3', '--C', '49e-15', '--v_dd', '.1',
            '--one_over_q', '1', '--w_bits', '5', '--enob', a.enob,
            '--nonlinear_R', 'false', '--mul_mismatch_mode', 'static_mismatch']
config = inference.parse_args()
torch.manual_seed(4096)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loader = inference.get_test_data(test_bs=128, img_type='scanGFI', task='cifar100')
path = ROOT / 'saved_ckpt' / name / f'{name}_best_ckpt.pth'
with torch.no_grad():
    net = inference.run_test_only(config, loader, str(path), PCConvReLU6Noisy,
                                  device, return_net=True, noise_level=0.)
    correct = total = 0
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if a.max_batches and i >= a.max_batches:
            break
        pred = net(x.to(device)).argmax(1)
        correct += int((pred == y.to(device)).sum())
        total += len(y)
        print(f'Batch {i+1}/{len(loader)}: {100*correct/total:.2f}%', flush=True)
    row = dict(mode=a.mode, checkpoint=str(path), enob=a.enob, total=total,
               top1=100*correct/total, seconds=time.perf_counter()-start,
               solver_eps=[str(b.option_aca.get('eps')) for b in net.PcConvs])
out = ROOT / 'results/legacy_scangfi_recovery'
out.mkdir(parents=True, exist_ok=True)
(out / f'{a.mode}_rep{a.rep}_enob{a.enob}.json').write_text(json.dumps(row, indent=2))
print('LEGACY_RESULT '+json.dumps(row), flush=True)
