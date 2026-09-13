"""Reproducible TC loop/grouped timing and CUDA allocator memory benchmark.

Uses a saved full-parameter checkpoint, ENOB=None and one clean recorded
scanGFI batch. Only nonlinear-R is enabled, isolating convolution cost.
No parameter updates, dataset accuracy evaluation, or checkpoint writes.
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import signal
import sys
import time
import types

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from inference_utils import load_and_prepare_model
from ode_pc import ODEXInitFFFB, QATWrapper1State
from pc_conv import PCConvReLU6Noisy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--method',choices=['loop','grouped'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--data',default='../cifar-10-data/scanGFI/cifar100_raw.h5')
    p.add_argument('--memory_fraction',type=float,default=.75)
    p.add_argument('--compare_only',action='store_true',help='Alternate both methods on identical cached inputs/curves; skip full backward.')
    p.add_argument('--disable_nonlinear_R',action='store_true')
    p.add_argument('--training_only',action='store_true',help='Measure full training forward/backward; omit per-convolution timings.')
    args=p.parse_args()
    logging.disable(logging.CRITICAL)
    torch.manual_seed(107);torch.cuda.manual_seed_all(107)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    result=dict(options=vars(args),device=torch.cuda.get_device_name(),torch=torch.__version__,
        enob=None,seed=107,scope='clean recorded input; no optimizer update',
        nonlinear_R=not args.disable_nonlinear_R,
        free_total_bytes=torch.cuda.mem_get_info(),measurements=[])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(result,indent=2))
    def measure(name,fn,repeats=1):
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        entry=dict(name=name,baseline_allocated_bytes=torch.cuda.memory_allocated())
        times=[]
        def timeout(*unused):raise TimeoutError('Phase exceeded 240 seconds')
        old=signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
        try:
            for _ in range(repeats):
                torch.cuda.synchronize();start=time.perf_counter()
                fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            entry.update(status='ok',seconds=times,median_seconds=float(np.median(times)))
        except (torch.OutOfMemoryError,TimeoutError) as exc:
            entry.update(status=type(exc).__name__,error=str(exc),seconds=times)
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,old)
            entry.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            result['measurements'].append(entry);save();print(json.dumps(entry),flush=True)
        return entry['status']=='ok'
    with h5py.File(args.data) as data:
        indices=np.flatnonzero(~data['train'][:])[:args.batch_size]
        x=torch.from_numpy(data['images'][indices]).to('cuda').clamp(0,1)
    result['input_shape']=list(x.shape)
    result['input_indices']=indices.tolist()
    model=load_and_prepare_model(args.checkpoint,'cuda',pc_conv_layer=PCConvReLU6Noisy,
        conv_only=True,fuse_bn=False,noise_level=0.,
        ode_params=dict(ode_block=ODEXInitFFFB,method='dopri5',t_end=1.75,tol=1e-4,n_steps=5),
        ode_wrapper_params=dict(ode_wrapper=QATWrapper1State,tc_nonidealities=True,
            tc_conv_method=args.method,R=10e3,R_max=150e3,C=49e-15,v_dd=.1,
            one_over_q=1,w_bits=5,enob=None,weight_quant_factor_bits=None,thermal_noise=False,
            nonlinear_R=not args.disable_nonlinear_R,
            nonlinear_R_train_mode='none' if args.disable_nonlinear_R else 'exact_curve',nonlinear_R_curve_sharing='shared',
            nonlinear_R_table='hardware_data/res_vs_vin_10k_150k.csv',
            tc_covariance_table='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
            nonlinear_R_curve_seed=19))
    model.eval()
    samples={};calls=[0]
    for b in model.PcConvs:
        original=b._tc_dense_conv
        def instrument(self,module,source,curves,_original=original):
            calls[0]+=1
            key=(self.layer_idx,'FFconv' if module is self.FFconv else 'FBconv')
            if key not in samples:samples[key]=(self,module,source.detach().clone(),curves)
            return _original(module,source,curves)
        b._tc_dense_conv=types.MethodType(instrument,b)
    def forward():
        with torch.no_grad():
            y=model(x)
            if not torch.isfinite(y).all():raise ValueError('Nonfinite logits')
            result['last_logits']=y.cpu().tolist()
    if not measure('full_model_forward_warmup',forward):return
    result['warmup_dense_calls']=calls[0]
    if args.compare_only:
        # Same live model and curve realizations; alternate order to reduce
        # confounding from another process sharing this GPU.
        def set_mode(mode):
            for block in model.PcConvs:block._tc_conv_method=mode
        errors=[]
        for b,module,source,curves in samples.values():
            outputs=[];gradients=[]
            for mode in ('loop','grouped'):
                set_mode(mode)
                inp=source.detach().requires_grad_()
                y=b._tc_dense_conv(module,inp,curves)
                param=module.parametrizations.weight.original
                gradients.append(torch.autograd.grad(y.square().mean(),(inp,param)))
                outputs.append(y.detach())
            torch.testing.assert_close(*outputs,rtol=2e-4,atol=1e-6)
            for a,c in zip(*gradients):torch.testing.assert_close(a,c,rtol=2e-4,atol=1e-6)
            errors.append(float((outputs[0]-outputs[1]).abs().max()))
        result['cuda_operator_max_abs_errors']=errors
        def operator_pass(backward=False):
            model.zero_grad(set_to_none=True)
            for b,module,source,curves in samples.values():
                if backward:
                    inp=source.detach().requires_grad_()
                    b._tc_dense_conv(module,inp,curves).square().mean().backward()
                else:
                    with torch.no_grad():b._tc_dense_conv(module,source,curves)
            model.zero_grad(set_to_none=True)
        for repeat in range(args.repeats):
            for mode in (('loop','grouped') if repeat%2==0 else ('grouped','loop')):
                set_mode(mode)
                measure('paired_%d_%s_full_forward'%(repeat,mode),forward)
                measure('paired_%d_%s_operator_forward'%(repeat,mode),operator_pass)
                measure('paired_%d_%s_operator_backward'%(repeat,mode),lambda:operator_pass(True))
        save();return
    calls[0]=0
    measure('full_model_forward',forward,args.repeats)
    result['timed_dense_calls']=calls[0]
    if args.training_only:
        model.train()
        def training_forward():
            model.zero_grad(set_to_none=True)
            y=model(x)
            if not torch.isfinite(y).all():raise ValueError('Nonfinite training logits')
        measure('full_model_training_forward',training_forward)
    # Capture full-model backward feasibility before microbenchmarks.
    def backward():
        model.zero_grad(set_to_none=True)
        y=model(x);y.square().mean().backward()
        model.zero_grad(set_to_none=True)
    measure('full_model_forward_backward',backward)
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    if args.training_only:
        save();return
    for key,(b,module,source,curves) in samples.items():
        def micro(backward=False):
            model.zero_grad(set_to_none=True)
            if backward:
                inp=source.detach().requires_grad_()
                y=b._tc_dense_conv(module,inp,curves);y.square().mean().backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():b._tc_dense_conv(module,source,curves)
        micro();micro(True)  # Warm both execution paths.
        measure('conv_%s_%s_forward'%key,micro,args.repeats)
        measure('conv_%s_%s_backward'%key,lambda:micro(True),args.repeats)
    save()


if __name__=='__main__':main()
