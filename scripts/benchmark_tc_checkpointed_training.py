"""Dense TC grouped-convolution activation-checkpointing timing, no diagnostic hooks.
All-on defaults reproduce the TC training nonideality recipe. No optimizer updates.
"""
import argparse, collections, functools, json, logging, sys, time, types
from pathlib import Path
sys.path.insert(0, '/home/rongzeng/_workspce_old/repos/pcn/collaboration/ScAN-PCN')
import torch, h5py, numpy as np
from inference_utils import load_and_prepare_model
from ode_pc import ODEXInitFFFB, QATWrapper1State
from pc_conv import PCConvReLU6Noisy, PCConvReLU6
p=argparse.ArgumentParser()
p.add_argument('--batch',type=int,default=128)
p.add_argument('--mode',choices=['clean','loop','grouped','all','legacy'],default='all')
p.add_argument('--output',required=True)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--conv-method',choices=['loop','grouped','shared'],default=None)
p.add_argument('--curve-sampling',choices=['histogram','uniform'],default='histogram')
p.add_argument('--no-checkpoint',action='store_true')
p.add_argument('--sampling-repeats',type=int,default=0)
p.add_argument('--sensitivity-projections',type=int,default=0)
p.add_argument('--input-offset',type=int,default=0)
p.add_argument('--compare-sampling',action='store_true')
p.add_argument('--count-stages',action='store_true',help='Diagnostic counts; keep separate from uninstrumented timing.')
p.add_argument('--tol',type=float,default=1e-6)
p.add_argument('--enob',type=lambda s:None if s.lower()=='none' else int(s),default=None)
p.add_argument('--effects', help='Explicit comma-separated TC effects: nonlinear,spin,summing,coupler,relu,pooling; empty disables all.')
p.add_argument('--legacy-mismatch',type=float,default=.25)
p.add_argument('--reference-correction',action='store_true',help='Diagnostic prior shared correction, for matched optimization benchmarks.')
p.add_argument('--compare-correction',action='store_true',help='Alternate optimized/prior correction in one process.')
p.add_argument('--save-final-tensors',action='store_true',help='Diagnostic final logits and parameter gradients.')

a=p.parse_args()
if a.reference_correction or a.compare_correction:
 import tc_shared_correction
 optimized_correction=tc_shared_correction.shared_correction
 from tc_nonidealities import positive_resistance
 def prior_correction(x,curve,grid,rail,floor):
  grid,table=grid.to(x),curve.resistance.to(x)
  query=x.clamp(-rail,rail).clamp(grid[0],grid[-1])
  idx=(torch.bucketize(query,grid)-1).clamp(0,grid.numel()-2)
  slope=(table[1:]-table[:-1])/(grid[1:]-grid[:-1])
  r=table[idx]+slope[idx]*(query-grid[idx])
  return x*curve.nominal_R.to(x)/positive_resistance(r,floor)
 if a.reference_correction:tc_shared_correction.shared_correction=prior_correction
effects=set(a.effects.split(','))-{''} if a.effects is not None else (
 set(('nonlinear','spin','summing','coupler','relu','pooling')) if a.mode=='all'
 else {'nonlinear'} if a.mode in ('loop','grouped') else set())
if effects-set(('nonlinear','spin','summing','coupler','relu','pooling')):
 p.error('Unknown effect')
if a.mode=='legacy' and effects:p.error('Legacy mode cannot enable TC effects')
logging.disable(logging.CRITICAL)
torch.manual_seed(107);torch.cuda.manual_seed_all(107)
torch.cuda.set_per_process_memory_fraction(.75)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
name='TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP'
wrappers={}
model=load_and_prepare_model(f'saved_ckpt/{name}/{name}_full_param_best_ckpt.pth','cuda',
 pc_conv_layer=PCConvReLU6 if a.mode=='legacy' else PCConvReLU6Noisy,wrappers=wrappers,
 conv_only=True,fuse_bn=False,noise_level=0.,
 ode_params=dict(ode_block=ODEXInitFFFB,method='dopri5',t_end=1.75,tol=a.tol,n_steps=5),
 ode_wrapper_params=dict(ode_wrapper=QATWrapper1State,tc_nonidealities=a.mode!='legacy',
 tc_conv_method=a.conv_method or ('grouped' if a.mode in ('grouped','all') else 'loop'),tc_curve_sampling=a.curve_sampling,
 R=10e3,R_max=150e3,C=49e-15,v_dd=.1,one_over_q=1,w_bits=5,enob=a.enob,weight_quant_factor_bits=None,thermal_noise=False,
 nonlinear_R='nonlinear' in effects,nonlinear_R_train_mode='none',nonlinear_R_curve_sharing='shared',
 nonlinear_R_table='hardware_data/res_vs_vin_10k_150k.csv',tc_covariance_table='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',nonlinear_R_curve_seed=19,
 enable_spin_variation='spin' in effects,sigma_spin=.1,spin_variation_seed=4096,
 enable_summing_current_noise='summing' in effects,summing_current_p=.6e-12,summing_noise_seed=4096,
 enable_coupler_noise='coupler' in effects,coupler_noise_p=.6e-12,coupler_noise_seed=4096,
 enable_measured_activation='relu' in effects,activation_curve_path='hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv',activation_corner='MC18'))
if 'pooling' in effects:
 from tc_cli import pooling_options
 from measured_pooling import configure_measured_pooling
 pool_args=types.SimpleNamespace(nonlinear_R_table='hardware_data/res_vs_vin_10k_150k.csv',
     tc_covariance_table='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',R=10e3,R_max=150e3)
 configure_measured_pooling(model,wrappers['wrappers'],enable_nonideality=True,seed=4096,**pooling_options(pool_args,wrappers['wrappers']))
with h5py.File('../cifar-10-data/scanGFI/cifar100_raw.h5') as f:
 idx=np.flatnonzero(~f['train'][:])[a.input_offset:a.input_offset+a.batch]
 x=torch.from_numpy(f['images'][idx]).cuda().clamp(0,1)

if a.sensitivity_projections:
 from tc_shared_sensitivity import run_sensitivity
 run_sensitivity(model,x,a.sensitivity_projections,a.output,
     dict(checkpoint=f'saved_ckpt/{name}/{name}_full_param_best_ckpt.pth',
          input_indices=idx.tolist(),mode=a.mode,batch=a.batch,seed=107))
 sys.exit(0)

from torch.utils.checkpoint import checkpoint
model.train()
runtime_model=model
if a.mode=='legacy':
 from trainer import WrappedNoisyModel
 # Same pre-quantization functional parameter-mismatch wrapper as legacy FT.
 runtime_model=WrappedNoisyModel(model,noise_levels=a.legacy_mismatch,noise_type='mul')
for block in model.PcConvs:
 if a.no_checkpoint:continue
 original=block._tc_dense_conv
 def checkpointed(self,module,source,curves,_original=original):
  if torch.is_grad_enabled():
   return checkpoint(lambda src:_original(module,src,curves),source,use_reentrant=False)
  return _original(module,source,curves)
 block._tc_dense_conv=types.MethodType(checkpointed,block)

stage_counts=collections.Counter()
rhs_counts=collections.Counter()
if a.count_stages:
 for block in model.PcConvs:
  make_original=block._make_ode_fn
  def make_counted(self,*args,_make_original=make_original,**kwargs):
   fn=_make_original(*args,**kwargs)
   @functools.wraps(fn)
   def rhs(*args,**kwargs):
    rhs_counts['grad_enabled' if torch.is_grad_enabled() else 'no_grad']+=1
    return fn(*args,**kwargs)
   return rhs
  block._make_ode_fn=types.MethodType(make_counted,block)
  original=block._tc_dense_conv
  def counted(self,module,source,curves,_original=original):
   stage_counts['grad_enabled' if torch.is_grad_enabled() else 'no_grad']+=1
   return _original(module,source,curves)
  block._tc_dense_conv=types.MethodType(counted,block)

result=dict(batch=a.batch,mode=a.mode,checkpoint_conv=not a.no_checkpoint,
 reference_correction=a.reference_correction,
 conv_method=a.conv_method,curve_sampling=a.curve_sampling,device=torch.cuda.get_device_name(),
 checkpoint_path=f'saved_ckpt/{name}/{name}_full_param_best_ckpt.pth',
 enob=a.enob,tol=a.tol,effects=sorted(effects),legacy_mismatch_std=a.legacy_mismatch if a.mode=='legacy' else None,
 unrolled=False,teacher=False,optimizer=False,loss='mean squared logits',
 memory_fraction=.75,seed=107,curve_seed=19,hardware_seeds=4096,
 input_indices=idx.tolist(),input_shape=list(x.shape),rows=[])
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
def save():path.write_text(json.dumps(result,indent=2))
save()
for iteration in range((a.repeats+1)*(2 if a.compare_sampling or a.compare_correction else 1)):
 repeat=iteration//2 if a.compare_sampling or a.compare_correction else iteration
 correction_mode='reference' if a.reference_correction else 'optimized'
 if a.compare_correction:
  correction_mode=('reference','optimized')[(iteration%2)^(repeat%2)]
  tc_shared_correction.shared_correction=prior_correction if correction_mode=='reference' else optimized_correction
 if a.compare_sampling:
  sampling=('uniform','histogram')[(iteration%2)^(repeat%2)]
  for block in model.PcConvs:block._tc_curve_sampling=sampling
 else:sampling=a.curve_sampling
 model.zero_grad(set_to_none=True)
 stage_counts.clear()
 rhs_counts.clear()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 row=dict(repeat=repeat,warmup=repeat==0,phase='forward',sampling=sampling,correction=correction_mode)
 try:
  start=time.perf_counter()
  y=runtime_model(x)
  torch.cuda.synchronize();forward_end=time.perf_counter()
  row['phase']='backward'
  y.square().mean().backward()
  torch.cuda.synchronize();backward_end=time.perf_counter()
  row.update(status='ok',forward_seconds=forward_end-start,backward_seconds=backward_end-forward_end,
             total_seconds=backward_end-start,peak_allocated=torch.cuda.max_memory_allocated(),
             peak_reserved=torch.cuda.max_memory_reserved())
  row['finite_logits']=bool(y.isfinite().all())
  row['finite_gradients']=all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
  if a.count_stages:
   row['dense_stage_calls']=dict(stage_counts)
   row['rhs_calls']=dict(rhs_counts)
  if a.save_final_tensors:
   torch.save(dict(logits=y.detach().cpu(),gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}),path.with_suffix('.pt'))
  del y
 except torch.OutOfMemoryError as exc:
  row.update(status='OOM',error=str(exc),peak_allocated=torch.cuda.max_memory_allocated(),
             peak_reserved=torch.cuda.max_memory_reserved())
 result['rows'].append(row);save();print(json.dumps(row),flush=True)
 if row['status']!='ok':break
rows=[r for r in result['rows'] if not r['warmup'] and r['status']=='ok']
if rows:
 result['median']={k:float(np.median([r[k] for r in rows])) for k in
                   ('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')}
 save();print(json.dumps(result['median']),flush=True)
 if a.compare_correction:
  result['by_correction']={mode:{k:float(np.median([r[k] for r in rows if r['correction']==mode]))
      for k in ('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')}
      for mode in ('reference','optimized')}
  save();print(json.dumps(result['by_correction']),flush=True)
 if a.compare_sampling:
  result['by_sampling']={mode:{k:float(np.median([r[k] for r in rows if r['sampling']==mode]))
      for k in ('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')}
      for mode in ('histogram','uniform')}
  save();print(json.dumps(result['by_sampling']),flush=True)

if a.sampling_repeats:
 # Includes detached code mapping, histogram (where selected), categorical
 # choice, Gaussian draw and positive guard for every FF/FB tensor.
 timing={mode:dict(seconds=[]) for mode in ('uniform','histogram')}
 for iteration in range(a.sampling_repeats+3):
  order=('uniform','histogram') if iteration%2==0 else ('histogram','uniform')
  for sampling in order:
   for block in model.PcConvs:
    block._tc_conv_method='shared';block._tc_curve_sampling=sampling
   torch.cuda.synchronize();start=time.perf_counter()
   for block in model.PcConvs:block._tc_curves_for_solve()
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if iteration>=3:timing[sampling]['seconds'].append(elapsed)
 for item in timing.values():item['median_seconds']=float(np.median(item['seconds']))
 result['sampling_only_all_tensors']=timing;save();print(json.dumps(timing),flush=True)
