"""Dense TC training benchmark for checkpointing and accepted-step reuse.

All-on defaults reproduce the TC training nonideality recipe. No optimizer
updates are performed.
"""
import argparse, collections, json, logging, pickle, sys, time, types
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import torch, h5py, numpy as np
from inference_utils import load_and_prepare_model
from ode_pc import (ODEXInitFFFB, QATWrapper1State, QATWrapper2State,
                    S2NoisyIYAsXZAs0)
from pc_conv import PCConvReLU6Noisy, PCConvReLU6
p=argparse.ArgumentParser()
p.add_argument('--batch',type=int,default=128)
p.add_argument('--memory-fraction',type=float,default=.75)
p.add_argument('--checkpoint',type=Path,help='Override the pinned legacy checkpoint.')
p.add_argument('--state',type=int,choices=(1,2),default=1)
p.add_argument('--dataset',choices=('cifar10','cifar100'),default='cifar100')
p.add_argument('--input-kind',choices=('scangfi','rgb'),default='scangfi')
p.add_argument('--method',choices=('euler','rk2','rk4','rk12','rk23','dopri5'),default='dopri5')
p.add_argument('--mode',choices=['clean','loop','grouped','all','legacy'],default='all')
p.add_argument('--output',required=True)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--conv-method',choices=['loop','grouped','shared'],default=None)
p.add_argument('--curve-sampling',choices=['histogram','uniform'],default='histogram')
p.add_argument('--rhs-checkpoint',action='store_true',help='Enable production ODE-RHS activation checkpointing.')
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
p.add_argument('--training-step-reuse',action='store_true',help='Retain the accepted adaptive-training candidate instead of replaying it.')
p.add_argument('--compare-training-step-reuse',action='store_true',help='Alternate baseline/reuse adaptive training in one process.')
p.add_argument('--compare-rhs-checkpoint',action='store_true',help='Alternate ODE-RHS checkpointing off/on in one process.')
p.add_argument('--compare-training-optimizations',action='store_true',help='Benchmark baseline, step reuse only, RHS checkpointing only, and both.')
p.add_argument('--save-final-tensors',action='store_true',help='Diagnostic final logits and parameter gradients.')
p.add_argument('--optimizer-step',action='store_true',help='Apply an SGD update after each measured backward pass.')

a=p.parse_args()
comparison_count=sum((a.compare_sampling, a.compare_correction,
                      a.compare_training_step_reuse, a.compare_rhs_checkpoint,
                      a.compare_training_optimizations))
if comparison_count > 1:
 p.error('Choose only one --compare-* option per benchmark.')
if a.optimizer_step and comparison_count:
 p.error('--optimizer-step cannot be combined with a comparison benchmark.')
effective_conv_method=(a.conv_method or
 ('grouped' if a.mode=='grouped' else 'shared' if a.mode=='all' else 'loop'))
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
torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
legacy_name='TIMMQAT5b8aNT0p25mulTIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_C100_3K1S96C_0.25Dropout_16Layers4l5l4_2Pool_srrlDistill_a0p3_t2p0_scanGFI_1REP'
block_name='ODEXInitFFFB' if a.state==1 else 'S2NoisyIYAsXZAs0'
cifar10_name=f'TIMMPCNetNoBatchNorm_PCConvReLU6_0.0eps_{block_name}_dopri5Solver_1.75TEnd_0.0001Tol_0.001WD_128BS_0.01LR_3K1S64C_0.25Dropout_22Layers6l7l6_2Pool_srrlDistill_a0p3_t2p0_2REP'
name=legacy_name if a.dataset=='cifar100' else cifar10_name
if a.checkpoint:
 checkpoint_path=a.checkpoint.resolve()
else:
 candidates=[]
 for source_root in (REPO_ROOT,REPO_ROOT.parent/'ScAN-PCN'):
  if a.dataset=='cifar100':
   candidates.append(source_root/'saved_ckpt'/name/f'{name}_full_param_best_ckpt.pth')
  else:
   candidates.append(source_root/'saved_ckpt_runs'/f'tc_rgb_cifar10_state{a.state}_pcn_resnet_depth_study'/name/f'{name}_last_ckpt.pth')
 checkpoint_path=next((path for path in candidates if path.exists()),candidates[0])
block_cls=ODEXInitFFFB if a.state==1 else S2NoisyIYAsXZAs0
wrapper_cls=QATWrapper1State if a.state==1 else QATWrapper2State
wrappers={}
model=load_and_prepare_model(str(checkpoint_path),'cuda',
 pc_conv_layer=PCConvReLU6 if a.mode=='legacy' else PCConvReLU6Noisy,wrappers=wrappers,
 conv_only=True,fuse_bn=False,noise_level=0.,
 ode_params=dict(ode_block=block_cls,method=a.method,t_end=1.75,tol=a.tol,n_steps=5,
                 reuse_accepted_step_training=a.training_step_reuse,
                 checkpoint_ode_rhs_training=a.rhs_checkpoint),
 ode_wrapper_params=dict(ode_wrapper=wrapper_cls,tc_nonidealities=a.mode!='legacy',
 tc_conv_method=effective_conv_method,tc_curve_sampling=a.curve_sampling,
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
     measured_pooling_curve_path='hardware_data/res_vs_vin_10k_150k.csv',
     measured_pooling_nominal_R=10e3,
     tc_covariance_table='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',R=10e3,R_max=150e3)
 configure_measured_pooling(model,wrappers['wrappers'],enable_nonideality=True,seed=4096,**pooling_options(pool_args,wrappers['wrappers']))
if a.input_kind=='scangfi':
 scan_file='cifar100_raw.h5' if a.dataset=='cifar100' else 'cifar10_raw.h5'
 with h5py.File(REPO_ROOT.parent/'cifar-10-data/scanGFI'/scan_file) as f:
  idx=np.flatnonzero(~f['train'][:])[a.input_offset:a.input_offset+a.batch]
  x=torch.from_numpy(f['images'][idx]).cuda().clamp(0,1)
else:
 rgb_file=(REPO_ROOT.parent/'data/cifar-100-python/train' if a.dataset=='cifar100'
           else REPO_ROOT.parent/'data/cifar-10-batches-py/data_batch_1')
 with open(rgb_file,'rb') as f:
  rgb=pickle.load(f,encoding='bytes')[b'data']
 idx=np.arange(a.input_offset,a.input_offset+a.batch)
 x=torch.from_numpy(rgb[idx]).reshape(-1,3,32,32).cuda().float().div_(255)
 stats=((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)) if a.dataset=='cifar100' else ((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
 mean=x.new_tensor(stats[0]).view(1,3,1,1)
 std=x.new_tensor(stats[1]).view(1,3,1,1)
 x=(x-mean)/std

if a.sensitivity_projections:
 from tc_shared_sensitivity import run_sensitivity
 run_sensitivity(model,x,a.sensitivity_projections,a.output,
     dict(checkpoint=f'saved_ckpt/{name}/{name}_full_param_best_ckpt.pth',
          input_indices=idx.tolist(),mode=a.mode,batch=a.batch,seed=107))
 sys.exit(0)

model.train()
runtime_model=model
if a.mode=='legacy':
 from trainer import WrappedNoisyModel
 # Same pre-quantization functional parameter-mismatch wrapper as legacy FT.
 runtime_model=WrappedNoisyModel(model,noise_levels=a.legacy_mismatch,noise_type='mul')
optimizer=torch.optim.SGD(model.parameters(),lr=.005) if a.optimizer_step else None
stage_counts=collections.Counter()
rhs_counts=collections.Counter()

def reset_benchmark_rng():
 # Every configuration sees the same dropout and sampled hardware realization.
 torch.manual_seed(107);torch.cuda.manual_seed_all(107);np.random.seed(107)
 seen=set()
 def reset(value):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,torch.Generator):value.manual_seed(value.initial_seed())
  elif isinstance(value,dict):
   for item in value.values():reset(item)
  elif isinstance(value,(list,tuple)):
   for item in value:reset(item)
  elif isinstance(value,types.SimpleNamespace):
   for item in vars(value).values():reset(item)
 for module in model.modules():
  for value in vars(module).values():reset(value)

def set_training_step_reuse(enabled):
 for block in model.PcConvs:
  for name in ('option_aca','option_init','option_patch'):
   option=getattr(block,name,None)
   if option is not None:option['reuse_accepted_step_training']=enabled
 for wrapper in wrappers.get('wrappers',[]):
  for name in ('orig_option_aca','orig_option_init','orig_option_patch'):
   option=getattr(wrapper,name,None)
   if option is not None:option['reuse_accepted_step_training']=enabled

def set_rhs_checkpoint(enabled):
 for block in model.PcConvs:
  for name in ('option_aca','option_init','option_patch'):
   option=getattr(block,name,None)
   if option is not None:option['checkpoint_ode_rhs_training']=enabled
 for wrapper in wrappers.get('wrappers',[]):
  for name in ('orig_option_aca','orig_option_init','orig_option_patch'):
   option=getattr(wrapper,name,None)
   if option is not None:option['checkpoint_ode_rhs_training']=enabled

if a.count_stages:
 class CountedRHS(torch.nn.Module):
  def __init__(self,fn):
   super().__init__()
   if isinstance(fn,torch.nn.Module):self.fn=fn
   else:self.__dict__['fn']=fn
   for name in ('tc_context','energy_meter'):
    if hasattr(fn,name):setattr(self,name,getattr(fn,name))
  def forward(self,*args,**kwargs):
   rhs_counts['grad_enabled' if torch.is_grad_enabled() else 'no_grad']+=1
   return self.fn(*args,**kwargs)
 for block in model.PcConvs:
  make_original=block._make_ode_fn
  def make_counted(self,*args,_make_original=make_original,**kwargs):
   return CountedRHS(_make_original(*args,**kwargs))
  block._make_ode_fn=types.MethodType(make_counted,block)
  original=block._tc_dense_conv
  def counted(self,module,source,curves,_original=original):
   stage_counts['grad_enabled' if torch.is_grad_enabled() else 'no_grad']+=1
   return _original(module,source,curves)
  block._tc_dense_conv=types.MethodType(counted,block)

result=dict(batch=a.batch,dataset=a.dataset,mode=a.mode,
 checkpoint_ode_rhs_training=a.rhs_checkpoint,
 compare_training_optimizations=a.compare_training_optimizations,
 matched_rng_reset=True,
 reference_correction=a.reference_correction,
 training_step_reuse=a.training_step_reuse,
 conv_method=effective_conv_method,curve_sampling=a.curve_sampling,device=torch.cuda.get_device_name(),
 checkpoint_path=str(checkpoint_path),
 enob=a.enob,tol=a.tol,method=a.method,state=a.state,input_kind=a.input_kind,
 effects=sorted(effects),legacy_mismatch_std=a.legacy_mismatch if a.mode=='legacy' else None,
 unrolled=False,teacher=False,optimizer=a.optimizer_step,loss='mean squared logits',
 memory_fraction=a.memory_fraction,seed=107,curve_seed=19,hardware_seeds=4096,
 input_indices=idx.tolist(),input_shape=list(x.shape),rows=[])
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
def save():path.write_text(json.dumps(result,indent=2))
save()
paired=a.compare_sampling or a.compare_correction or a.compare_training_step_reuse or a.compare_rhs_checkpoint
width=4 if a.compare_training_optimizations else 2 if paired else 1
for iteration in range((a.repeats+1)*width):
 repeat=iteration//width
 correction_mode='reference' if a.reference_correction else 'optimized'
 if a.compare_correction:
  correction_mode=('reference','optimized')[(iteration%2)^(repeat%2)]
  tc_shared_correction.shared_correction=prior_correction if correction_mode=='reference' else optimized_correction
 if a.compare_sampling:
  sampling=('uniform','histogram')[(iteration%2)^(repeat%2)]
  for block in model.PcConvs:block._tc_curve_sampling=sampling
 else:sampling=a.curve_sampling
 if a.compare_training_step_reuse:
  training_step_reuse=bool((iteration%2)^(repeat%2))
  set_training_step_reuse(training_step_reuse)
 else:training_step_reuse=a.training_step_reuse
 if a.compare_training_optimizations:
  configurations=((False,False),(True,False),(False,True),(True,True))
  training_step_reuse,rhs_checkpoint=configurations[(iteration+repeat)%4]
  set_training_step_reuse(training_step_reuse)
  set_rhs_checkpoint(rhs_checkpoint)
 elif a.compare_rhs_checkpoint:
  rhs_checkpoint=bool((iteration%2)^(repeat%2))
  set_rhs_checkpoint(rhs_checkpoint)
 else:rhs_checkpoint=a.rhs_checkpoint
 reset_benchmark_rng()
 model.zero_grad(set_to_none=True)
 stage_counts.clear()
 rhs_counts.clear()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 row=dict(repeat=repeat,warmup=repeat==0,phase='forward',sampling=sampling,
          correction=correction_mode,training_step_reuse=training_step_reuse,
          checkpoint_ode_rhs_training=rhs_checkpoint)
 y=None
 try:
  start=time.perf_counter()
  y=runtime_model(x)
  torch.cuda.synchronize();forward_end=time.perf_counter()
  row['phase']='backward'
  y.square().mean().backward()
  if optimizer is not None:optimizer.step()
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
   label=(f'sampling-{sampling}' if a.compare_sampling else
          f'correction-{correction_mode}' if a.compare_correction else
          f'reuse-{str(training_step_reuse).lower()}' if a.compare_training_step_reuse
          else f'rhs-checkpoint-{str(rhs_checkpoint).lower()}' if a.compare_rhs_checkpoint
          else f'reuse-{str(training_step_reuse).lower()}_rhs-checkpoint-{str(rhs_checkpoint).lower()}' if a.compare_training_optimizations
          else 'result')
   tensor_path=path.with_name(f'{path.stem}_{label}_repeat-{repeat}.pt')
   torch.save(dict(logits=y.detach().cpu(),gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}),tensor_path)
   row['tensor_path']=str(tensor_path)
 except torch.OutOfMemoryError as exc:
  row.update(status='OOM',error=str(exc),peak_allocated=torch.cuda.max_memory_allocated(),
             peak_reserved=torch.cuda.max_memory_reserved())
 finally:
  del y
 result['rows'].append(row);save();print(json.dumps(row),flush=True)
 if row['status']!='ok':
  if a.compare_training_optimizations or a.compare_rhs_checkpoint:
   model.zero_grad(set_to_none=True);torch.cuda.empty_cache();continue
  break
rows=[r for r in result['rows'] if not r['warmup'] and r['status']=='ok']
metrics=('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')
def summarize(selected):
 if not selected:return dict(status='no successful run')
 return dict(status='ok',**{key:float(np.median([row[key] for row in selected])) for key in metrics})
if rows:
 result['median']=summarize(rows)
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
 if a.compare_training_step_reuse:
  result['by_training_step_reuse']={str(mode).lower():{k:float(np.median(
      [r[k] for r in rows if r['training_step_reuse']==mode]))
      for k in ('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')}
      for mode in (False,True)}
  save();print(json.dumps(result['by_training_step_reuse']),flush=True)
 if a.compare_rhs_checkpoint:
  result['by_rhs_checkpoint']={str(mode).lower():{k:float(np.median(
      [r[k] for r in rows if r['checkpoint_ode_rhs_training']==mode]))
      for k in ('forward_seconds','backward_seconds','total_seconds','peak_allocated','peak_reserved')}
      for mode in (False,True)}
  save();print(json.dumps(result['by_rhs_checkpoint']),flush=True)
 if a.compare_training_optimizations:
  result['by_training_optimizations']={
   f'reuse-{str(reuse).lower()}_rhs-checkpoint-{str(rhs).lower()}':
    summarize([r for r in rows
        if r['training_step_reuse']==reuse and r['checkpoint_ode_rhs_training']==rhs])
   for reuse,rhs in ((False,False),(True,False),(False,True),(True,True))}
  save();print(json.dumps(result['by_training_optimizations']),flush=True)

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
