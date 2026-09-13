"""Independent validation and five colored iso/prop reruns; no baseline mutation."""
import gc,json,os,time,traceback,fcntl
import torch
import study_switch_full_network_r10k as original
import study_switch_isolated_layers as iso
from switch_coloring import ColorLie,ColorStrang,plan_for,validate_numerical_independence
s=original.study;ROOT=s.ROOT;REF=original.reporting.OUT
OUT=ROOT/'results/switch_color_grouped_r10k_2b'
PAPER=ROOT.parent/'papers/hardware-native-neural-ode/shared/evidence/pixel-switched/color_grouped_results.md'
CONFIGS=[('ColorStrang',m,5,1e-8) for m in (2,3,4)]+[('ColorLie',m,10,1e-8) for m in (1,2)]
s.METHODS.update(ColorLie=ColorLie,ColorStrang=ColorStrang)
def status(state,**kw):s.atomic_json(OUT/'status.json',dict(status=state,pid=os.getpid(),updated_unix=time.time(),**kw))
def report():
 rows=[]
 for c in CONFIGS:
  name=s.case_id(c);ip=OUT/'iso/cases'/name/'summary.json';pp=OUT/'prop/cases'/name/'summary.json'
  rows.append(dict(case=name,iso=json.loads(ip.read_text()) if ip.exists() else None,prop=json.loads(pp.read_text()) if pp.exists() else None))
 text=['# Color-grouped Lie/Strang: isolated and propagated results','','PIXEL_SWITCHED software. Same original frozen batches 0 and 1 (256 images, batch size 128), R=10 kΩ, R_max=150 kΩ, ENOB=None. Existing full unsplit Dopri5 1e-6 reference reused; mini-tolerance 1e-8. Original baseline/source unchanged.','','| Configuration | Isolated mean error | Propagated mean error | Propagated layer 16 | Logit error | Accuracy |','|---|---:|---:|---:|---:|---:|']
 for r in rows:
  i,p=r['iso'],r['prop'];iv=f"{i['mean_layer_relative_error']:.6f}" if i else 'pending'
  pv=[f"{sum(x['output_relative_error'] for x in p['layers'])/16:.6f}",f"{p['layers'][-1]['output_relative_error']:.6f}",f"{p['logit_relative_error']:.6f}",f"{p['accuracy']*100:.2f}%"] if p else ['pending']*4
  text.append('| '+r['case']+' | '+iv+' | '+' | '.join(pv)+' |')
 text+=['',f'Complete metrics and tensors: `{OUT}`.',f'Validation, M/K, color memberships and first-block errors: `{OUT}/validation.json`.','No new training; existing raster/Jacobi results remain the baselines.','']
 for p in (OUT/'summary.md',PAPER):
  tmp=p.with_suffix('.color.tmp');tmp.write_text('\n'.join(text));tmp.replace(p)
 s.atomic_json(OUT/'results.json',rows)
def validate_real():
 if (OUT/'validation.json').exists():return
 data=torch.load(REF/'reference_1e-06/batch_0.pt',weights_only=True,map_location='cpu');x=data['layers'][0]['input'][:1].cuda();del data
 s.seed();net,wraps=s.build(s.ODEXInitFFFB,1);ref=net.PcConvs[0](x,0).detach().cpu();del net,wraps;gc.collect();torch.cuda.empty_cache();records=json.loads((OUT/'validation_progress.json').read_text()) if (OUT/'validation_progress.json').exists() else []
 for m in (1,2,3,4):
  for method,cls in [('raster',s.Strang),('color',ColorStrang)]:
   if any(r['method']==method and r['m']==m for r in records):continue
   class Configured(cls):
    def __init__(self,**kw):super().__init__(block_size=m,**kw)
   s.seed();net,wraps=s.build(Configured,5);op=net.PcConvs[0];op.option_aca.update(rtol=1e-8,atol=1e-8)
   checked=[False];flow=op.local_block_flow;probes=[]
   def audit(xlocal,y,block,duration):
    if not checked[0]:
     checked[0]=True
     if method=='color':probes.append(validate_numerical_independence(op,xlocal,y,atol=1e-6,rtol=1e-6,max_pairs_per_color=1))
    return flow(xlocal,y,block,duration)
   op.local_block_flow=audit;t=time.perf_counter();z=op(x,0).detach().cpu();e=float(torch.linalg.vector_norm(z-ref)/torch.linalg.vector_norm(ref));geometry=[]
   for shape in ((16,16),(8,8),(4,4)):
    bs,ds,gs=plan_for(op,shape);geometry.append(dict(shape=shape,M=len(bs),K=len(gs),groups=[dict(ids=g,blocks=[bs[i] for i in g]) for g in gs],color_stages=2*len(gs),central_halves_merged=False))
   records.append(dict(method=method,m=m,N=5,tol=1e-8,reference_tol=1e-6,relative_error=e,seconds=time.perf_counter()-t,numerical_directed_checks=probes,geometry=geometry));s.atomic_json(OUT/'validation_progress.json',records)
   s.atomic_torch(OUT/f'validation_{method}_m{m}.pt',dict(input=x.cpu(),output=z,reference=ref));print('VALIDATION',method,m,e,flush=True)
   op.local_block_flow=flow;del op,flow,audit,net,wraps;gc.collect();torch.cuda.empty_cache()
 s.atomic_json(OUT/'validation.json',dict(scope='first trained block, one saved image; unsplit reference recomputed on that same image, not batch-128 reference',records=records))
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);lock=open(OUT/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1)
 if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Insufficient GPU headroom')
 status('validating');report();base=json.loads((REF/'manifest.json').read_text())
 for p,h in base['source_sha256'].items():assert s.digest(ROOT/p)==h,p
 validate_real();hashes={**base['source_sha256'],**{p:s.digest(ROOT/p) for p in ('spatial_coloring.py','switch_coloring.py','scripts/study_color_grouped.py')}}
 for folder in ('iso','prop'):(OUT/folder).mkdir(exist_ok=True)
 im=json.loads((ROOT/'results/switch_isolated_layers_r10k_rmax150k_enobnone_2b/manifest.json').read_text());im.update(source_sha256=hashes,configs=CONFIGS,config_count=5);s.atomic_json(OUT/'iso/manifest.json',im)
 pm=dict(base);pm.update(n_batches=2,batch_sha256=base['batch_sha256'][:2],source_sha256=hashes,config_count=5,scope='original frozen batches 0 and 1; colored methods')
 pm['reference_sha256']={k:v for k,v in base['reference_sha256'].items() if k.startswith('reference_1e-06/') and k.endswith(('batch_0.pt','batch_1.pt'))}
 for folder in ('data','reference_1e-06'):
  (OUT/'prop'/folder).mkdir(exist_ok=True)
  for i in (0,1):
   src=REF/folder/f'batch_{i}.pt';dest=OUT/'prop'/folder/f'batch_{i}.pt'
   if not dest.exists():os.link(src,dest)
   assert s.digest(src)==s.digest(dest)
 s.atomic_json(OUT/'prop/manifest.json',pm);iso.OUT=OUT/'iso';iso.CONFIGS=CONFIGS;iso.REF=REF;base_report=iso.report
 def isolated_report(*a,**kw):
  base_report(*a,**kw)
  p=iso.OUT/'status.json';d=json.loads(p.read_text());d.update(expected=5,expected_layer_batches=160);s.atomic_json(p,d)
  p=iso.OUT/'summary.md';text=p.read_text().replace('52 configurations','5 configurations').replace('/52 configurations','/5 configurations');iso.write_text(p,text)
  report()
 iso.report=isolated_report
 for c in CONFIGS:
  if torch.cuda.mem_get_info()[0]<3*1024**3:raise RuntimeError('GPU headroom fell below guard')
  status('isolated',case=s.case_id(c));iso.run_case(c,im,{});isolated_report('running',{})
  iso.CACHE=None;gc.collect();torch.cuda.empty_cache();status('propagated',case=s.case_id(c));s.OUT=OUT/'prop';original.run_case(c);report();s._MODEL_CACHE=None;gc.collect();torch.cuda.empty_cache()
 isolated_report('complete',{});status('complete',configs=5);report();print('COLOR STUDY COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:status('failed',traceback=traceback.format_exc());raise
