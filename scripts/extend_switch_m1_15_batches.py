"""Extend five m=1 cases from seven to fifteen batches; never rerun saved batches."""
import concurrent.futures as futures
import fcntl,gc,json,logging,multiprocessing,os,time,traceback
import torch
import study_switch_full_network_r10k as driver
s=driver.study;ROOT=s.ROOT;OLD=driver.reporting.OUT
FRESH=ROOT/'results/switch_fresh_batches_7_8_m1_j15_s7_s8'
OUT=ROOT/'results/switch_m1_accuracy_15_batches'
PAPER=ROOT.parent/'papers/hardware-native-neural-ode/shared/evidence/pixel-switched/m1_accuracy_15_batches.md'
CONFIGS=[('Jacobi',1,15,1e-8),('Jacobi',1,20,1e-8),('Strang',1,7,1e-8),('Strang',1,8,1e-8),('Strang',1,10,1e-8)]
s.OUT=OUT

def link(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():os.link(src,dest)
 assert s.digest(src)==s.digest(dest)

def publish(state,failures):
 rows=[]
 for c in CONFIGS:
  folder=OUT/'cases'/s.case_id(c);p=folder/'summary.json';d=json.loads(p.read_text()) if p.exists() else None
  rows.append((c,len(list(folder.glob('batch_*.pt'))),d))
 s.atomic_json(OUT/'status.json',dict(status=state,completed=sum(d is not None for c,b,d in rows),expected=5,workers=4,cases=[dict(case=s.case_id(c),batches=b,complete=d is not None) for c,b,d in rows],failed_cases=failures,pid=os.getpid(),updated_unix=time.time()))
 text=['# m=1 accuracy comparison: 15 batches','','Same checkpoint and physical settings as prop-324; m=1, mini-tolerance 1e-8, reference Dopri5 1e-6. Original batches 0–6 are reused. The added eight are batches 7–14 (dataset indices 896–1919), each size 128, for 1920 total images. Previously completed batches 7–8 are reused where available. Identical saved noisy inputs and reference outputs are used for all five configurations.','',f'Status: {state}; {sum(d is not None for c,b,d in rows)}/5 configurations complete; {len(failures)} failures. Four workers; starting order: Jacobi N=15, Jacobi N=20, Strang N=7, Strang N=8; Strang N=10 follows.','','| Method | N | Batches | Mean layer error | Layer 16 error | Logit error | Accuracy | Reference mistakes corrected | Reference-correct predictions broken | Total correct |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 rp=OUT/'reference_summary.json'
 if rp.exists():
  ref=json.loads(rp.read_text());text.append(f"| Full reference | — | 15/15 | 0 | 0 | 0 | {100*ref['accuracy']:.2f}% | — | — | {ref['correct']} |")
 for c,b,d in rows:
  vals=[f"{sum(l['output_relative_error'] for l in d['layers'])/16:.6f}",f"{d['layers'][-1]['output_relative_error']:.6f}",f"{d['logit_relative_error']:.6f}",f"{100*d['accuracy']:.2f}%",str(d['helped']),str(d['harmed']),str(round(d['accuracy']*1920))] if d else ['pending']*7
  text.append(f'| {c[0]} | {c[2]} | {b}/15 | '+' | '.join(vals)+' |')
 text+=['',f'Raw records, manifest and hashes: `{OUT}`. Full-coverage metrics remain pending until all fifteen batches for that method are available.','']
 for p in (OUT/'summary.md',PAPER):
  tmp=p.with_suffix('.extend.tmp');tmp.write_text('\n'.join(text));tmp.replace(p)

@torch.no_grad()
def prepare():
 base=json.loads((OLD/'manifest.json').read_text())
 for p,h in base['source_sha256'].items():assert s.digest(ROOT/p)==h,p
 assert s.digest(s.CKPT)==base['checkpoint_sha256']
 fresh=json.loads((FRESH/'manifest.json').read_text())
 assert fresh['converted_parameter_sha256']==base['converted_parameter_sha256']
 assert fresh['source_sha256']==base['source_sha256']
 for folder in ('data','reference_1e-06'):
  for b in range(7):link(OLD/folder/f'batch_{b}.pt',OUT/folder/f'batch_{b}.pt')
  for b in (7,8):link(FRESH/folder/f'batch_{b-7}.pt',OUT/folder/f'batch_{b}.pt')
 reused={}
 for c in CONFIGS:
  name=s.case_id(c);reused[name]=[]
  for b in range(7):link(OLD/'cases'/name/f'batch_{b}.pt',OUT/'cases'/name/f'batch_{b}.pt');reused[name].append(b)
  for b in (7,8):
   src=FRESH/'cases'/name/f'batch_{b-7}.pt'
   if src.exists():link(src,OUT/'cases'/name/f'batch_{b}.pt');reused[name].append(b)
 if not (OUT/'manifest.json').exists():
  s.seed();loader=s.get_test_data(test_bs=128,img_type='scanGFI',task='cifar100',shuffle=False);s.seed()
  for b,(x,labels) in enumerate(loader):
   path=OUT/'data'/f'batch_{b}.pt'
   if path.exists():
    prev=torch.load(path,weights_only=True);assert torch.equal(x,prev['input']) and torch.equal(labels,prev['labels']),f'Input replay mismatch {b}'
   else:s.atomic_torch(path,dict(input=x.cpu(),labels=labels.cpu(),indices=torch.arange(b*128,(b+1)*128),original_batch_index=b))
   if b==14:break
  assert b==14
  manifest=dict(base);manifest.update(n_batches=15,config_count=5,scope='first fifteen frozen test batches; original seven reused',batch_sha256=[s.digest(OUT/'data'/f'batch_{b}.pt') for b in range(15)],input_origin=str(OUT/'data'),parent_manifest_sha256=s.digest(OLD/'manifest.json'),reused_case_batches=reused,reference_policy='reuse references 0..8; evaluate 9..14 once at 1e-6',reference_controls=[],driver_sha256=s.digest(__file__))
  manifest.pop('reference_sha256',None);manifest.pop('input_origin_manifest_sha256',None);s.atomic_json(OUT/'manifest.json',manifest)
 manifest=json.loads((OUT/'manifest.json').read_text());driver.verify_inputs(manifest)
 s.seed();net,wraps=s.build(s.ODEXInitFFFB,1);assert s.weight_digest(net)==base['converted_parameter_sha256']
 for block in net.PcConvs:block.option_aca.update(rtol=1e-6,atol=1e-6)
 for b in range(15):
  path=OUT/'reference_1e-06'/f'batch_{b}.pt'
  if path.exists():continue
  data=torch.load(OUT/'data'/f'batch_{b}.pt',weights_only=True);s.seed();start=time.perf_counter();logits,traces=s.capture_forward(net,wraps,data['input'].cuda());torch.cuda.synchronize();s.atomic_torch(path,dict(logits=logits.cpu(),layers=traces,seconds=time.perf_counter()-start));print('NEW REFERENCE',b,flush=True)
 correct=0
 for b in range(15):
  ref=torch.load(OUT/'reference_1e-06'/f'batch_{b}.pt',weights_only=True);data=torch.load(OUT/'data'/f'batch_{b}.pt',weights_only=True);correct+=int((ref['logits'].argmax(1)==data['labels']).sum())
 manifest['reference_sha256']={f'reference_1e-06/batch_{b}.pt':s.digest(OUT/'reference_1e-06'/f'batch_{b}.pt') for b in range(15)};s.atomic_json(OUT/'manifest.json',manifest);s.atomic_json(OUT/'reference_summary.json',dict(correct=correct,count=1920,accuracy=correct/1920));del net,wraps;gc.collect();torch.cuda.empty_cache()

def worker(c):
 torch.set_num_threads(1);logging.disable(logging.WARNING)
 print('START',s.case_id(c),'pid',os.getpid(),flush=True)
 return driver.run_case(c)

def _run_locked():
 torch.set_num_threads(1);logging.disable(logging.WARNING)
 if torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('Insufficient memory for four workers')
 failures={};publish('preparing',failures);prepare();publish('running',failures)
 with futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  pending={pool.submit(worker,c):c for c in CONFIGS}
  while pending:
   done,_=futures.wait(pending,timeout=15,return_when=futures.FIRST_COMPLETED)
   for f in done:
    c=pending.pop(f)
    try:f.result()
    except BaseException:failures[s.case_id(c)]=traceback.format_exc();print(failures[s.case_id(c)],flush=True)
   publish('running',failures)
 publish('complete' if not failures else 'finished with failures',failures)
 return 1 if failures else 0
def main():
 # Keep ownership through failure publication; rejected launches never write.
 OUT.mkdir(exist_ok=True)
 with open(OUT/'supervisor.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   return _run_locked()
  except BaseException:
   publish('failed',{'supervisor':traceback.format_exc()})
   raise

if __name__=='__main__':
 raise SystemExit(main())
