"""Add Strang N=5/Jacobi N=10; reuse batches 0..6 and all saved references."""
import concurrent.futures as futures
import fcntl,json,multiprocessing,os,traceback
import extend_switch_m1_15_batches as base
s=base.s
CONFIGS=[('Strang',1,5,1e-8),('Jacobi',1,10,1e-8)]
PAPER=base.PAPER.with_name('m1_accuracy_15_batches_n5_n10.md')

def publish(state,failures):
 rows=[];cases=[]
 for c in CONFIGS:
  folder=base.OUT/'cases'/s.case_id(c);p=folder/'summary.json'
  d=json.loads(p.read_text()) if p.exists() else None
  b=len(list(folder.glob('batch_*.pt')))
  cases.append(dict(case=s.case_id(c),batches=b,complete=d is not None))
  vals=[f"{sum(l['output_relative_error'] for l in d['layers'])/16:.6f}",f"{d['layers'][-1]['output_relative_error']:.6f}",f"{d['logit_relative_error']:.6f}",f"{100*d['accuracy']:.2f}%",str(d['helped']),str(d['harmed']),str(round(d['accuracy']*1920))] if d else ['pending']*7
  rows.append(f'| {c[0]} | {c[2]} | {b}/15 | '+' | '.join(vals)+' |')
 s.atomic_json(base.OUT/'status_n5_n10.json',dict(status=state,expected=2,completed=sum(c['complete'] for c in cases),workers=2,cases=cases,failed_cases=failures,pid=os.getpid()))
 text='\n'.join(['# Additional m=1 fifteen-batch comparisons','','m=1; mini-tolerance 1e-8; batch size 128; 1920 images. Original batches 0–6 reused without evaluation; only batches 7–14 evaluated. All fifteen frozen inputs and unsplit references reused from the completed fifteen-batch study.','',f'Status: {state}.','', '| Method | N | Batches | Mean layer error | Layer 16 error | Logit error | Accuracy | Reference mistakes corrected | Reference-correct predictions broken | Total correct |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*rows,''])
 tmp=PAPER.with_suffix('.tmp');tmp.write_text(text);tmp.replace(PAPER)

def _run_locked():

 if base.torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('Insufficient memory for two workers')
 manifest=json.loads((base.OUT/'manifest.json').read_text())
 assert manifest['n_batches']==15
 base.driver.verify_inputs(manifest)
 for name,h in manifest['reference_sha256'].items():assert s.digest(base.OUT/name)==h,name
 for name,h in manifest['source_sha256'].items():assert s.digest(s.ROOT/name)==h,name
 reused={}
 for c in CONFIGS:
  name=s.case_id(c);reused[name]=[]
  for b in range(7):
   base.link(base.OLD/'cases'/name/f'batch_{b}.pt',base.OUT/'cases'/name/f'batch_{b}.pt');reused[name].append(b)
 s.atomic_json(base.OUT/'manifest_n5_n10.json',dict(configs=CONFIGS,reused_case_batches=reused,new_batch_indices=list(range(7,15)),parent_manifest_sha256=s.digest(base.OUT/'manifest.json'),driver_sha256=s.digest(__file__)))
 failures={};publish('running',failures)
 with futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  pending={pool.submit(base.worker,c):c for c in CONFIGS}
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
 base.OUT.mkdir(exist_ok=True)
 with open(base.OUT/'supervisor.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   return _run_locked()
  except BaseException:
   publish('failed',{'supervisor':traceback.format_exc()})
   raise

if __name__=='__main__':
 raise SystemExit(main())
