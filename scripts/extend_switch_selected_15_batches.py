"""Extend selected m=1/4 cases using frozen fifteen-batch inputs/references."""
import concurrent.futures as futures
import fcntl,json,multiprocessing,os,traceback
import extend_switch_m1_15_batches as base
s=base.s
CONFIGS=([('Jacobi',4,n,1e-8) for n in (10,15,20)]
         +[('Strang',4,n,1e-8) for n in (5,7,8,10)]
         +[('Lie',m,n,1e-8) for n in (10,15,20) for m in (1,4)])
ALL_CONFIGS=sorted(set(CONFIGS+base.CONFIGS+[('Strang',1,5,1e-8),('Jacobi',1,10,1e-8)]),key=lambda c:({'Jacobi':0,'Lie':1,'Strang':2}[c[0]],c[2],c[1]))
PAPER=base.PAPER

def publish(state,failures):
 rows=[];cases=[]
 for c in ALL_CONFIGS:
  folder=base.OUT/'cases'/s.case_id(c);p=folder/'summary.json'
  d=json.loads(p.read_text()) if p.exists() else None
  b=len(list(folder.glob('batch_*.pt')))
  if d:assert d['n_samples']==1920,(c,d['n_samples'])
  cases.append(dict(case=s.case_id(c),batches=b,complete=d is not None,new_extension=c in CONFIGS))
  vals=[f"{sum(l['output_relative_error'] for l in d['layers'])/16:.6f}",f"{d['layers'][-1]['output_relative_error']:.6f}",f"{d['logit_relative_error']:.6f}",f"{100*d['accuracy']:.2f}%",str(d['helped']),str(d['harmed']),str(round(d['accuracy']*1920))] if d else ['pending']*7
  rows.append(f'| {c[0]} | {c[2]} | {c[1]} | {b}/15 | '+' | '.join(vals)+' |')
 completed=sum(c['complete'] for c in cases if c['new_extension'])
 s.atomic_json(base.OUT/'status_selected.json',dict(status=state,expected=13,completed=completed,workers=4,cases=cases,failed_cases=failures,pid=os.getpid()))
 ref=json.loads((base.OUT/'reference_summary.json').read_text())
 text='\n'.join(['# Selected switched-dynamics comparisons: 15 batches','','Same checkpoint and physical settings as prop-324; mini-tolerance 1e-8, reference Dopri5 1e-6, batch size 128, 1920 images. Original batches 0–6 are reused without evaluation; only batches 7–14 are evaluated for the 13 extensions. All fifteen saved noisy inputs and reference outputs are reused identically. Seven completed m=1 configurations are retained below.','',f'Status: {state}; {completed}/13 extensions complete; {len(failures)} failures; four workers. Pending rows do not report seven-batch metrics as fifteen-batch results.','', '| Method | N | m | Batches | Mean layer error | Layer 16 error | Logit error | Accuracy | Reference mistakes corrected | Reference-correct predictions broken | Total correct |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',f"| Full reference | — | — | 15/15 | 0 | 0 | 0 | {100*ref['accuracy']:.2f}% | — | — | {ref['correct']} |",*rows,'',f'Raw records, frozen inputs, references, manifests and hashes: `{base.OUT}`.',''])
 tmp=PAPER.with_suffix('.selected.tmp');tmp.write_text(text);tmp.replace(PAPER)

def _run_locked():

 if base.torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('Insufficient memory for four workers')
 manifest=json.loads((base.OUT/'manifest.json').read_text())
 assert manifest['n_batches']==15 and len(CONFIGS)==13 and len(ALL_CONFIGS)==20
 base.driver.verify_inputs(manifest)
 for name,h in manifest['reference_sha256'].items():assert s.digest(base.OUT/name)==h,name
 for name,h in manifest['source_sha256'].items():assert s.digest(s.ROOT/name)==h,name
 reused={}
 for c in CONFIGS:
  name=s.case_id(c);reused[name]=[]
  for b in range(7):
   base.link(base.OLD/'cases'/name/f'batch_{b}.pt',base.OUT/'cases'/name/f'batch_{b}.pt');reused[name].append(b)
 s.atomic_json(base.OUT/'manifest_selected.json',dict(configs=CONFIGS,all_report_configs=ALL_CONFIGS,reused_case_batches=reused,new_batch_indices=list(range(7,15)),parent_manifest_sha256=s.digest(base.OUT/'manifest.json'),driver_sha256=s.digest(__file__)))
 failures={};publish('running',failures)
 with futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
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
