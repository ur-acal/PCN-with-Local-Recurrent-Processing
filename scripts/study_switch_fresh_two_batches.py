"""Independent held-out two-batch check; reuses unchanged production study helpers."""
import gc
import json
import logging
import os
import time
import traceback
import fcntl
from pathlib import Path
import torch
import study_switch_full_network_r10k as driver
s=driver.study
ORIGIN=driver.reporting.OUT
OUT=s.ROOT/'results/switch_fresh_batches_7_8_m1_j15_s7_s8'
s.OUT=OUT
PAPER=s.ROOT.parent/'papers/hardware-native-neural-ode/shared/evidence/pixel-switched/fresh_batches_7_8.md'
CONFIGS=[('Jacobi',1,15,1e-8),('Strang',1,7,1e-8),('Strang',1,8,1e-8)]

def status(state,**extra):
    s.atomic_json(OUT/'status.json',dict(status=state,pid=os.getpid(),updated_unix=time.time(),**extra))

def report():
    refpath=OUT/'reference_summary.json'
    ref=json.loads(refpath.read_text()) if refpath.exists() else None
    rows=[]
    for c in CONFIGS:
        p=OUT/'cases'/s.case_id(c)/'summary.json'
        rows.append(json.loads(p.read_text()) if p.exists() else None)
    lines=['# Fresh two-batch accuracy/error check','',
           'PIXEL_SWITCHED software evaluation. New test indices 896–1151 (loader batches 7 and 8, zero-based), 256 images total; each batch remains size 128. These images are disjoint from the original seven-batch study. Inputs, including noise, are frozen once and reused by every method.', '',
           'Same checkpoint and R=10 kΩ, R_max=150 kΩ, ENOB=None. All switched cases: m=1, mini-solver tolerance 1e-8. Full unsplit reference: Dopri5 rtol=atol=1e-6, evaluated once on each new batch. Errors are relative to this reference; mean error equally averages 16 pooled layer errors.','',
           '| Method | N | Layer 1 error | Mean layer error | Layer 16 error | Logit error | Accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|']
    lines.append('| Full reference | — | 0 | 0 | 0 | 0 | '+(f"{100*ref['accuracy']:.2f}%" if ref else 'pending')+' |')
    for c,d in zip(CONFIGS,rows):
        if d:
            e=[x['output_relative_error'] for x in d['layers']]
            lines.append(f"| {c[0]} | {c[2]} | {e[0]:.6f} | {sum(e)/16:.6f} | {e[-1]:.6f} | {d['logit_relative_error']:.6f} | {100*d['accuracy']:.2f}% |")
        else:lines.append(f'| {c[0]} | {c[2]} | pending | pending | pending | pending | pending |')
    lines+=['','| Method | Reference mistakes corrected | Reference-correct predictions broken | Total correct |','|---|---:|---:|---:|',f"| Full reference | — | — | {ref['correct'] if ref else 'pending'} |"]
    for c,d in zip(CONFIGS,rows):
        lines.append(f"| {c[0]} N={c[2]} | {d['helped'] if d else 'pending'} | {d['harmed'] if d else 'pending'} | {round(d['accuracy']*256) if d else 'pending'} |")
    lines+=['',f'Raw artifacts: `{OUT}`. Separate from the original seven-batch results; no pooled nine-batch conclusions are implied.','']
    text='\n'.join(lines)
    for p in (OUT/'summary.md',PAPER):
        tmp=p.with_suffix('.fresh.tmp');tmp.write_text(text);tmp.replace(p)
    # Paper-facing compact evidence, separate from large tensor artifacts.
    s.atomic_json(PAPER.with_suffix('.json'),dict(reference=ref,cases=rows))

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True);lock=open(OUT/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);logging.disable(logging.WARNING)
    free,total=torch.cuda.mem_get_info()
    if free<4*1024**3:
        status('stopped: insufficient GPU headroom',free_bytes=free);return
    status('preparing',initial_gpu_free_bytes=free);report()
    original=json.loads((ORIGIN/'manifest.json').read_text())
    for name,h in original['source_sha256'].items():assert s.digest(s.ROOT/name)==h,name
    assert s.digest(s.CKPT)==original['checkpoint_sha256']
    for folder in ('data','reference_1e-06','cases'):(OUT/folder).mkdir(exist_ok=True)
    if not (OUT/'manifest.json').exists():
        s.seed();loader=s.get_test_data(test_bs=128,img_type='scanGFI',task='cifar100',shuffle=False);s.seed()
        for i,(x,labels) in enumerate(loader):
            if i<7:
                prev=torch.load(ORIGIN/'data'/f'batch_{i}.pt',weights_only=True)
                assert torch.equal(labels,prev['labels']) and torch.equal(x,prev['input']),f'Loader replay mismatch batch {i}'
            else:
                s.atomic_torch(OUT/'data'/f'batch_{i-7}.pt',dict(input=x.cpu(),labels=labels.cpu(),indices=torch.arange(i*128,(i+1)*128),original_batch_index=i))
            if i==8:break
        assert i==8
        manifest=dict(original);manifest.update(n_batches=2,scope='new disjoint dataset indices 896:1152; original loader batches 7,8',reference_controls=[],reference_policy='new reference once per new batch at 1e-6',config_count=3,input_origin=str(OUT/'data'),original_batch_indices=[7,8],parent_manifest_sha256=s.digest(ORIGIN/'manifest.json'),driver_sha256=s.digest(__file__))
        manifest.pop('reference_sha256',None);manifest.pop('input_origin_manifest_sha256',None)
        manifest['batch_sha256']=[s.digest(OUT/'data'/f'batch_{i}.pt') for i in range(2)]
        s.atomic_json(OUT/'manifest.json',manifest)
    manifest=json.loads((OUT/'manifest.json').read_text())
    for i,h in enumerate(manifest['batch_sha256']):assert s.digest(OUT/'data'/f'batch_{i}.pt')==h
    net,wraps=s.build(s.ODEXInitFFFB,1);assert s.weight_digest(net)==manifest['converted_parameter_sha256']
    for b in net.PcConvs:b.option_aca.update(rtol=1e-6,atol=1e-6)
    for i in range(2):
        path=OUT/'reference_1e-06'/f'batch_{i}.pt'
        if path.exists():continue
        status('reference',batch=i);data=torch.load(OUT/'data'/f'batch_{i}.pt',weights_only=True);s.seed();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        logits,traces=s.capture_forward(net,wraps,data['input'].cuda());torch.cuda.synchronize()
        s.atomic_torch(path,dict(logits=logits.cpu(),layers=traces,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print(f'REFERENCE batch {i+1}/2 complete',flush=True)
    del net,wraps,logits,traces;gc.collect();torch.cuda.empty_cache()
    correct=0
    for i in range(2):
        ref=torch.load(OUT/'reference_1e-06'/f'batch_{i}.pt',weights_only=True);data=torch.load(OUT/'data'/f'batch_{i}.pt',weights_only=True);correct+=int((ref['logits'].argmax(1)==data['labels']).sum())
    s.atomic_json(OUT/'reference_summary.json',dict(correct=correct,count=256,accuracy=correct/256,tol=1e-6));report()
    for c in CONFIGS:
        free,_=torch.cuda.mem_get_info()
        if free<3*1024**3:status('stopped: insufficient GPU headroom',free_bytes=free);return
        status('running',case=s.case_id(c));driver.run_case(c);report()
    status('complete',completed=3);report();print('COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
