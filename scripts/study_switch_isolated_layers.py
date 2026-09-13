"""52 isolated-layer configurations, two frozen batches of 128, one worker."""
import csv
import fcntl
import gc
import io
import json
import os
from pathlib import Path
import time
import traceback

import torch
import study_switch_full_network_normsafe as recovery
from switch_study_r10k_model import build

s = recovery.study
ROOT = s.ROOT
REF = ROOT/'results/switch_full_network_r10k_rmax150k_enobnone'
OUT = ROOT/'results/switch_isolated_layers_r10k_rmax150k_enobnone_2b'
CONFIGS = [(method,m,n,1e-8) for method in ('Strang','Lie','Jacobi')
           for n in ((5,7,10) if method=='Strang' else (5,7,10,14,20))
           for m in (1,2,3,4)]
assert len(CONFIGS)==52
KINDS = ('input','pre_quant','output')
CACHE = None


def write_text(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def csv_write(path,fields,rows):
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader();writer.writerows(rows);write_text(path,buffer.getvalue())


def relative(diff,ref):
    return (diff/max(ref,1e-30))**.5


def prepare():
    original=json.loads((REF/'manifest.json').read_text())
    assert (original['R'],original['R_max'],original['enob'])==(10000.,150000.,None)
    sources=dict(original['source_sha256'])
    sources['scripts/study_switch_isolated_layers.py']=s.digest(__file__)
    for name,h in sources.items():assert s.digest(ROOT/name)==h,f'Changed source: {name}'
    assert s.digest(original['checkpoint'])==original['checkpoint_sha256']
    artifacts={}
    for batch in (0,1):
        for relative_path,expected in (
            (f'data/batch_{batch}.pt',original['batch_sha256'][batch]),
            (f'reference_1e-06/batch_{batch}.pt',original['reference_sha256'][f'reference_1e-06/batch_{batch}.pt'])):
            assert s.digest(REF/relative_path)==expected
            artifacts[relative_path]=expected
    manifest=dict(reference_directory=str(REF),reference_manifest_sha256=s.digest(REF/'manifest.json'),
                  reference_artifact_sha256=artifacts,source_sha256=sources,
                  checkpoint=original['checkpoint'],checkpoint_sha256=original['checkpoint_sha256'],
                  converted_parameter_sha256=original['converted_parameter_sha256'],
                  configs=[dict(method=a,m=b,n_iters=c,tol=d) for a,b,c,d in CONFIGS],
                  batch_ids=[0,1],batch_size=128,n_samples=256,layers=16,workers=1,
                  R=10000.,R_max=150000.,enob=None,C=original['C'],v_dd=original['v_dd'],w_bits=5,
                  reference_tol=1e-6,scope='independent wrapped layers on saved reference inputs; no network propagation',
                  numerical_control=original['numerical_control'],
                  gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    assert manifest['cuda_matmul_allow_tf32']==original['cuda_matmul_allow_tf32']
    assert manifest['cudnn_allow_tf32']==original['cudnn_allow_tf32']
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Manifest mismatch on resume'
    else:s.atomic_json(path,manifest)
    return manifest


@torch.no_grad()
def evaluate_layer(block,wrapper,trace,layer,device='cuda'):
    """The only model call here is this layer on its own reference input."""
    x=trace['input'].to(device)
    assert len(x)==128
    captured={}
    original=wrapper._quantize_output
    def quant(value):
        captured['pre_quant']=value.detach().cpu()
        captured['rail_fraction']=(value.abs()>=wrapper.v_dd).float().flatten(1).mean(1).cpu()
        return original(value)
    wrapper._quantize_output=quant
    try:
        output=block(x,layer)
    finally:
        wrapper._quantize_output=original
    assert torch.equal(x.cpu(),trace['input']),'Layer mutated its saved reference input'
    captured['input']=x.cpu()
    captured['output']=output.detach().cpu()
    assert all(bool(torch.isfinite(v).all()) for v in captured.values()),'Nonfinite layer result'
    errors={kind:s.tensor_error(captured[kind],trace[kind]) for kind in KINDS}
    assert not bool(errors['input']['difference_sq'].any())
    return {'index':layer,'output':captured['output'],'pre_quant':captured['pre_quant'],
            'rail_fraction':captured['rail_fraction'],'errors':errors}


def compact(record,batch):
    result={'layer':record['index']+1,'batch':batch,'n_samples':128,
            'seconds':record['seconds'],'peak_allocated_bytes':record['peak_allocated_bytes'],
            'rail_sum':float(record['rail_fraction'].double().sum())}
    for kind in KINDS:
        metric=record['errors'][kind]
        diff=float(metric['difference_sq'].double().sum());ref=float(metric['reference_sq'].double().sum())
        result[kind+'_difference_sq']=diff;result[kind+'_reference_sq']=ref
        result[kind+'_relative_error']=relative(diff,ref)
    return result


def collect(config):
    folder=OUT/'cases'/s.case_id(config)
    batches=[];layers=[]
    for layer in range(16):
        parts=[]
        for batch in (0,1):
            path=folder/'layers'/f'batch_{batch}_layer_{layer:02d}.json'
            if path.exists():
                part=json.loads(path.read_text());parts.append(part);batches.append(part)
        if len(parts)==2:
            result={'layer':layer+1,'n_samples':256,'rail_fraction':sum(p['rail_sum'] for p in parts)/256}
            for kind in KINDS:
                result[kind+'_relative_error']=relative(sum(p[kind+'_difference_sq'] for p in parts),sum(p[kind+'_reference_sq'] for p in parts))
            layers.append(result)
    return batches,layers


def report(status,failures,current=None):
    all_layers=[];all_batches=[];table=[];completed=0;finished_calls=0
    for number,config in enumerate(CONFIGS,1):
        name=s.case_id(config);folder=OUT/'cases'/name
        batches,layers=collect(config);finished_calls+=len(batches)
        done=(folder/'summary.json').exists()
        completed+=done
        state='complete' if done else ('failed' if name in failures else ('running' if folder.exists() else 'pending'))
        meta=dict(id=f'I{number:03d}',case=name,method=config[0],m=config[1],n_iters=config[2],tol=config[3])
        by_layer={x['layer']:x for x in layers};by_batch={(x['batch'],x['layer']):x for x in batches}
        for layer in range(1,17):
            row=dict(meta,layer=layer,status='complete' if layer in by_layer else state)
            row.update(by_layer.get(layer,{}));all_layers.append(row)
            for batch in (0,1):
                part=by_batch.get((batch,layer));row=dict(meta,batch=batch,layer=layer,status='complete' if part else state)
                if part:
                    row.update(n_samples=128,rail_fraction=part['rail_sum']/128,seconds=part['seconds'],peak_allocated_bytes=part['peak_allocated_bytes'])
                    row.update({k+'_relative_error':part[k+'_relative_error'] for k in KINDS})
                all_batches.append(row)
        mean=f"{sum(x['output_relative_error'] for x in layers)/16:.7f}" if done else '—'
        table.append(f"| {meta['id']} | {name} | {state} | {len(batches)}/32 | {mean} |")
    fields=['id','case','method','m','n_iters','tol','layer','status','n_samples',*[k+'_relative_error' for k in KINDS],'rail_fraction']
    csv_write(OUT/'layer_metrics.csv',fields,all_layers)
    csv_write(OUT/'layer_batch_metrics.csv',fields+['batch','seconds','peak_allocated_bytes'],all_batches)
    s.atomic_json(OUT/'status.json',dict(status=status,completed=completed,expected=52,finished_layer_batches=finished_calls,
                 expected_layer_batches=1664,failed_cases=failures,current=current,pid=os.getpid(),updated_unix=time.time()))
    text=['# Isolated-layer diagnostic: 52 configurations','',f'Status: {status}; {completed}/52 configurations complete; {len(failures)} failures.',
          '', 'R=10 kΩ, R_max=150 kΩ, ENOB=None. Mini-tolerance 1e-8; full reference 1e-6.',
          'First two frozen batches, 128 images per batch (256 total), processed separately by one worker.',
          'Each of 16 wrapped layers receives its own saved full-reference input. No switched network propagation.',
          'Mean error is the equal-weight mean of 16 isolated output relative L2 errors, each pooled over 256 images.',
          'This diagnostic produces no end-to-end accuracy or classifier-logit metric. No baseline reference was rerun.',
          '', '[Complete layer CSV](layer_metrics.csv) · [Batch-level layer CSV](layer_batch_metrics.csv) · [Manifest](manifest.json) · [Status](status.json)',
          '', '| ID | Configuration | Status | Layer-batches | Mean isolated layer error |','|---|---|---|---:|---:|',*table]
    write_text(OUT/'summary.md','\n'.join(text)+'\n')


@torch.no_grad()
def run_case(config,manifest,failures):
    global CACHE
    method,m,n,tol=config;name=s.case_id(config);folder=OUT/'cases'/name
    if (folder/'summary.json').exists():return
    for path,h in manifest['source_sha256'].items():assert s.digest(ROOT/path)==h,f'Changed source: {path}'
    if CACHE is None or CACHE[0]!=(method,m):
        CACHE=None;gc.collect();torch.cuda.empty_cache();s.seed()
        cls=s.METHODS[method]
        class Configured(cls):
            def __init__(self,**kwargs):super().__init__(block_size=m,**kwargs)
        net,wraps=build(Configured,n)
        assert s.weight_digest(net)==manifest['converted_parameter_sha256']
        for block in net.PcConvs:
            for bank in block.FBconv_copies+getattr(block,'block_FBconv_copies',[]):
                assert torch.equal(bank.weight,block.FBconv.weight)
        CACHE=((method,m),net,wraps)
    _,net,wraps=CACHE
    (folder/'layers').mkdir(parents=True,exist_ok=True)
    for block in net.PcConvs:block.n_iters=n;block.option_aca.update(rtol=tol,atol=tol)
    for batch in (0,1):
        path=REF/'reference_1e-06'/f'batch_{batch}.pt'
        assert s.digest(path)==manifest['reference_artifact_sha256'][str(path.relative_to(REF))]
        ref=torch.load(path,map_location='cpu',weights_only=True)
        assert len(ref['layers'])==16
        for layer in range(16):
            dest=folder/'layers'/f'batch_{batch}_layer_{layer:02d}.pt'
            if dest.exists() and dest.with_suffix('.json').exists():continue
            s.seed();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            record=evaluate_layer(net.PcConvs[layer],wraps[layer],ref['layers'][layer],layer)
            torch.cuda.synchronize()
            record.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            s.atomic_torch(dest,record);s.atomic_json(dest.with_suffix('.json'),compact(record,batch))
            del record
            current=dict(case=name,batch=batch,layer=layer+1)
            report('running',failures,current)
            print(f'{name} batch {batch+1}/2 layer {layer+1}/16 saved',flush=True)
        records=[torch.load(folder/'layers'/f'batch_{batch}_layer_{layer:02d}.pt',map_location='cpu',weights_only=True) for layer in range(16)]
        s.atomic_torch(folder/f'batch_{batch}.pt',dict(batch_id=batch,n_samples=128,layers=records,
                       reference_sha256=manifest['reference_artifact_sha256'][str(path.relative_to(REF))]))
        del records,ref
    batches,layers=collect(config)
    assert len(layers)==16 and len(batches)==32
    s.atomic_json(folder/'summary.json',dict(case=name,method=method,m=m,n_iters=n,tol=tol,n_samples=256,
                 mean_layer_relative_error=sum(x['output_relative_error'] for x in layers)/16,layers=layers,
                 seconds=sum(x['seconds'] for x in batches),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in batches)))
    print('COMPLETE '+name,flush=True)


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures={}
    if (OUT/'status.json').exists():
        s.atomic_json(OUT/f'status_before_resume_{time.time_ns()}.json',json.loads((OUT/'status.json').read_text()))
    report('validating saved reference',failures)
    try:
        manifest=prepare()
    except Exception:
        report('failed during validation',{'preparation':traceback.format_exc()})
        raise
    # Fast successful cases establish memory/recording behavior before the long
    # singleton Strang cases; this reorders execution only, not comparison IDs.
    pilot=[('Jacobi',1,5,1e-8)]+[('Strang',m,5,1e-8) for m in (4,3,2,1)]
    ordered=pilot+[c for c in CONFIGS if c not in pilot]
    for config in ordered:
        try:run_case(config,manifest,failures)
        except Exception:
            failures[s.case_id(config)]=traceback.format_exc()
            print('FAILED '+s.case_id(config)+'\n'+failures[s.case_id(config)],flush=True)
            global CACHE
            CACHE=None;gc.collect();torch.cuda.empty_cache()
        report('running',failures)
    report('complete' if not failures else 'finished with failures',failures)
    if failures:raise RuntimeError(f'{len(failures)} failed cases')


if __name__=='__main__':main()
