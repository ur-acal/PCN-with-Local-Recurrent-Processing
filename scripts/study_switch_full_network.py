"""Persistent matched full-network switching study (no production math changes).

Freezes seven actual test batches, builds three full-reference tolerance controls,
then evaluates all 324 requested switching configurations. Per-batch artifacts
are restartable; the supervisor reports failures rather than silently dropping jobs.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'docs/switched_dynamics_diagnostic_snapshot'), str(ROOT)]
from debug_strang_real_block import build, CKPT
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import (ODEXInitFFFBPixelSwitchEfficient as Jacobi,
                    ODEXInitFFFBPixelSwitchExplicit as Lie,
                    ODEXInitFFFBPixelSwitchStrang as Strang)

OUT = ROOT / 'results/switch_full_network_matched'
METHODS = {'Jacobi': Jacobi, 'Lie': Lie, 'Strang': Strang}
ITERS = (5, 6, 7, 8, 9, 10, 12, 15, 20)
TOLS = (1e-6, 1e-7, 1e-8)
SOURCE_FILES = ('switch.py', 'ode_pc.py', 'pc_model.py', 'inference_utils.py',
                'TorchDiffEqPack/odesolver/adaptive_grid_solver.py',
                'TorchDiffEqPack/odesolver/base.py',
                'TorchDiffEqPack/odesolver/ode_solver.py')
_MODEL_CACHE = None


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed():
    random.seed(4096)
    np.random.seed(4096)
    torch.manual_seed(4096)
    torch.cuda.manual_seed_all(4096)


def weight_digest(net):
    h = hashlib.sha256()
    for name, p in net.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_error(a, b):
    a, b = a.detach().float().flatten(1), b.detach().float().flatten(1)
    d = a-b
    return {'difference_sq': d.square().sum(1).cpu(),
            'reference_sq': b.square().sum(1).cpu(),
            'max_abs_difference': d.abs().amax(1).cpu()}


def global_relative(metric):
    return float((metric['difference_sq'].sum()/metric['reference_sq'].sum().clamp_min(1e-30)).sqrt())


def prediction_metrics(logits, reference, labels):
    logits, reference, labels = logits.cpu(), reference.cpu(), labels.cpu()
    pred, ref_pred = logits.argmax(1), reference.argmax(1)
    top = reference.topk(2, dim=1).values
    margin = top[:, 0]-top[:, 1]
    delta = (logits-reference).abs().amax(1)
    certified = margin > 2*delta
    changed = pred != ref_pred
    # A diagnostic invariant, not an assumption that smaller L2 ensures accuracy.
    assert not (certified & changed).any(), 'Logit/margin consistency violation'
    correct, ref_correct = pred == labels, ref_pred == labels
    metric = tensor_error(logits, reference)
    return {'count': len(labels), 'correct': int(correct.sum()),
            'reference_correct': int(ref_correct.sum()), 'disagreements': int(changed.sum()),
            'harmed': int((ref_correct & ~correct).sum()),
            'helped': int((~ref_correct & correct).sum()),
            'certified_same': int(certified.sum()), 'prediction': pred,
            'reference_prediction': ref_pred, 'reference_margin': margin,
            'logit_max_abs_delta': delta, 'logit_error': metric}


def capture_forward(net, wrappers, x, reference=None):
    """No change to values: capture inputs, pre-ENOB state, post-block output."""
    traces = [{} for _ in net.PcConvs]
    hooks, originals = [], []
    def record(i, kind, value):
        value = value.detach()
        if reference is None:
            traces[i][kind] = value.cpu()
        else:
            target = reference[i][kind].to(value.device)
            traces[i][kind] = tensor_error(value, target)
        if kind == 'pre_quant':
            traces[i]['rail_fraction'] = (value.abs() >= wrappers[i].v_dd).float().flatten(1).mean(1).cpu()
    for i, (b, w) in enumerate(zip(net.PcConvs, wrappers)):
        def before(module, args, idx=i): record(idx, 'input', args[0])
        def after(module, args, value, idx=i): record(idx, 'output', value)
        hooks += [b.register_forward_pre_hook(before), b.register_forward_hook(after)]
        original = w._quantize_output
        originals.append(original)
        def quant(value, idx=i, fn=original):
            record(idx, 'pre_quant', value)
            return fn(value)
        w._quantize_output = quant
    try:
        logits = net(x)
    finally:
        for hook in hooks: hook.remove()
        for w, original in zip(wrappers, originals): w._quantize_output = original
    assert all({'input', 'pre_quant', 'output'} <= set(t) for t in traces)
    return logits, traces


@torch.no_grad()
def prepare(n_batches):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'data').mkdir(exist_ok=True)
    if not (OUT/'manifest.json').exists():
        seed()
        loader = get_test_data(test_bs=128, img_type='scanGFI', task='cifar100', shuffle=False)
        # Same seed boundary as accuracy evaluation; data saved once, never regenerated per case.
        seed()
        for i, (x, labels) in enumerate(loader):
            atomic_torch(OUT/'data'/f'batch_{i}.pt', {'input': x.cpu(), 'labels': labels.cpu(),
                         'indices': torch.arange(i*128, i*128+len(labels))})
            if i+1 == n_batches: break
        assert i+1 == n_batches
        atomic_json(OUT/'manifest.json', {
            'checkpoint': CKPT, 'checkpoint_sha256': digest(CKPT),
            'source_sha256': {s: digest(ROOT/s) for s in SOURCE_FILES},
            'batch_sha256': [digest(OUT/'data'/f'batch_{i}.pt') for i in range(n_batches)],
            'n_batches': n_batches, 'batch_size': 128, 'seed': 4096,
            'reference_tol': 1e-6, 'reference_controls': list(TOLS),
            'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'scope': 'first seven fixed test batches, not full CIFAR accuracy' if n_batches==7 else 'fixed test batches',
            'enob': 8, 'R': 20000., 'R_max': 300000., 'C': 49e-15,
            'v_dd': .1, 'one_over_q': 1., 'w_bits': 5,
            'nonlinear_R': False, 'noise_level': 0., 'i_leak': None,
            'test_expanded': False, 'model_builder': 'docs/switched_dynamics_diagnostic_snapshot/debug_strang_real_block.py'})
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['n_batches'] == n_batches
    assert manifest['checkpoint_sha256'] == digest(CKPT)
    for s,h in manifest['source_sha256'].items(): assert digest(ROOT/s) == h, f'Code changed: {s}'
    for i,h in enumerate(manifest['batch_sha256']): assert digest(OUT/'data'/f'batch_{i}.pt') == h
    net, wraps = build(ODEXInitFFFB, 1)
    wh = weight_digest(net)
    manifest['converted_parameter_sha256'] = wh
    atomic_json(OUT/'manifest.json', manifest)
    summary = []
    for tol in TOLS:
        folder = OUT/f'reference_{tol:.0e}'
        folder.mkdir(exist_ok=True)
        for b in net.PcConvs: b.option_aca.update(rtol=tol, atol=tol)
        for i in range(n_batches):
            path = folder/f'batch_{i}.pt'
            if path.exists(): continue
            data = torch.load(OUT/'data'/f'batch_{i}.pt', weights_only=True)
            seed()
            start = time.perf_counter()
            logits, traces = capture_forward(net, wraps, data['input'].cuda())
            torch.cuda.synchronize()
            atomic_torch(path, {'logits': logits.cpu(), 'layers': traces,
                               'seconds': time.perf_counter()-start})
            print(f'REFERENCE {tol:.0e} batch {i+1}/{n_batches}', flush=True)
        records = [torch.load(folder/f'batch_{i}.pt',weights_only=True) for i in range(n_batches)]
        logits = torch.cat([r['logits'] for r in records])
        labels = torch.cat([torch.load(OUT/'data'/f'batch_{i}.pt',weights_only=True)['labels'] for i in range(n_batches)])
        base = torch.cat([torch.load(OUT/'reference_1e-06'/f'batch_{i}.pt',weights_only=True)['logits'] for i in range(n_batches)])
        pm = prediction_metrics(logits, base, labels)
        summary.append({'tol':tol,'accuracy':pm['correct']/pm['count'],
                        'prediction_disagreements_vs_1e6':pm['disagreements'],
                        'logit_relative_error_vs_1e6':global_relative(pm['logit_error'])})
        atomic_json(OUT/'reference_summary.json',summary)
    del net, wraps, records
    torch.cuda.empty_cache()
    return manifest


def case_id(config):
    method,m,n,tol=config
    return f'{method}_m{m}_n{n}_tol{tol:.0e}'


@torch.no_grad()
def run_case(config):
    global _MODEL_CACHE
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    method,m,n,tol=config
    name=case_id(config)
    folder=OUT/'cases'/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists(): return json.loads((folder/'summary.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    for s,h in manifest['source_sha256'].items(): assert digest(ROOT/s)==h, f'Code changed: {s}'
    if _MODEL_CACHE is None or _MODEL_CACHE[0] != (method,m):
        _MODEL_CACHE=None
        torch.cuda.empty_cache()
        cls=METHODS[method]
        class Configured(cls):
            def __init__(self,**kw): super().__init__(block_size=m,**kw)
        seed()
        net,wraps=build(Configured,n)
        assert weight_digest(net)==manifest['converted_parameter_sha256'], 'Physical weights differ from reference'
        for b in net.PcConvs:
            for c in b.FBconv_copies+getattr(b,'block_FBconv_copies',[]):
                assert torch.equal(c.weight,b.FBconv.weight), 'Stale FB bank'
        _MODEL_CACHE=((method,m),net,wraps)
    _,net,wraps=_MODEL_CACHE
    for b in net.PcConvs:
        b.n_iters=n
        b.option_aca.update(rtol=tol,atol=tol)
    for i in range(manifest['n_batches']):
        path=folder/f'batch_{i}.pt'
        if path.exists(): continue
        data=torch.load(OUT/'data'/f'batch_{i}.pt',weights_only=True)
        ref=torch.load(OUT/'reference_1e-06'/f'batch_{i}.pt',weights_only=True)
        seed()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        logits,traces=capture_forward(net,wraps,data['input'].cuda(),ref['layers'])
        torch.cuda.synchronize()
        record={'logits':logits.cpu(),'layers':traces,
                'prediction_metrics':prediction_metrics(logits,ref['logits'],data['labels']),
                'seconds':time.perf_counter()-start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        atomic_torch(path,record)
        pm=record['prediction_metrics']
        atomic_json(folder/'progress.json',{'status':'running','last_finished_batch':i+1,'pid':os.getpid()})
        print(f'{name} BATCH {i+1}/{manifest["n_batches"]}: accuracy={pm["correct"]/pm["count"]:.4f}; ref_disagreements={pm["disagreements"]}',flush=True)
    records=[torch.load(folder/f'batch_{i}.pt',weights_only=True) for i in range(manifest['n_batches'])]
    pms=[r['prediction_metrics'] for r in records]
    total=sum(p['count'] for p in pms)
    combine=lambda field: {key:torch.cat([p[field][key] for p in pms]) for key in pms[0][field]}
    result={'case':name,'method':method,'m':m,'n_iters':n,'tol':tol,'n_samples':total,
            'accuracy':sum(p['correct'] for p in pms)/total,
            'reference_accuracy':sum(p['reference_correct'] for p in pms)/total,
            'disagreement_fraction':sum(p['disagreements'] for p in pms)/total,
            'harmed':sum(p['harmed'] for p in pms),'helped':sum(p['helped'] for p in pms),
            'logit_relative_error':global_relative(combine('logit_error')),
            'seconds':sum(r['seconds'] for r in records),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in records),'layers':[]}
    for j in range(len(net.PcConvs)):
        layer={'index':j}
        for kind in ('input','pre_quant','output'):
            metric={key:torch.cat([r['layers'][j][kind][key] for r in records]) for key in records[0]['layers'][j][kind]}
            layer[kind+'_relative_error']=global_relative(metric)
        layer['rail_fraction']=float(torch.cat([r['layers'][j]['rail_fraction'] for r in records]).mean())
        result['layers'].append(layer)
    atomic_json(folder/'summary.json',result)
    atomic_json(folder/'progress.json',{'status':'complete','last_finished_batch':manifest['n_batches'],'pid':os.getpid()})
    print('COMPLETE '+name,flush=True)
    return result


def report(status, failures):
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*/summary.json'))]
    atomic_json(OUT/'status.json',{'status':status,'completed':len(rows),'expected':324,'failed_cases':failures})
    inversions=[]
    for a in rows:
        for b in rows:
            if a['m']<=b['m'] or (a['method'],a['n_iters'],a['tol'])!=(b['method'],b['n_iters'],b['tol']): continue
            if a['logit_relative_error']<b['logit_relative_error'] and abs(a['accuracy']-a['reference_accuracy'])>abs(b['accuracy']-b['reference_accuracy']):
                inversions.append({'lower_error_case':a['case'],'other_case':b['case'],
                                   'lower_error_case_acc_gap':abs(a['accuracy']-a['reference_accuracy']),
                                   'other_acc_gap':abs(b['accuracy']-b['reference_accuracy'])})
    atomic_json(OUT/'error_accuracy_inversions.json',inversions)
    lines=['# Matched full-network switching study','',f'Status: {status}; {len(rows)}/324 complete.',
           '','Fixed full reference: Dopri5 1e-6. Same saved 896 test inputs (7 batches of 128).',
           'These are subset accuracies, not full-dataset accuracies. Raw paired logits, margins,',
           'prediction changes and per-image layer error statistics are saved per batch.',
           '','| Method | m | Iterations | Tol | Acc (%) | Reference (%) | Disagreement (%) | Logit error | Seconds |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['m']} | {r['n_iters']} | {r['tol']:.0e} | {100*r['accuracy']:.4f} | {100*r['reference_accuracy']:.4f} | {100*r['disagreement_fraction']:.4f} | {r['logit_relative_error']:.7f} | {r['seconds']:.1f} |")
    lines+=['','## Lower error but greater accuracy gap', '',f'{len(inversions)} matched larger-m/smaller-m pairs so far.',
            'See error_accuracy_inversions.json. A scalar L2 improvement is not a proof of prediction agreement;',
            'each prediction is checked against the reference margin/L-infinity sufficient bound.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batches',type=int,default=7);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'supervisor.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    failures={}
    report('preparing fixed data and references',failures)
    try:
        prepare(args.batches)
        # First reproduce the suspected m1/m3 gap; then complete the entire grid.
        pilot=[('Strang',m,5,1e-8) for m in (1,3,2,4)]
        configs=pilot+[(name,m,n,tol) for name in METHODS for m in (1,2,3,4) for n in ITERS for tol in TOLS
                       if (name,m,n,tol) not in pilot]
        assert len(configs)==324
        ctx=multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
            pending={pool.submit(run_case,c):c for c in configs}
            while pending:
                done,_=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    config=pending.pop(future)
                    try: future.result()
                    except BaseException:
                        failures[case_id(config)]=traceback.format_exc()
                        print('FAILED '+case_id(config)+'\n'+failures[case_id(config)],flush=True)
                report('running',failures)
        report('complete' if not failures else 'finished with failures',failures)
        if failures: raise RuntimeError(f'{len(failures)} cases failed; see status.json')
        print('STUDY COMPLETE: 324/324',flush=True)
    except BaseException:
        report('failed/interrupted',failures)
        traceback.print_exc()
        raise


if __name__=='__main__':main()
