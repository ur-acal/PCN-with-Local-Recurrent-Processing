"""Isolated consolidation validation and min/max-checkpoint FT measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'diagnostic_scripts'),str(ROOT/'tests')]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def performance_table(root,records):
    """Generated results only; never turn an unfinished search into a minimum."""
    lines=['| State | Distribution | N | Backend | Individually covered weights | Minimum checkpoints | Seconds at minimum | Peak GiB at minimum | Seconds at 22/22 | Peak GiB at 22/22 |',
           '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    keys=dict.fromkeys((r['state'],r['mode'],r['n'],r['variant']) for r in records)
    for key in keys:
        rows=[r for r in records if (r['state'],r['mode'],r['n'],r['variant'])==key]
        full=next(r for r in rows if r['checkpoint_layers']==22)
        ok=[r for r in rows if r['status']=='ok']
        best=min(ok,key=lambda r:r['checkpoint_layers']) if ok else None
        completed={r['checkpoint_layers'] for r in rows}
        proven=best is not None and set(range(best['checkpoint_layers']))<=completed
        def stats(row):
            if row is None:return ('—','—')
            if row['status']!='ok':return ('OOM','OOM')
            return (f"{row['median_seconds']:.3f}",f"{row['peak_allocated_including_warmup_bytes']/2**30:.3f}")
        coverage=full.get('weight_coverage',{}).get('total',{}).get('individually_covered_weights','—')
        minimum=str(best['checkpoint_layers']) if proven else ('none fits' if len(completed)==23 else 'searching')
        a,b=stats(best if proven else None);c,d=stats(full)
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {coverage} | {minimum} | {a} | {b} | {c} | {d} |')
    lines+=['','Batch 128; two warm-up and five measured optimizer steps per fresh process. Peak allocated memory includes warm-up.',
            'Numerical acceptance is reported separately in the parity JSON files; timing does not imply a path passed those checks.']
    (root/'performance.md').write_text('\n'.join(lines)+'\n')


def idle():
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('GPU is occupied; no existing jobs changed: '+busy)


def micro(output,transpose=False):
    """Same sampled curves, shapes and gradients; preparation excluded from timing."""
    idle()
    import torch
    from types import SimpleNamespace
    from torch import nn
    from tc_curve_groups import sample_groups,group_convolution
    from tc_nonidealities import prepare_tc_resistance_curves
    from tc_empirical import prepare_empirical_curves
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    levels=torch.arange(16,device='cuda')/15
    packages={
        'gaussian':prepare_tc_resistance_curves('hardware_data/res_vs_vin_10k_150k.csv',
            'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',levels=levels,R=1e4,R_max=150000.,dtype=torch.float32,device='cuda'),
        'empirical':prepare_empirical_curves('hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog',
            levels=levels,R=13300.,R_max=200000.,dtype=torch.float32,device='cuda')}
    records=[]
    for mode,p in packages.items():
        for n in (0,1,2,3,15):
            for channels,side in ((16,32),(32,16),(64,8)):
                torch.manual_seed(71)
                module=(nn.ConvTranspose2d if transpose else nn.Conv2d)(channels,channels,3,padding=1,bias=False).cuda()
                codes=torch.arange(module.weight.numel(),device='cuda').reshape_as(module.weight).remainder(16)
                with torch.no_grad():module.weight.copy_(codes/15)
                draw=(p.sample_shared(codes,generator=torch.Generator(device='cuda').manual_seed(19))
                      if mode=='gaussian' and n==0 else
                      sample_groups(p,codes,n,generator=torch.Generator(device='cuda').manual_seed(19)))
                owner=SimpleNamespace(v_dd=.1,_tc_curve_package=p)
                data=torch.rand(128,channels,side,side,device='cuda')*.2-.1
                reference=None
                for name,fused,batched,grouped in (
                    ('off',False,False,False),('scalar_serial',True,False,False),
                    ('scalar_grouped',True,False,True),('batched_grouped',True,True,True)):
                    if n==0 and name=='scalar_serial':continue
                    owner._tc_fused_correction=fused
                    owner._tc_batched_correction=batched
                    owner._tc_grouped_convolution=grouped
                    def run():
                        x=data.detach().requires_grad_()
                        y=group_convolution(owner,module,x,draw)
                        gradients=torch.autograd.grad(y.square().mean(),(x,module.weight))
                        return (y.detach(),*gradients)
                    answer=run()
                    if reference is None:reference=[v.clone() for v in answer]
                    error=[float((a-b).abs().max()) for a,b in zip(answer,reference)]
                    passed=all(torch.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(answer,reference))
                    del answer
                    for _ in range(2):run()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    seconds=[]
                    for _ in range(5):
                        torch.cuda.synchronize();start=time.perf_counter();run();torch.cuda.synchronize()
                        seconds.append(time.perf_counter()-start)
                    row=dict(mode=mode,n=n,channels=channels,side=side,transpose=transpose,variant=name,batch_size=128,
                        passed=passed,max_abs_output_input_weight=error,median_seconds=statistics.median(seconds),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    records.append(row);write(output/'micro.json',records)
                    print('MICRO',mode,n,channels,name,row['median_seconds'],passed,flush=True)
                del reference,data,draw,module
    return records


def commands(state,mode,n,root,batch,checkpoint):
    parents={1:'tc_rgb_cifar100_state1_pcn_resnet_depth_study_student_teacher_no_norm',
             2:'tc_rgb_cifar100_state2_pcn_22L7l6l6_pipeline_student_teacher_no_norm_random_per_forward_per_spin_relu_rgb_mid'}
    candidates=[p for p in (ROOT/'saved_ckpt_runs'/parents[state]).glob('*/*_last_ckpt.pth') if not p.parent.name.startswith('TIMMQAT')]
    if len(candidates)!=1:raise ValueError(candidates)
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
        TASK='cifar100',IMG_TYPE='rgb',CKPT='last',TC_STATE=str(state),TC_DRY_RUN='true',
        MODEL_NAME=candidates[0].parent.name,MODEL_DIR=str(candidates[0].parent.parent),OUTPUT_DIR=str(root/'artifacts'),
        NORMALIZE_STUDENT_INPUT='false',CENTER_STUDENT_INPUT='false',FT_TIMM_AUG_LEVEL='rgb_mid',
        ACTIVATION_CORNER_MODE='random_per_forward',ACTIVATION_RANDOM_CURVE_SHARING='per_spin',
        TEACHER_CKPT='./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill_NoNorm.pth',
        TEACHER_ARCH='efficientnet_v2_l',TEACHER_ARCH_SOURCE='torchvision',
        NONLINEAR_R_CURVE_SAMPLING=mode,
        TC_EMPIRICAL_CURVE_BANK=str(ROOT/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'),
        R_VAL='10e3' if mode=='multivariate_gaussian' else '13.3e3',
        R_MAX='150e3' if mode=='multivariate_gaussian' else '200e3',
        TC_CONV_METHOD='shared',TC_CURVE_TOP_N=str(n),
        REUSE_ACCEPTED_STEP_TRAINING='true',CHECKPOINT_ODE_RHS_TRAINING=str(checkpoint>0).lower(),
        CHECKPOINT_ODE_RHS_PORTION=str(checkpoint/22),FUSE_MEASURED_ACTIVATION='true',
        REHEARSAL_ONLY='true',REHEARSAL_BATCHES='1',REHEARSAL_ARTIFACT_ROOT=str(root/'artifacts'),
        PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        BENCHMARK_MEASURED_STEPS='5')
    command=shlex.split(subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],cwd=ROOT,env=env,text=True))[3:]
    return command+['--batch_size',str(batch),'--num_workers','0'],env


def run_case(root,state,mode,n,variant,checkpoint,verify=False,ordered=False,original=None):
    idle()
    tag=f's{state}_{mode}_n{n}_{variant}_cp{checkpoint}'
    report=root/(tag+'.json')
    if report.exists():raise FileExistsError(report)
    command,env=commands(state,mode,n,root/tag,4 if verify else 128,checkpoint)
    env.update(BENCHMARK_FUSED_CORRECTION='off' if variant=='off' else 'on',
               BENCHMARK_GROUPED_CONVOLUTION='off' if variant=='off' else 'on',
               BENCHMARK_BATCHED_CORRECTION='on')
    worker=[sys.executable,'-u',str(ROOT/'diagnostic_scripts/benchmark_tc_top_n.py'),
            '--worker','--n',str(n),'--report',str(report)]
    if verify:worker.append('--verify')
    if ordered:worker.append('--ordered-groups')
    if original:worker+=['--original',str(original)]
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('tc_curve_groups.py','tc_empirical.py','tc_shared_correction.py','ode_pc.py','physical_feedforward_tc.py')}
    print('START',tag,flush=True)
    with (root/(tag+'.log')).open('w') as log:
        result=subprocess.run(worker+['--',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():raise RuntimeError(f'Worker failed ({result.returncode}): {root/(tag+".log")}')
    record=dict(json.loads(report.read_text()),state=state,mode=mode,n=n,variant=variant,checkpoint_layers=checkpoint,
        source_sha256=source)
    if original:
        record['pre_edit_dispatch_and_kernel_sha256']={name:hashlib.sha256((original.parent/name).read_bytes()).hexdigest()
            for name in ('ode_pc.py','tc_shared_correction.py')}
    if result.returncode and not verify:raise RuntimeError(f'Worker failed: {root/(tag+".log")}')
    print('FINISH',tag,record.get('status',record.get('passed')),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=('micro','verify','ordered','memory','baseline'),required=True)
    p.add_argument('--original',type=Path,help='Pre-edit ode_pc.py snapshot for the Gaussian N=0 timing check.')
    p.add_argument('--ns',nargs='+',type=int,default=[0,1,2,3,15])
    p.add_argument('--states',nargs='+',type=int,default=[1,2])
    p.add_argument('--wait-idle',action='store_true')
    p.add_argument('--transpose',action='store_true',help='Measure transposed convolutions in the micro phase.')
    args=p.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT)
    if args.wait_idle:
        while True:
            try:idle();break
            except RuntimeError:print('WAITING: existing GPU jobs unchanged',flush=True);time.sleep(20)
    if args.phase=='micro':micro(root,args.transpose);return
    records=[]
    if args.phase=='baseline':
        if args.original is None:p.error('--phase baseline requires --original')
        for state in args.states:
            for checkpoint in (22,0):
                records.append(run_case(root,state,'multivariate_gaussian',0,'before',checkpoint,original=args.original))
                write(root/'summary.json',records)
        return
    for state in args.states:
        for mode in ('multivariate_gaussian','empirical_with_replacement'):
            for n in args.ns:
                if args.phase in ('verify','ordered'):
                    records.append(run_case(root,state,mode,n,'parity',22,verify=True,ordered=args.phase=='ordered'))
                else:
                    for variant in ('off','on'):
                        maximum=run_case(root,state,mode,n,variant,22)
                        records.append(maximum)
                        write(root/'summary.json',records)
                        performance_table(root,records)
                        # Ascending search proves minimality for this exact protocol;
                        # do not assume monotonically increasing memory/solver work.
                        for count in range(22):
                            row=run_case(root,state,mode,n,variant,count)
                            records.append(row)
                            write(root/'summary.json',records)
                            performance_table(root,records)
                            if row['status']=='ok':break
                write(root/'summary.json',records)
    if args.phase in ('verify','ordered') and not all(r['passed'] for r in records):
        raise SystemExit('Some OFF/ON checks failed; see retained per-variant metrics.')


if __name__=='__main__':main()
