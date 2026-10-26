"""Matched real 22-layer TC recipe benchmarks; no production checkpoints changed.

FT measures two warm-ups and three optimizer steps with a cached batch from the
real augmented loader, including the real SRRL teacher. Eval compares complete
expanded-network logits and timings with only empirical fusion toggled.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import shlex
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path, record):
    Path(path).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def module_name(model,target):
    """Include lazy analog-head circuits, which intentionally are not children."""
    from final_linear import AnalogLinear
    for name,module in model.named_modules():
        if module is target:return name
        if isinstance(module,AnalogLinear) and module._circuit is not None:
            for suffix,child in module._circuit.named_modules():
                if child is target:return name+'._circuit'+('.'+suffix if suffix else '')
    raise ValueError('Observed convolution does not belong to the benchmark model.')


def count_rhs_calls(function, counter):
    """Instrument without stripping the state-2 RHS module/solver interface."""
    from functools import wraps
    import torch
    if isinstance(function, torch.nn.Module):
        def count(module, inputs):
            counter[0] += 1
        function.register_forward_pre_hook(count)
        return function
    @wraps(function)
    def counted(*args, **kwargs):
        counter[0] += 1
        return function(*args, **kwargs)
    return counted


def routing_cache_stats(model):
    from tc_empirical import EmpiricalLevels
    tensors={};draws=set()
    for module in model.modules():
        for name in ('_tc_curve_samples','_tc_samples'):
            for draw in (getattr(module,name,None) or {}).values():
                if not isinstance(draw,EmpiricalLevels) or not draw._routing:continue
                draws.add(id(draw))
                for key in ('codes','mask'):
                    value=draw._routing[key]
                    tensors[id(value)]=value.numel()*value.element_size()
    return dict(routing_cache_bytes=sum(tensors.values()),routing_cached_convolutions=len(draws))


def ft(report, argv):
    import torch
    import train_ode_cifar as training
    sys.argv=['train_ode_cifar.py',*argv]
    args=training.get_args()  # Installs the existing bounded initialization hooks.
    training.get_args=lambda:args
    from trainer_timm import TrainerCiFarTimmStyle
    def benchmark(trainer):
        rhs_calls=[0]
        for block in trainer.model.PcConvs:
            make=block._make_ode_fn
            def counted_make(*a,_make=make,**kw):
                return count_rhs_calls(_make(*a,**kw),rhs_calls)
            block._make_ode_fn=counted_make
        cached=os.environ.get('BENCHMARK_LEVEL_ROUTING_CACHE','on')=='on'
        batched=os.environ.get('BENCHMARK_BATCHED_CORRECTION','on')=='on'
        fused=os.environ.get('BENCHMARK_FUSED_CORRECTION','on')=='on'
        grouped=os.environ.get('BENCHMARK_GROUPED_CONVOLUTION','on')=='on'
        for module in trainer.model.modules():
            if getattr(module,'_tc_curve_package',None) is not None or getattr(module,'_tc_resistance_package',None) is not None:
                module._tc_cache_level_routing=cached
                module._tc_batched_correction=batched
                module._tc_fused_correction=fused
                module._tc_grouped_convolution=grouped
        trainer._maybe_profile_ode_rhs_checkpointing()
        metadata=dict(
            checkpoint_selected_layers=getattr(trainer.model,'checkpoint_ode_rhs_selected_layers',None),
            checkpoint_total_layers=getattr(trainer.model,'checkpoint_ode_rhs_total_layers',None),
            convolution=args.tc_conv_method,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            correction_fusion=fused,
            grouped_convolution=grouped,
            level_routing_cache=cached,
            batched_correction=batched,
            rhs_counter='module_pre_hook_or_function_wrapper_v2',
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        batch=next(iter(trainer.train_dataloader))
        assert batch[0].shape[0]==args.batch_size,(batch[0].shape,args.batch_size)
        metadata['actual_batch_shape']=list(batch[0].shape)
        trainer.train_dataloader=[batch]
        rows=[];all_steps=[]
        measured_steps=int(os.environ.get('BENCHMARK_MEASURED_STEPS','3'))
        if measured_steps < 1:
            raise ValueError('BENCHMARK_MEASURED_STEPS must be positive.')
        for i in range(2+measured_steps):
            before_nfe=rhs_calls[0]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            try:
                loss=trainer.train_one_epoch(0)
            except torch.OutOfMemoryError as error:
                write(report,dict(phase='ft',batch_size=args.batch_size,status='OOM',
                    distribution=args.nonlinear_R_curve_sampling,
                    checkpoint_portion=args.checkpoint_ode_rhs_portion,
                    completed_measurements=rows,failed_step=i,
                    peak_allocated_at_failure_bytes=torch.cuda.max_memory_allocated(),
                    oom_message=str(error),**metadata))
                return
            torch.cuda.synchronize()
            row=dict(seconds=time.perf_counter()-start,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss=float(loss),
                     rhs_evaluations=rhs_calls[0]-before_nfe,
                     **routing_cache_stats(trainer.model))
            all_steps.append(row)
            print('BENCHMARK_STEP',i,row,flush=True)
            if i>=2:rows.append(row)
        write(report,dict(phase='ft',status='ok',batch_size=args.batch_size,steps=rows,
            warmups=2,measurements=measured_steps,
            median_seconds=statistics.median(r['seconds'] for r in rows),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_allocated_including_warmup_bytes=max(r['peak_allocated_bytes'] for r in all_steps),
            peak_reserved_including_warmup_bytes=max(r['peak_reserved_bytes'] for r in all_steps),
            routing_cache_bytes=max(r['routing_cache_bytes'] for r in all_steps),
            distribution=args.nonlinear_R_curve_sampling,
            checkpoint_portion=args.checkpoint_ode_rhs_portion,
            **metadata,
            note=f'Two warm-ups, {measured_steps} real measured optimizer steps, cached real augmented batch, real SRRL teacher.'))
    TrainerCiFarTimmStyle.train=benchmark
    training.main()


def routing_ablation(args,output,env,roots):
    feature='batched' if args.batched_correction_ablation else 'routing'
    records=[]
    for state,parent in enumerate(roots,1):
        candidates=[p for p in (ROOT/'saved_ckpt_runs'/parent).glob('*/*_last_ckpt.pth')
                    if not p.parent.name.startswith('TIMMQAT')]
        assert len(candidates)==1,candidates
        variants=('verify',) if args.training_optimizations else ('off','on')
        for variant in variants:
            tag=f'state{state}_{feature}_{variant}'
            report=output/f'{tag}.json'
            if not (args.resume and report.is_file()):
                env.update(TC_STATE=str(state),MODEL_NAME=candidates[0].parent.name,
                    MODEL_DIR=str(candidates[0].parent.parent),OUTPUT_DIR=str(output/tag),
                    NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                    R_VAL='13.3e3',R_MAX='200e3',TC_CONV_METHOD='grouped',
                    CHECKPOINT_ODE_RHS_PORTION='1',REHEARSAL_ONLY='true',
                    REHEARSAL_ARTIFACT_ROOT=str(output/tag),
                    PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'),
                    BENCHMARK_LEVEL_ROUTING_CACHE='on' if feature=='batched' else variant,
                    BENCHMARK_BATCHED_CORRECTION=variant if feature=='batched' else 'off',
                    BENCHMARK_MEASURED_STEPS='5',
                    VERIFY_LEVEL_ROUTING='1' if feature=='routing' else '0',
                    VERIFY_BATCHED_CORRECTION='1' if feature=='batched' else '0')
                printed=subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],
                    cwd=ROOT,env=env,text=True)
                cmd=shlex.split(printed)[3:]+['--batch_size',str(args.batch_size),'--num_workers','0']
                print('START',tag,'batch',args.batch_size,flush=True)
                with (output/f'{tag}.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',__file__,'--worker',
                        'verify' if args.training_optimizations else 'ft',
                        '--report',str(report),'--',*cmd],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            record=dict(json.loads(report.read_text()),state=state,variant=variant)
            records.append(record)
            (output/f'{feature}_summary.json').write_text(json.dumps(records,indent=2)+'\n')
            print('FINISH',tag,record.get('median_seconds'),record.get('routing_cache_bytes'),flush=True)


def checkpoint_sweep(args, output, env, roots):
    """Each checkpoint portion runs in a fresh production-precision FT process."""
    summary=output/'checkpoint_summary.json'
    records=json.loads(summary.read_text()) if args.resume and summary.is_file() else []
    for state,parent in enumerate(roots,1):
        candidates=[p for p in (ROOT/'saved_ckpt_runs'/parent).glob('*/*_last_ckpt.pth')
                    if not p.parent.name.startswith('TIMMQAT')]
        assert len(candidates)==1,candidates
        for count in args.checkpoint_layers:
            if not 0<=count<=22:
                raise ValueError('Checkpoint layer count must be between 0 and 22.')
            tag=f'state{state}_checkpoint{count:02d}'
            report=output/f'{tag}.json'
            if not (args.resume and report.is_file()):
                env.update(TC_STATE=str(state),MODEL_NAME=candidates[0].parent.name,
                    MODEL_DIR=str(candidates[0].parent.parent),OUTPUT_DIR=str(output/tag),
                    NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                    R_VAL='13.3e3',R_MAX='200e3',TC_CONV_METHOD='grouped',
                    CHECKPOINT_ODE_RHS_PORTION=str(count/22),
                    REHEARSAL_ONLY='true',REHEARSAL_ARTIFACT_ROOT=str(output/tag),
                    PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'))
                printed=subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],
                    cwd=ROOT,env=env,text=True)
                cmd=shlex.split(printed)[3:]+['--batch_size',str(args.batch_size),'--num_workers','0']
                print('START',tag,'batch',args.batch_size,'checkpoint',f'{count}/22',flush=True)
                with (output/f'{tag}.log').open('w') as log:
                    completed=subprocess.run([sys.executable,'-u',__file__,'--worker','ft',
                        '--report',str(report),'--',*cmd],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{tag} failed; inspect {output/f"{tag}.log"}')
            record=json.loads(report.read_text())
            # Include startup/warm-up peaks when assessing whether production
            # fits, while retaining the post-warm-up timing/memory fields.
            measured=[ast.literal_eval(line.split(maxsplit=2)[2])
                for line in (output/f'{tag}.log').read_text().splitlines()
                if line.startswith('BENCHMARK_STEP ')]
            if record['status']=='ok':
                # Use the saved workload, not today's environment: --resume can
                # read reports produced with a different measurement count.
                expected=record.get('warmups',2)+len(record['steps'])
                if len(measured)!=expected:
                    raise ValueError(f'{tag}: expected {expected} logged steps, found {len(measured)}')
                record['peak_allocated_including_warmup_bytes']=max(
                    row['peak_allocated_bytes'] for row in measured)
                record['peak_reserved_including_warmup_bytes']=max(
                    row['peak_reserved_bytes'] for row in measured)
                report.write_text(json.dumps(record,indent=2)+'\n')
            assert record['checkpoint_selected_layers']==count,record
            assert record['checkpoint_total_layers']==22,record
            record=dict(record,state=state)
            records=[row for row in records if
                (row['state'],row['checkpoint_selected_layers'])!=(state,count)]
            records.append(record)
            records.sort(key=lambda row:(row['state'],-row['checkpoint_selected_layers']))
            summary.write_text(json.dumps(records,indent=2)+'\n')
            print('FINISH',tag,record['status'],record.get('median_seconds'),
                record.get('peak_allocated_bytes'),flush=True)


def evaluate(report, argv):
    import torch
    import validation
    from tc_cli import reset_after_probe
    # Initialization is not timed. Use the exact original CSR builder on CPU
    # to avoid a GPU scalar .item() synchronization for every expanded edge.
    # Only the resulting CSR tensors move to GPU; arithmetic/order is unchanged.
    original_builder=validation.build_unrolled_csr_trimmed
    def build_on_cpu(*args):
        kernel=args[-1]
        return original_builder(*args[:-1],kernel.detach().cpu()).to(kernel.device)
    validation.build_unrolled_csr_trimmed=build_on_cpu
    original=validation.Validator.__init__
    def initialize(owner,*a,**kw):
        original(owner,*a,**kw)
        model=owner.model.eval()
        x=next(iter(owner.dataloader))[0].to(owner.device)
        modules=[m for m in model.modules() if isinstance(m,validation.MVMConv)]
        def forward(fused):
            for m in modules:m._tc_fused_edges=fused
            reset_after_probe(model)
            torch.manual_seed(4096)
            with torch.no_grad():return model(x)
        expected=forward(False).detach()
        actual=forward(True).detach()
        difference=(actual-expected).abs()
        result=dict(phase='expanded_eval',batch_size=x.shape[0],modules=len(modules),
            max_abs_logits_difference=difference.max().item(),
            rms_logits_difference=difference.square().mean().sqrt().item(),
            prediction_flips=(actual.argmax(1)!=expected.argmax(1)).sum().item(),rows=[])
        print('FULL_NETWORK_LOGITS',result,flush=True)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        # Each complete forward resets hardware/noise RNGs identically.
        for fused in (False,True):
            forward(fused);torch.cuda.synchronize()
            times=[]
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                begin=time.perf_counter();forward(fused);torch.cuda.synchronize()
                times.append(time.perf_counter()-begin)
                print('INFERENCE_STEP',fused,len(times),times[-1],flush=True)
            result['rows'].append(dict(fused=fused,seconds=times,
                median_seconds=statistics.median(times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        write(report,result)
        raise SystemExit(0)
    validation.Validator.__init__=initialize
    sys.argv=['ode_inference.py',*argv]
    runpy.run_path(str(ROOT/'ode_inference.py'),run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('ft','eval','verify'))
    parser.add_argument('--report')
    parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--ft-only',action='store_true')
    parser.add_argument('--training-optimizations',action='store_true')
    parser.add_argument('--routing-cache-ablation',action='store_true')
    parser.add_argument('--batched-correction-ablation',action='store_true')
    parser.add_argument('--checkpoint-layers',type=int,nargs='+',
                        help='Benchmark empirical grouped FT at these layer counts, each out of 22.')
    parser.add_argument('argv',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.worker:
        argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
        if args.worker=='verify':
            from validate_tc_empirical_training import verify
            return verify(args.report,argv)
        return (ft if args.worker=='ft' else evaluate)(args.report,argv)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=args.resume)
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
        TC_DRY_RUN='true',TASK='cifar100',IMG_TYPE='rgb',CKPT='last',
        NORMALIZE_STUDENT_INPUT='false',CENTER_STUDENT_INPUT='false',
        FT_TIMM_AUG_LEVEL='rgb_mid',ACTIVATION_CORNER_MODE='random_per_forward',
        ACTIVATION_RANDOM_CURVE_SHARING='per_spin',
        TEACHER_CKPT='./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill_NoNorm.pth',
        TEACHER_ARCH='efficientnet_v2_l',TEACHER_ARCH_SOURCE='torchvision',
        REUSE_ACCEPTED_STEP_TRAINING='true',CHECKPOINT_ODE_RHS_TRAINING='true',
        CHECKPOINT_ODE_RHS_PORTION=str(2/22),FUSE_MEASURED_ACTIVATION='true',
        EXPANDED_GAUSSIAN_CURVE_CPU_OFFLOAD='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TEST_BS='4',N_TRIALS='1',REHEARSAL_BATCHES='1',
        TC_EMPIRICAL_CURVE_BANK=str(ROOT/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'))
    roots=('tc_rgb_cifar100_state1_pcn_resnet_depth_study_student_teacher_no_norm',
           'tc_rgb_cifar100_state2_pcn_22L7l6l6_pipeline_student_teacher_no_norm_random_per_forward_per_spin_relu_rgb_mid')
    if args.routing_cache_ablation and args.batched_correction_ablation:
        parser.error('Select one ablation at a time.')
    if args.routing_cache_ablation or args.batched_correction_ablation:
        return routing_ablation(args,output,env,roots)
    if args.checkpoint_layers is not None:
        return checkpoint_sweep(args,output,env,roots)
    failures=[]
    for state,parent in enumerate(roots,1):
        candidates=[p for p in (ROOT/'saved_ckpt_runs'/parent).glob('*/*_last_ckpt.pth')
                    if not p.parent.name.startswith('TIMMQAT')]
        assert len(candidates)==1,candidates
        for mode in ('multivariate_gaussian','empirical_with_replacement'):
            if args.training_optimizations and mode!='empirical_with_replacement':continue
            tag=f'state{state}_{mode}'
            env.update(TC_STATE=str(state),MODEL_NAME=candidates[0].parent.name,
                MODEL_DIR=str(candidates[0].parent.parent),OUTPUT_DIR=str(output/tag),
                NONLINEAR_R_CURVE_SAMPLING=mode,
                R_VAL='10e3' if mode=='multivariate_gaussian' else '13.3e3',
                R_MAX='150e3' if mode=='multivariate_gaussian' else '200e3',
                REHEARSAL_ONLY='true',REHEARSAL_ARTIFACT_ROOT=str(output/tag),
                PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'))
            printed=subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],
                cwd=ROOT,env=env,text=True)
            cmd=shlex.split(printed)[3:]+['--batch_size',str(args.batch_size),'--num_workers','0']
            if args.resume and (output/f'{tag}.json').is_file():
                continue
            with (output/f'{tag}.log').open('w') as log:
                completed=subprocess.run([sys.executable,'-u',__file__,'--worker','verify' if args.training_optimizations else 'ft',
                    '--report',str(output/f'{tag}.json'),'--',*cmd],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=not args.training_optimizations)
            if completed.returncode:
                failures.append(tag)
                print('VALIDATION_FAILED',tag,flush=True)
                continue
            print('FT_COMPLETE',tag,flush=True)
        if args.ft_only:
            continue
        # Use the real four-batch empirical rehearsal checkpoint for full-network validation.
        rehearsal=ROOT/'results/tc_multilevel_empirical_rehearsal_20261023_retry'/f'state{state}'
        candidates=[p for p in rehearsal.glob('*/*_last_ckpt.pth') if not p.name.endswith('_full_param_last_ckpt.pth')]
        assert len(candidates)==1,candidates
        env.update(MODEL_NAME=candidates[0].parent.name,MODEL_DIR=str(rehearsal),REHEARSAL_ONLY='false')
        env.pop('PYTHONPATH',None)
        printed=subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','eval'],cwd=ROOT,env=env,text=True)
        cmd=shlex.split(printed)[3:]+['--expanded_w_dir',str(output/'expanded'),
                                     '--hw_val_path',str(output/'evaluation')]
        with (output/f'state{state}_expanded.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--worker','eval',
                '--report',str(output/f'state{state}_expanded.json'),'--',*cmd],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('EVAL_COMPLETE',state,flush=True)
    if args.training_optimizations:
        from validate_tc_empirical_training import accepted,GRADIENT_MAX_ABS,GRADIENT_RELATIVE_L2
        reports={}
        for state in (1,2):
            path=output/f'state{state}_empirical_with_replacement.json'
            if not path.is_file():
                raise SystemExit(f'Missing verification report: {path}')
            record=json.loads(path.read_text())
            reports[f'state{state}']=dict(record,
                all_within_full_model_tolerance=all(accepted(row['metrics']) for row in record['rows']))
        summary=dict(gradient_max_abs_tolerance=GRADIENT_MAX_ABS,
            gradient_relative_l2_tolerance=GRADIENT_RELATIVE_L2,reports=reports,
            all_passed=all(r['all_within_full_model_tolerance'] for r in reports.values()))
        (output/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('FULL_MODEL_TOLERANCES_PASSED',summary['all_passed'],
              output/'training_summary.json',flush=True)
        if not summary['all_passed']:
            raise SystemExit('Full-model tolerance check failed; see training_summary.json')
    if failures:
        raise SystemExit('Validation failed: '+', '.join(failures))


if __name__=='__main__':main()
