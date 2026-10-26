"""Real FT-loop output/gradient and performance comparisons, no saved updates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ATOL,RTOL=1e-5,1e-4
# Full SRRL gradients include nonsmooth feature statistics and many solver
# updates. Retain the stricter elementwise result too; never hide its failures.
GRADIENT_MAX_ABS,GRADIENT_RELATIVE_L2=1e-3,1e-3


def accepted(metrics):
    return metrics['logits']['passed'] and all(
        metrics[key]['max_abs']<=GRADIENT_MAX_ABS
        and metrics[key]['relative_l2']<=GRADIENT_RELATIVE_L2
        for key in ('input_gradient','post_quant_input_gradient','parameter_gradients','auxiliary_gradients')
        if key in metrics)


def compare(actual,expected):
    import torch
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        values=[compare(actual[key],expected[key]) for key in actual]
        return dict(max_abs=max(v['max_abs'] for v in values),
                    max_tolerance_ratio=max(v['max_tolerance_ratio'] for v in values),
                    error_squared=sum(v['error_squared'] for v in values),
                    reference_squared=sum(v['reference_squared'] for v in values),
                    relative_l2=(sum(v['error_squared'] for v in values)/max(sum(v['reference_squared'] for v in values),1e-30))**.5,
                    passed=all(v['passed'] for v in values))
    difference=(actual-expected).abs()
    ratio=difference/(ATOL+RTOL*expected.abs())
    passed=bool(torch.isfinite(ratio).all() and ratio.max()<=1)
    error_squared=difference.double().square().sum().item()
    reference_squared=expected.double().square().sum().item()
    return dict(max_abs=difference.max().item(),max_tolerance_ratio=ratio.max().item(),
        error_squared=error_squared,reference_squared=reference_squared,
        relative_l2=(error_squared/max(reference_squared,1e-30))**.5,passed=passed)


def verify(report,argv):
    import numpy as np
    import torch
    import train_ode_cifar as training
    if os.environ.get('VERIFY_STRICT_FP32') == '1':
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    from training_recovery import _runtime
    sys.argv=['train_ode_cifar.py',*argv]
    args=training.get_args()  # Reuses opt-in bounded dataset/teacher initialization.
    training.get_args=lambda:args
    from trainer_timm import TrainerCiFarTimmStyle

    def run(trainer):
        if os.environ.get('VERIFY_STRICT_FP32') == '1':
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
        trainer._maybe_profile_ode_rhs_checkpointing()
        batch=next(iter(trainer.train_dataloader))
        initial=copy.deepcopy(trainer.model.state_dict())
        auxiliary=copy.deepcopy(trainer._feature_kd_loss.state_dict())
        runtime=copy.deepcopy(_runtime(trainer.model))
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        saved_forward=trainer._student_forward_feature_kd
        capture={}
        collecting=False
        import tc_shared_correction as correction
        original_batched=correction.batched_empirical_correction
        def observed_batched(inputs,draw,nominal,rail,**kwargs):
            result=original_batched(inputs,draw,nominal,rail,**kwargs)
            if collecting and result is not None:
                capture.setdefault('batched_draw_ids',set()).add(id(draw))
                capture['batched_calls']=capture.get('batched_calls',0)+1
            return result
        correction.batched_empirical_correction=observed_batched
        def student_forward(inputs):
            if collecting:
                inputs=inputs.detach().requires_grad_()
                inputs.register_hook(lambda g:capture.update(input_gradient=g.detach().cpu()))
            outputs,features=saved_forward(inputs)
            if collecting:capture['logits']=outputs.detach().cpu()
            return outputs,features
        trainer._student_forward_feature_kd=student_forward
        def step(*unused,**ignored):
            if collecting:
                capture['parameter_gradients']={n:p.grad.detach().cpu().clone()
                    for n,p in trainer.model.named_parameters() if p.grad is not None}
                capture['auxiliary_gradients']={n:p.grad.detach().cpu().clone()
                    for n,p in trainer._feature_kd_loss.named_parameters() if p.grad is not None}
            # No optimizer update: every comparison uses identical model/auxiliary weights.
        trainer.optimizer.step=step

        def restore(method,fused,cached,batched):
            trainer.model.load_state_dict(initial)
            trainer._feature_kd_loss.load_state_dict(auxiliary)
            for name,attributes in copy.deepcopy(runtime).items():
                module=trainer.model.get_submodule(name)
                for key,value in attributes.items():setattr(module,key,value)
            for module in trainer.model.modules():
                if getattr(module,'_tc_curve_package',None) is not None or getattr(module,'_tc_resistance_package',None) is not None:
                    module._tc_conv_method=method
                    module._tc_fused_correction=fused
                    module._tc_cache_level_routing=cached
                    module._tc_batched_correction=batched
                    if hasattr(module,'tc_conv_method'):module.tc_conv_method=method
                    reset=getattr(module,'reset_tc_curves',None)
                    if reset:reset()
                    if hasattr(module,'_tc_samples'):module._tc_samples.clear()
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
            trainer.train_dataloader=[tuple(v.clone() if torch.is_tensor(v) else v for v in batch)]

        rows=[]
        baseline=None
        if os.environ.get('VERIFY_BATCHED_CORRECTION')=='1':
            cases=(('grouped',True,True,False),('grouped',True,True,False),('grouped',True,True,True))
        elif os.environ.get('VERIFY_LEVEL_ROUTING')=='1':
            cases=(('grouped',True,False,False),('grouped',True,False,False),('grouped',True,True,False))
        else:
            cases=(('loop',False,True,False),('loop',False,True,False),('loop',True,True,False),
                   ('grouped',False,True,False),('grouped',True,True,False))
        for method,fused,cached,batched in cases:
            collecting=True;capture={}
            restore(method,fused,cached,batched)
            loss=trainer.train_one_epoch(0)
            indices=[m._tc_curve_samples[n].indices.detach().cpu().tolist()
                for m in trainer.model.PcConvs for n in ('FFconv','FBconv')]
            capture['curve_indices_hash']=hashlib.sha256(json.dumps(indices).encode()).hexdigest()
            if batched:
                expected={id(m._tc_curve_samples[n]) for m in trainer.model.PcConvs
                          for n in ('FFconv','FBconv')}
                assert capture.get('batched_draw_ids',set())==expected
            if baseline is None:baseline=capture
            metrics={key:compare(capture[key],baseline[key])
                     for key in ('logits','input_gradient','parameter_gradients','auxiliary_gradients')}
            assert capture['curve_indices_hash']==baseline['curve_indices_hash']
            flips=(capture['logits'].argmax(1)!=baseline['logits'].argmax(1)).sum().item()
            collecting=False
            times=[];peaks=[]
            for repeat in range(0 if os.environ.get('VERIFY_CAPTURE_ONLY') == '1' else 4):
                restore(method,fused,cached,batched)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                trainer.train_one_epoch(0)
                torch.cuda.synchronize()
                if repeat:
                    times.append(time.perf_counter()-start)
                    peaks.append(torch.cuda.max_memory_allocated())
            row=dict(method=method,correction_fusion=fused,metrics=metrics,
                level_routing_cache=cached,
                batched_correction=batched,
                batched_calls=capture.get('batched_calls',0),
                batched_convolutions=len(capture.get('batched_draw_ids',set())),
                accepted=accepted(metrics),
                prediction_changes=flips,curve_indices_hash=capture['curve_indices_hash'],loss=float(loss),
                median_seconds=statistics.median(times) if times else None,
                seconds=times,peak_allocated_bytes=max(peaks) if peaks else None)
            rows.append(row)
            result=dict(batch_size=args.batch_size,atol=ATOL,rtol=RTOL,
                gradient_max_abs_tolerance=GRADIENT_MAX_ABS,
                gradient_relative_l2_tolerance=GRADIENT_RELATIVE_L2,
                checkpoint_portion=args.checkpoint_ode_rhs_portion,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                max_gradient_norm=trainer.max_norm,
                description='Full real SRRL FT forward/backward; identical weights, data, curves and RNG; optimizer update suppressed.',
                rows=rows)
            Path(report).write_text(json.dumps(result,indent=2)+'\n')
            print('OPTIMIZATION_RESULT',json.dumps(row),flush=True)
        if not all(r['accepted'] for r in rows):
            raise AssertionError(f'End-to-end differences exceeded fixed tolerances: {report}')
    TrainerCiFarTimmStyle.train=run
    training.main()
