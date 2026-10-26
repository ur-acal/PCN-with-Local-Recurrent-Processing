"""Isolated full-recipe top-n parity/benchmark, with a literal original-code baseline."""
import argparse
import ast
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def original_methods(path):
    # Consolidation snapshots include the original correction implementation;
    # use it too, rather than timing old dispatch against today's new helper.
    helper=Path(path).parent/'tc_shared_correction.py'
    if helper.is_file():
        import importlib.util
        spec=importlib.util.spec_from_file_location('tc_shared_correction',helper)
        module=importlib.util.module_from_spec(spec)
        sys.modules['tc_shared_correction']=module
        spec.loader.exec_module(module)
    import ode_pc
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ODEBlockPC')
    for name in ('_tc_curves_for_solve', '_tc_dense_conv'):
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = dict(vars(ode_pc))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        setattr(ode_pc.ODEBlockPC, name, namespace[name])


def coverage(model, n):
    import torch
    rows = []
    with torch.no_grad():
        for layer, block in enumerate(model.PcConvs):
            for name in ('FFconv', 'FBconv'):
                codes = block._values_to_level_idx(getattr(block, name).weight.detach())
                hist = torch.bincount(codes.flatten(), minlength=block.q_hi+1).cpu()
                ranked = torch.argsort(hist[1:], descending=True, stable=True)
                active = ranked[hist[1:][ranked] > 0]
                top = active[:n]+1
                represented = int(hist[top].sum())
                nonzero = int(hist[1:].sum())
                rows.append(dict(layer=layer, convolution=name, histogram=hist.tolist(),
                    top_codes=top.tolist(), individually_covered_weights=represented,
                    shared_weights=nonzero-represented, nonzero_weights=nonzero,
                    zero_weights=int(hist[0]), coverage_percent=100*represented/max(1,nonzero)))
    total = {k:sum(r[k] for r in rows) for k in (
        'individually_covered_weights','shared_weights','nonzero_weights','zero_weights')}
    total['coverage_percent'] = 100*total['individually_covered_weights']/max(1,total['nonzero_weights'])
    return dict(total=total, convolutions=rows,
                note='Matched initial QAT weights before optimizer updates; zero codes excluded from coverage.')


def realized_groups(model, n, require_all=True):
    from tc_curve_groups import CurveGroups
    from tc_empirical import EmpiricalLevels
    rows=[]
    for layer,block in enumerate(model.PcConvs):
        expected=n if block._tc_conv_method=='shared' else 0
        assert getattr(block,'_tc_curve_top_n',0)==expected
        for name,draw in (getattr(block,'_tc_curve_samples',None) or {}).items():
            groups=(draw.nominal.numel() if isinstance(draw,CurveGroups) else
                    draw.indices.numel() if isinstance(draw,EmpiricalLevels) else
                    draw.shape[0] if hasattr(draw,'ndim') and draw.ndim==2 else 1)
            rows.append(dict(layer=layer,convolution=name,groups=groups,draw_type=type(draw).__name__))
    if require_all:assert len(rows)==44,len(rows)
    return rows


def worker(args, argv):
    import numpy as np
    import torch
    if getattr(args,'ordered_groups',False):
        import tc_empirical
        # Diagnostic candidate: same grouped convolution, serial group sum.
        import inspect
        namespace={}
        source=inspect.getsource(tc_empirical.grouped_convolution).replace('.sum(1)','.unbind(1)')
        exec(source,namespace)
        unsummed=namespace['grouped_convolution']
        def ordered(*a,**kw):
            values=unsummed(*a,**kw)
            result=values[0]
            for value in values[1:]:result=result+value
            return result
        tc_empirical.grouped_convolution=ordered
    if args.original and not args.verify:
        original_methods(args.original)
    if not args.verify:
        import benchmark_tc_empirical as bench
        original_write = bench.write
        saved_coverage = {}
        saved_model = []
        from trainer_timm import TrainerCiFarTimmStyle
        original_profile = TrainerCiFarTimmStyle._maybe_profile_ode_rhs_checkpointing
        def profile(trainer):
            result = original_profile(trainer)
            saved_coverage.update(coverage(trainer.model, args.n))
            saved_model.append(trainer.model)
            return result
        TrainerCiFarTimmStyle._maybe_profile_ode_rhs_checkpointing = profile
        bench.write = lambda path, record: original_write(path, dict(record,
            top_n=args.n, weight_coverage=saved_coverage,
            realized_groups=realized_groups(saved_model[0],args.n,require_all=record.get('status')!='OOM')))
        return bench.ft(args.report, argv)
    import train_ode_cifar as training
    from trainer_timm import TrainerCiFarTimmStyle
    from training_recovery import _runtime
    from validate_tc_empirical_training import compare, accepted
    import tc_curve_groups
    sys.argv = ['train_ode_cifar.py', *argv]
    parsed = training.get_args()
    training.get_args = lambda: parsed
    torch.backends.cudnn.allow_tf32 = args.production_precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def run(trainer):
        trainer._maybe_profile_ode_rhs_checkpointing()
        accurate_convolutions = {}
        if getattr(args, 'audit_fp64_convolution', False):
            # Diagnostic only: retain FP32 states/corrections/solver, but remove
            # FP32 convolution and group-output accumulation as error sources.
            # This is not a production precision change or a timing candidate.
            from torch.nn import functional as functional
            def accurate_convolution(original):
                def apply(inputs, weight, bias=None, *a, **kw):
                    return original(inputs.double(), weight.double(),
                        None if bias is None else bias.double(), *a, **kw)
                return apply
            for name in ('conv2d', 'conv_transpose2d'):
                accurate_convolutions[name] = accurate_convolution(getattr(functional, name))
        batch = next(iter(trainer.train_dataloader))
        initial = copy.deepcopy(trainer.model.state_dict())
        auxiliary = copy.deepcopy(trainer._feature_kd_loss.state_dict())
        runtime = copy.deepcopy(_runtime(trainer.model))
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        saved_forward = trainer._student_forward_feature_kd
        captured = {}
        def forward(inputs):
            inputs = inputs.detach().requires_grad_()
            inputs.register_hook(lambda gradient:captured.update(input_gradient=gradient.detach().cpu()))
            out, features = saved_forward(inputs)
            captured['logits'] = out.detach().cpu()
            return out, features
        trainer._student_forward_feature_kd = forward
        def after_input_quantization(module,inputs):
            if inputs[0].requires_grad:
                inputs[0].register_hook(lambda g:captured.update(post_quant_input_gradient=g.detach().cpu().clone()))
        trainer.model.register_forward_pre_hook(after_input_quantization)
        # A tiny state difference can change a nonsmooth SRRL connector gate.
        # Record it explicitly rather than attributing every gradient change to
        # the grouped-convolution backward implementation.
        for name, module in trainer._feature_kd_loss.named_modules():
            if isinstance(module, torch.nn.ReLU):
                def record_relu(module, inputs, name=name):
                    captured.setdefault('srrl_relu_inputs', {})[name] = inputs[0].detach().cpu().clone()
                module.register_forward_pre_hook(record_relu)
        def step(*unused, **ignored):
            captured['parameter_gradients'] = {name:p.grad.detach().cpu().clone()
                for name,p in trainer.model.named_parameters() if p.grad is not None}
            captured['auxiliary_gradients'] = {name:p.grad.detach().cpu().clone()
                for name,p in trainer._feature_kd_loss.named_parameters() if p.grad is not None}
        sys.path.insert(0,str(ROOT/'tests'))
        from test_tc_unified_backend import original_off
        trainer.optimizer.step=step
        original_group=tc_curve_groups.group_convolution
        module_names = {id(module): name for name, module in trainer.model.named_modules()}
        variants = ('reference','off_repeat','off_grouped','on_scalar_grouped','on_batched_grouped','on_scalar_serial')
        base = None
        reference_draws = None
        rows = []
        for variant in variants:
            trainer.model.load_state_dict(initial)
            trainer._feature_kd_loss.load_state_dict(auxiliary)
            for name, attributes in copy.deepcopy(runtime).items():
                for key,value in attributes.items():setattr(trainer.model.get_submodule(name),key,value)
            for module in trainer.model.modules():
                if hasattr(module,'reset_tc_curves'):module.reset_tc_curves()
                module._tc_fused_correction = variant.startswith('on_')
                module._tc_batched_correction = variant == 'on_batched_grouped'
                module._tc_grouped_convolution = variant.endswith('_grouped')
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            sampled = {}
            backend = original_off if variant == 'reference' else original_group
            def observed(owner,module,source,draw):
                key = module_names[id(module)]
                if key not in sampled:
                    from tc_nonidealities import TCSharedCurve
                    if isinstance(draw,TCSharedCurve):
                        tensors=draw.prepare(owner._tc_curve_package.v_grid,source,owner._tc_curve_package.floor_ohms)
                    else:
                        tensors=draw.corrections.prepare_batched(source,draw.nominal)
                        if draw.mask is not None:tensors=(*tensors,draw.mask)
                    digest=hashlib.sha256()
                    for tensor in tensors:
                        if tensor is not None:digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                    sampled[key]=digest.hexdigest()
                if accurate_convolutions:
                    from unittest.mock import patch
                    with ExitStack() as stack:
                        for name, function in accurate_convolutions.items():
                            stack.enter_context(patch.object(functional, name, function))
                        return backend(owner,module,source,draw).to(source.dtype)
                return backend(owner,module,source,draw)
            tc_curve_groups.group_convolution = observed
            captured.clear()
            trainer.train_dataloader = [tuple(v.clone() if torch.is_tensor(v) else v for v in batch)]
            trainer.train_one_epoch(0)
            if base is None:
                base = copy.deepcopy(captured)
                reference_draws = sampled
            else:
                metrics = {key:compare(captured[key],base[key]) for key in base}
                rows.append(dict(variant=variant, metrics=metrics,
                    matched_curve_draws=sampled==reference_draws,
                    accepted=accepted(metrics) and sampled==reference_draws,
                    srrl_relu_gate_flips={name:int(((value>0)!=(base['srrl_relu_inputs'][name]>0)).sum())
                        for name,value in captured.get('srrl_relu_inputs',{}).items()},
                    prediction_flips=int((captured['logits'].argmax(1)!=base['logits'].argmax(1)).sum())))
        passed=all(row['accepted'] for row in rows)
        record = dict(top_n=args.n, batch_size=parsed.batch_size, rows=rows, weight_coverage=coverage(trainer.model,args.n),
                      realized_groups=realized_groups(trainer.model,args.n),
                      atol=1e-5,rtol=1e-4, passed=passed,
                      reference='original PyTorch correction + serial convolution OFF expression',
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      audit_fp64_convolution=getattr(args, 'audit_fp64_convolution', False),
                      audit_fp64_scope='convolution_and_group_sum' if accurate_convolutions else None,
                      gradient_max_abs_tolerance=1e-3,gradient_relative_l2_tolerance=1e-3)
        Path(args.report).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if not passed:raise RuntimeError('Full-model numerical tolerance exceeded; see report.')
    TrainerCiFarTimmStyle.train = run
    training.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--production-precision',action='store_true',
        help='Validate with production CuDNN TF32 permission, instead of strict FP32.')
    parser.add_argument('--ordered-groups',action='store_true',help='Diagnostic: serial reduction after grouped convolution.')
    parser.add_argument('--audit-fp64-convolution',action='store_true',
        help='Diagnostic only: accumulate convolutions/group sums in FP64, retaining FP32 states.')
    parser.add_argument('--original')
    parser.add_argument('--n',type=int,default=0)
    parser.add_argument('--report')
    parser.add_argument('--output')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--states',type=int,nargs='+',choices=(1,2),default=[1,2])
    parser.add_argument('--ns',type=int,nargs='+',choices=range(16),default=[0,1,2,3,15])
    parser.add_argument('--modes',nargs='+',choices=('multivariate_gaussian','empirical_with_replacement'),
                        default=['multivariate_gaussian','empirical_with_replacement'])
    parser.add_argument('argv',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:return worker(args,args.argv[1:])
    # Every invocation is a fresh experiment: never certify stale reports from
    # another batch size, precision mode or source revision as current results.
    output = Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    records = {}
    env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
        TC_DRY_RUN='true',TASK='cifar100',IMG_TYPE='rgb',CKPT='last',
        NORMALIZE_STUDENT_INPUT='false',CENTER_STUDENT_INPUT='false',
        FT_TIMM_AUG_LEVEL='rgb_mid',ACTIVATION_CORNER_MODE='random_per_forward',
        ACTIVATION_RANDOM_CURVE_SHARING='per_spin',
        TEACHER_CKPT='./checkpoint/efficientnet_v2_l_cifar100_rgb_OldNoTimm_MatchDistill_NoNorm.pth',
        TEACHER_ARCH='efficientnet_v2_l',TEACHER_ARCH_SOURCE='torchvision',
        REUSE_ACCEPTED_STEP_TRAINING='true',CHECKPOINT_ODE_RHS_TRAINING='true',
        CHECKPOINT_ODE_RHS_PORTION='1', FUSE_MEASURED_ACTIVATION='true',
        TC_CONV_METHOD='shared',REHEARSAL_ONLY='true',REHEARSAL_BATCHES='1',
        BENCHMARK_MEASURED_STEPS='5',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONPATH=str(ROOT/'diagnostic_scripts/rehearsal_hooks'),
        TC_EMPIRICAL_CURVE_BANK=str(ROOT/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'))
    roots = ('tc_rgb_cifar100_state1_pcn_resnet_depth_study_student_teacher_no_norm',
        'tc_rgb_cifar100_state2_pcn_22L7l6l6_pipeline_student_teacher_no_norm_random_per_forward_per_spin_relu_rgb_mid')
    for state in args.states:
        candidates = [p for p in (ROOT/'saved_ckpt_runs'/roots[state-1]).glob('*/*_last_ckpt.pth')
                      if not p.parent.name.startswith('TIMMQAT')]
        assert len(candidates)==1,candidates
        for mode in args.modes:
            variants = [('original',0)] if mode=='multivariate_gaussian' and args.original else []
            variants += [('current',n) for n in args.ns]
            if mode=='empirical_with_replacement' and not args.verify:
                variants.append(('full_levels',15))
            for version,n in variants:
                tag=f's{state}_{mode}_{version}_n{n}'
                report=output/f'{tag}.json'
                if report.exists():
                    raise FileExistsError(f'Duplicate configuration/report: {report}')
                env.update(TC_STATE=str(state),MODEL_NAME=candidates[0].parent.name,
                    MODEL_DIR=str(candidates[0].parent.parent),OUTPUT_DIR=str(output/tag),
                    REHEARSAL_ARTIFACT_ROOT=str(output/tag), TC_CURVE_TOP_N=str(0 if version=='full_levels' else n),
                    TC_CONV_METHOD='grouped' if version=='full_levels' else 'shared',
                    NONLINEAR_R_CURVE_SAMPLING=mode,
                    R_VAL='10e3' if mode=='multivariate_gaussian' else '13.3e3',
                    R_MAX='150e3' if mode=='multivariate_gaussian' else '200e3')
                printed=subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],cwd=ROOT,env=env,text=True)
                cmd=shlex.split(printed)[3:]+['--batch_size',str(args.batch_size),'--num_workers','0']
                options=['--original',args.original] if version=='original' else []
                if args.verify:options+=['--verify']
                if args.production_precision:options+=['--production-precision']
                if args.ordered_groups:options+=['--ordered-groups']
                if args.audit_fp64_convolution:options+=['--audit-fp64-convolution']
                print('START',tag,flush=True)
                with (output/f'{tag}.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',__file__,'--worker','--n',str(n),
                        '--report',str(report),*options,'--',*cmd],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                print('FINISH',tag,flush=True)
                records[tag] = json.loads(report.read_text())
    (output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    if args.verify:
        if not all(record.get('passed',False) for record in records.values()):
            raise RuntimeError('One or more validation records failed; see summary.json.')
    elif not all(record.get('status')=='ok' for record in records.values()):
        raise RuntimeError('One or more benchmarks did not complete; see summary.json.')


if __name__=='__main__':main()
