"""Real pinned-toggle FT parity and sequential fresh-process timing.

Capture the actual FT command from the pinned scheduler/worker (no submission),
then initialize the production trainer, source checkpoint and SRRL teacher.
No checkpoint save or full training run is performed. Only diagnostic outputs,
loader workers (zero; loading excluded from timing) and the fusion flag differ.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ATOL, RTOL = 1e-5, 1e-4
GRAD_ABS, GRAD_REL = 1e-3, 1e-3


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n')


def parity_passed(results, precision):
    rows = results[precision]['rows']
    return len(rows) == 3 and all(row['accepted'] for row in rows)


def require_parity(results):
    failed = [precision for precision in ('strict', 'production')
              if not parity_passed(results, precision)]
    if failed:
        raise SystemExit('Parity criteria failed: '+', '.join(failed)+
                         '; reports retained; no tolerances relaxed.')


def report(output, results):
    lines=['# Toggle nonlinear-R input-correction fusion: real FT validation', '',
        'PCN 16L96C, CiFAIR-100, batch 128, real SRRL EfficientNet-V2-L teacher. '
        'Verified job-2090063 pretrained source; fixed 10 ns, VDD=0.5, inverse-q=5, '
        'full-range couplers, 0906 per-spin ReLU, corner-coupled sampling and all pinned FT defects.', '',
        'Parity suppresses optimizer updates and restores model/auxiliary weights, data, global '
        'and module-local RNGs. Actual nonlinear-R/ReLU/pooling/spin assignments are hashed and compared. '
        'One full-size training batch per comparison; not an accuracy evaluation.', '',
        'Criteria fixed before running: logits atol=1e-5 / rtol=1e-4; each gradient category '
        'maximum absolute error <=1e-3 AND relative L2 error <=0.1%. '
        'Strict elementwise gradient atol=1e-5 / rtol=1e-4 is additionally retained.', '']
    names={'logits':'Final logits (raw digital accumulator units)',
           'loss_logits':'Loss-scaled logits','input_gradients':'Raw input gradients',
           'quantized_input_gradients':'Post-quantization network-input gradients',
           'model_gradients':'All model-parameter gradients','auxiliary_gradients':'SRRL auxiliary gradients'}
    for precision in ('strict','production'):
        d=results[precision];repeat,on=d['rows'][1:]
        lines += ['## '+('Deterministic FP32' if precision=='strict' else 'Production precision'), '',
            f"cuDNN TF32={d['cudnn_allow_tf32']}; matmul TF32={d['matmul_allow_tf32']}; "
            f"cuDNN deterministic={d['cudnn_deterministic']}; checkpointed layers={d['checkpoint_selected_layers']}.", '',
            '| Quantity | OFF/OFF max abs | OFF/OFF relative L2 | OFF/ON max abs | OFF/ON relative L2 | OFF/ON strict elementwise |',
            '|---|---:|---:|---:|---:|---|']
        for key,name in names.items():
            a,b=repeat['metrics'][key],on['metrics'][key]
            lines.append(f"| {name} | {a['max_abs']:.8g} | {100*a['relative_l2']:.8g}% | "
                         f"{b['max_abs']:.8g} | {100*b['relative_l2']:.8g}% | {b['strict_elementwise']} |")
        finite = all(metric['finite'] for row in d['rows'] for metric in row['metrics'].values())
        structure_matches = all(
            metric.get('keys_match', False) and metric.get('none_status_match', False)
            for row in d['rows'] for key,metric in row['metrics'].items()
            if key in ('model_gradients','auxiliary_gradients'))
        lines += ['',f"Overall accepted: OFF/OFF={repeat['accepted']}, OFF/ON={on['accepted']}. "
            f"Prediction changes: {repeat['prediction_changes']}/128 and {on['prediction_changes']}/128. "
            f"Hardware assignments match={repeat['assignments_match'] and on['assignments_match']}; "
            f"end RNG/runtime match={repeat['end_rng_match'] and on['end_rng_match']}.",
            f"All compared tensors finite={finite}; gradient key/None-status matches={structure_matches}. "
            'Raw and post-quantization input gradients are reported separately.', '']
    lines += ['## Training benchmark', '',
        'OFF and ON ran sequentially in fresh processes on an idle RTX 4090. '
        'Production precision, two warm-ups and five measured steps; real teacher/student '
        'forward, backward and optimizer updates. Cached real augmented batch: data loading '
        'excluded, loader workers=0. Startup-only teacher accuracy sweep omitted. '
        'Memory includes warm-up peaks. No profiler, checkpoint saves or full training run.', '',
        '| Fusion | Median step (ms) | Peak allocated (GiB) | Peak reserved (GiB) |',
        '|---|---:|---:|---:|']
    for key in ('benchmark_off','benchmark_on'):
        d=results[key]
        if d['status']=='ok':
            lines.append(f"| {d['fusion']} | {1000*d['median_seconds']:.3f} | "
                f"{d['peak_allocated_bytes']/2**30:.4f} | {d['peak_reserved_bytes']/2**30:.4f} |")
        else:lines.append(f"| {d['fusion']} | OOM | — | — |")
    if 'timing' in results:
        t=results['timing']
        lines += ['',f"Speedup={t['speedup']:.4f}x; time reduction={t['time_reduction_percent']:.3f}% "
            '(negative means slower). This short measurement is not evidence of a general speedup.']
    strict = 'passes' if parity_passed(results, 'strict') else 'fails'
    production = 'passes' if parity_passed(results, 'production') else 'fails'
    repeat_note = ('Production OFF/OFF also fails; do not attribute all observed error to fusion.'
                   if not results['production']['rows'][1]['accepted'] else
                   'Production OFF/OFF meets the fixed criteria.')
    lines += ['', '## Conclusion', '',
        f'Deterministic FP32 {strict}. Production-precision parity {production} the fixed criteria. '
        f'{repeat_note} No thresholds were relaxed. Fusion remains opt-in/default OFF.', '',
        'Full outputs/gradients are saved in the parity directories as *_outputs_gradients.pt. '
        'Each *_command.json records the exact command and recipe environment. '
        'Recorded initial-weight, auxiliary-weight and batch hashes are available in summary.json.', '',
        'Reproduce in a new output directory:', '', '```bash',
        'OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 /home/rongzeng/anaconda3/envs/scanbase/bin/python -u \\',
        '  diagnostic_scripts/benchmark_toggle_input_correction.py \\',
        '  --output results/toggle_input_correction_fusion/full_model_validation_repeat', '```', '']
    (output/'summary.md').write_text('\n'.join(lines))


def digest(value):
    import torch
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, torch.Generator):
            visit(v.get_state())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for item in v: visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def hardware(model):
    result = {}
    for name, module in model.named_modules():
        values = {}
        for attr in ('_nonlinear_R_training_curve_indices', '_sampled_curve_indices',
                     '_curve_assignments', '_spin_factor_y', '_spin_factor_z'):
            if hasattr(module, attr): values[attr] = getattr(module, attr)
        if values: result[name] = values
    result['common_corner'] = getattr(model, '_last_ft_corner_coupled_corner', None)
    return result


def compare(actual, expected):
    import torch
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('Gradient keys differ')
        none_match = all((actual[k] is None) == (expected[k] is None) for k in actual)
        if not none_match: raise AssertionError('Gradient None status differs')
        rows = {k: compare(actual[k], expected[k]) for k in actual if actual[k] is not None}
        if not rows: raise AssertionError('No gradients in category')
        e2 = sum(v['error_squared'] for v in rows.values())
        r2 = sum(v['reference_squared'] for v in rows.values())
        worst = max(rows, key=lambda k: rows[k]['max_abs'])
        return dict(max_abs=rows[worst]['max_abs'], relative_l2=(e2/max(r2, 1e-30))**.5,
            error_squared=e2, reference_squared=r2, finite=all(v['finite'] for v in rows.values()),
            strict_elementwise=all(v['strict_elementwise'] for v in rows.values()),
            max_tolerance_ratio=max(v['max_tolerance_ratio'] for v in rows.values()),
            worst_key=worst, keys_match=True, none_status_match=True,
            tensors=len(rows), none_count=len(actual)-len(rows))
    a, e = actual.double(), expected.double()
    delta = a-e
    e2, r2 = delta.square().sum().item(), e.square().sum().item()
    ratio = delta.abs()/(ATOL+RTOL*e.abs())
    return dict(max_abs=delta.abs().max().item(), relative_l2=(e2/max(r2,1e-30))**.5,
        error_squared=e2, reference_squared=r2,
        finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
        strict_elementwise=bool((ratio <= 1).all()), max_tolerance_ratio=ratio.max().item())


def capture(output):
    """Use real scheduler exports and the real worker's FT command builder."""
    text = (ROOT/'docs/current_pcn_reference.md').read_text()
    section = text.split('## Main-model SLURM training:', 1)[1].split('### Provisional', 1)[0]
    exports = 'export TOGGLE_MODE='+section.split('  export TOGGLE_MODE=', 1)[1].split('\n\n  source ', 1)[0]
    # A fresh login-like recipe environment, retaining paths/device visibility.
    keep = ('PATH','HOME','USER','LOGNAME','SHELL','CUDA_VISIBLE_DEVICES','CONDA_PREFIX',
            'LD_LIBRARY_PATH','UV_CACHE_DIR','TMPDIR')
    env = {k:os.environ[k] for k in keep if k in os.environ}
    env.update(OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(output/'matplotlib'), XDG_CACHE_HOME=str(output/'cache'))
    printed = subprocess.run(['bash','-c',exports+'\nTC_DRY_RUN=true\nsource ./launch_scripts/slurm_search_config.sh'],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    (output/'scheduler_capture.log').write_text(printed.stdout+printed.stderr)
    lines = [l for l in printed.stdout.splitlines() if l.startswith('sbatch --parsable')]
    if len(lines) != 1: raise AssertionError('Expected exactly one pinned model')
    decoded = subprocess.check_output(['bash','-c',
        'sbatch() { for arg; do [[ "$arg" != --export=* ]] || printf "%s" "${arg#--export=}"; done; };\n'+lines[0]],
        cwd=ROOT, env=env, text=True)
    for value in decoded.split(','):
        if '=' in value:
            key, val = value.split('=', 1); env[key] = val
    env.update(FT_OUTPUT_SAVE_PATH=str(output/'captured_output'),
        VALIDATION_PYTHON=sys.executable, VALIDATION_SCRIPT=str(Path(__file__).resolve()),
        CAPTURE_JSON=str(output/'ft_command.json'))
    worker = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text().split('# PHASE 1:')[0]
    worker = worker.replace('source activate base', ':').replace('conda activate scanbase', ':')
    worker = worker.replace('/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/', str(output/'capture_logs')+'/')
    suffix = '''
IFS=$'\\t' read -r comb_tag params inp_str out_str pool_str <<< "${COMBS[0]}"
name="$("$VALIDATION_PYTHON" launch_scripts/find_tc_pretrain.py --root "$PRETRAIN_SAVE_PATH" \\
  --inp "$inp_str" --out "$out_str" --pool "$pool_str" --stride "${STRIDE[*]}" \\
  --kernel "${KERNEL[*]}" --task "$TASK" --img_type "$IMG_TYPE" --block "$TRAIN_ODE_BLOCK" \\
  --pcn "$PCN" --bits "$INPUT_QUANT_BITS" --center "$CENTER_STUDENT_INPUT" --selection best)" || exit
python() { "$VALIDATION_PYTHON" "$VALIDATION_SCRIPT" --capture "$@"; }
finetune_one_combo_model "$comb_tag" "$name" "$FT_ODE_BLOCK"
'''
    r = subprocess.run(['bash','-c',worker+suffix], cwd=ROOT, env=env,
        text=True, capture_output=True, check=True)
    (output/'worker_capture.log').write_text(r.stdout+r.stderr)
    write(output/'recipe_environment.json', env)
    return env


def worker(options):
    import numpy as np
    import torch
    import train_ode_cifar as training
    from trainer_timm import TrainerCiFarTimmStyle
    from training_recovery import _runtime
    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=False)
    argv = json.loads(Path(options.ft_args).read_text())
    argv += ['--output_save_path', str(out/'initialization'), '--num_workers', '0']
    sys.argv = ['train_ode_cifar.py', *argv]
    args = training.get_args()
    args.fuse_toggle_nonlinear_R_input = options.fused == 'on'
    training.get_args = lambda: args
    source = Path(args.save_path)/args.model_name/(args.model_name+'_'+args.ckpt+'_ckpt.pth')
    if not source.is_file(): raise FileNotFoundError(source)
    metadata = dict(batch_size=args.batch_size, precision=options.precision,
        source_checkpoint=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_loading_in_timing=False, loader_workers=0,
        description='Cached real augmented training batch; real student/teacher/SRRL forward, backward; '
                    'optimizer updates only in benchmark. No model checkpoint saves.')
    write(out/'resolved_args.json', vars(args))
    if options.precision == 'strict':
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    # Teacher accuracy evaluation is startup-only, not a training computation.
    # Keep real teacher initialization/feature extraction and every timed forward.
    training.evaluate_teacher = lambda *unused: print('DIAGNOSTIC: startup teacher accuracy sweep omitted', flush=True)

    def run(trainer):
        trainer._maybe_profile_ode_rhs_checkpointing()
        if options.precision == 'strict':
            torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        batch = next(iter(trainer.train_dataloader))
        assert batch[0].shape[0] == 128 == args.batch_size
        trainer.train_dataloader = [batch]
        metadata.update(actual_batch_shape=list(batch[0].shape),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            checkpoint_selected_layers=getattr(trainer.model,'checkpoint_ode_rhs_selected_layers',None),
            checkpoint_portion=args.checkpoint_ode_rhs_portion,
            batch_hash=digest(batch), initial_weights_hash=digest(trainer.model.state_dict()),
            initial_auxiliary_hash=digest(trainer._feature_kd_loss.state_dict()),
            gpu=torch.cuda.get_device_name())
        blocks = [m for m in trainer.model.modules() if hasattr(m,'_nonlinear_R_training_pkg')]
        assert len(blocks) == 16, len(blocks)
        assert args.ft_corner_coupled_sampling and args.activation_corner_mode == 'random_per_forward'
        assert args.activation_random_curve_sharing == 'per_spin'
        def fused(value):
            for b in blocks: b._fuse_toggle_nonlinear_R_input = value
        if options.worker == 'benchmark':
            fused(options.fused == 'on')
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            rows=[]
            try:
                for step in range(7):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    loss=trainer.train_one_epoch(0)
                    torch.cuda.synchronize()
                    rows.append(dict(step=step,warmup=step<2,seconds=time.perf_counter()-start,loss=float(loss)))
                    if options.fused == 'on':
                        assert all(len(b._toggle_nonlinear_R_prepared)==2 for b in blocks)
                    print('BENCHMARK_STEP',rows[-1],flush=True)
                result=dict(metadata,status='ok',fusion=options.fused,steps=rows,
                    median_seconds=statistics.median(r['seconds'] for r in rows[2:]),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),warmups=2,measurements=5)
            except torch.OutOfMemoryError as error:
                result=dict(metadata,status='OOM',fusion=options.fused,steps=rows,error=str(error),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write(out/'result.json',result)
            return
        initial=copy.deepcopy(trainer.model.state_dict())
        auxiliary=copy.deepcopy(trainer._feature_kd_loss.state_dict())
        runtime=copy.deepcopy(_runtime(trainer.model))
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        original_forward=trainer._student_forward_feature_kd
        captured={}
        def forward(inputs):
            from pc_model import logits_for_loss
            leaf=inputs.detach().requires_grad_()
            leaf.register_hook(lambda g:captured.update(input_gradients=g.detach().cpu().clone()))
            logits, features=original_forward(leaf)
            captured['logits']=logits.detach().cpu().clone()
            captured['loss_logits']=logits_for_loss(logits,trainer.model).detach().cpu().clone()
            return logits,features
        trainer._student_forward_feature_kd=forward
        def after_input_quantization(module, inputs):
            # Observe the actual network-input gradient, without bypassing or
            # replacing the production quantizer (whose round gradient is zero).
            assert inputs[0].requires_grad
            inputs[0].register_hook(lambda g:captured.update(
                quantized_input_gradients=g.detach().cpu().clone()))
        trainer.model.register_forward_pre_hook(after_input_quantization)
        def no_update(*unused,**ignored):
            for category,module in (('model_gradients',trainer.model),('auxiliary_gradients',trainer._feature_kd_loss)):
                captured[category]={n:None if p.grad is None else p.grad.detach().cpu().clone()
                                    for n,p in module.named_parameters()}
        trainer.optimizer.step=no_update
        rows=[];baseline=None;assignment_baseline=None;end_rng_baseline=None
        for label, enabled in (('OFF_reference',False),('OFF_repeat',False),('ON',True)):
            trainer.model.load_state_dict(initial)
            trainer._feature_kd_loss.load_state_dict(auxiliary)
            for name,attributes in copy.deepcopy(runtime).items():
                module=trainer.model.get_submodule(name)
                for key,value in attributes.items(): setattr(module,key,value)
            sampler=getattr(trainer.model,'_ft_corner_coupled_sampler',None)
            if sampler is not None: sampler.clear()
            for b in blocks: b._toggle_nonlinear_R_prepared.clear()
            fused(enabled)
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            captured={}
            start_rng=digest(_runtime(trainer.model))
            loss=trainer.train_one_epoch(0)
            assignments=digest(hardware(trainer.model))
            end_rng=digest((torch.get_rng_state(),torch.cuda.get_rng_state_all(),_runtime(trainer.model)))
            if enabled: assert all(len(b._toggle_nonlinear_R_prepared)==2 for b in blocks)
            if baseline is None:
                baseline=captured;assignment_baseline=assignments;end_rng_baseline=end_rng
            metrics={key:compare(captured[key],baseline[key]) for key in (
                'logits','loss_logits','input_gradients','quantized_input_gradients',
                'model_gradients','auxiliary_gradients')}
            predictions=int((captured['logits'].argmax(1)!=baseline['logits'].argmax(1)).sum())
            accepted=(metrics['logits']['strict_elementwise'] and
                all(v['finite'] for v in metrics.values()) and all(
                    metrics[k]['max_abs']<=GRAD_ABS and metrics[k]['relative_l2']<=GRAD_REL
                    for k in ('input_gradients','quantized_input_gradients','model_gradients','auxiliary_gradients')) and
                assignments==assignment_baseline and end_rng==end_rng_baseline)
            row=dict(label=label,fusion=enabled,metrics=metrics,prediction_changes=predictions,
                hardware_assignments_sha256=assignments,assignments_match=assignments==assignment_baseline,
                start_runtime_sha256=start_rng,end_runtime_sha256=end_rng,
                end_rng_match=end_rng==end_rng_baseline,loss=float(loss),accepted=accepted)
            rows.append(row)
            torch.save(captured,out/(label+'_outputs_gradients.pt'))
            write(out/'result.json',dict(metadata,rows=rows,atol=ATOL,rtol=RTOL,
                gradient_max_abs_tolerance=GRAD_ABS,gradient_relative_l2_tolerance=GRAD_REL))
            print('PARITY_RESULT',json.dumps(row),flush=True)
    TrainerCiFarTimmStyle.train=run
    training.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--capture':
        argv=sys.argv[2:]
        if argv[0]=='-u':argv=argv[1:]
        assert argv[0]=='train_ode_cifar.py',argv
        write(os.environ['CAPTURE_JSON'],argv[1:]);print('FT command captured; no training launched')
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--worker',choices=('parity','benchmark'))
    p.add_argument('--precision',choices=('strict','production'),default='production')
    p.add_argument('--fused',choices=('on','off'),default='off')
    p.add_argument('--ft-args')
    p.add_argument('--report-only',action='store_true')
    o=p.parse_args()
    if o.worker:return worker(o)
    if o.report_only:
        out=Path(o.output).resolve()
        results=json.loads((out/'summary.json').read_text())
        report(out,results)
        require_parity(results)
        return
    out=Path(o.output).resolve();out.mkdir(parents=True,exist_ok=False)
    env=capture(out)
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    for name,kind,precision,fusion in (('strict','parity','strict','off'),
            ('production','parity','production','off'),('benchmark_off','benchmark','production','off'),
            ('benchmark_on','benchmark','production','on')):
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
        if busy:raise RuntimeError(f'GPU not idle; no jobs changed: {busy}')
        current=dict(env)
        if precision=='strict':current['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',kind,'--precision',precision,
             '--fused',fusion,'--ft-args',str(out/'ft_command.json'),'--output',str(out/name)]
        write(out/(name+'_command.json'),dict(command=cmd,environment=current))
        print('START',name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            r=subprocess.run(cmd,cwd=ROOT,env=current,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{name} failed ({r.returncode}); inspect {out/(name+".log")}')
        print('FINISHED',name,flush=True)
    results={name:json.loads((out/name/'result.json').read_text()) for name in (
        'strict','production','benchmark_off','benchmark_on')}
    off,on=results['benchmark_off'],results['benchmark_on']
    for key in ('source_sha256','initial_weights_hash','initial_auxiliary_hash','batch_hash'):
        assert off[key]==on[key],key
    if off['status']=='ok' and on['status']=='ok':
        results['timing']=dict(speedup=off['median_seconds']/on['median_seconds'],
            time_reduction_percent=100*(1-on['median_seconds']/off['median_seconds']))
    write(out/'summary.json',results)
    report(out,results)
    require_parity(results)
    print('COMPLETE',out/'summary.json',flush=True)


def s2_build_model(o, dynamics, scene, representation):
    """Six PC blocks: expansion + one repeat per stage; pool before expansion."""
    import torch
    from pc_model import PCNetNoBatchNorm
    from pc_conv import PCConvReLU6
    from ode_pc import (S2DTCFFFB, S2DTCPulseFFFB, S2DTCWrapper2State,
                        S2DTCQATWrapper2State, make_ode_block, wrap_ode_block)
    from inference_utils import replace_transpose_conv
    from validation import Validator
    torch.manual_seed(1907)
    model = PCNetNoBatchNorm(inp_channels=[4, 24, 24, 48, 48, 96],
        out_channels=[24, 24, 48, 48, 96, 96], max_pool=[0, 1, 0, 1, 0, 0],
        num_classes=100, pc_conv_layer=PCConvReLU6, first_bn=False, avg_pooling=True,
        dropout=0, bypass=False, tie_weights=False, linear_bias=False,
        measured_activation_scope='all', final_head_type='old_ideal')
    # Match the fixed-timing recipe's weight compensation relative to 1.75.
    with torch.no_grad():
        for block in model.PcConvs:
            for conv in (block.FFconv, block.FBconv):
                conv.weight.mul_(7 / 8)
    initial_hash = digest(model.state_dict())
    replace_transpose_conv(model)
    model.device = torch.device('cuda')
    model.cuda()
    cls = S2DTCFFFB if dynamics == 'average' else S2DTCPulseFFFB
    make_ode_block(model, ode_block=cls, method='rk2', t_end=1.75, n_steps=5,
        tol=1e-6, toggle_n_cycles=5, toggle_timing_mode='fixed', toggle_y_time=1e-8,
        z_over_y_time=1, toggle_timing_R=5e4, toggle_timing_C=5e-13,
        y_init_with='x', z_init_with='0', s2_dtc_substeps=1,
        s2_dtc_ordering_pairs=o.ordering_pairs,
        s2_dtc_reset_z_per_cycle=getattr(o, 'reset_z_per_cycle', False),
        checkpoint_ode_rhs_training=o.checkpoint)
    measured = scene in ('measured', 'nonlinear')
    nonlinear = scene == 'nonlinear'
    training = o.s2_worker == 'benchmark' and o.train_step
    wrapper = S2DTCQATWrapper2State if training else S2DTCWrapper2State
    model, wrappers = wrap_ode_block(model, ode_wrapper=wrapper,
        R=5e4, C=5e-13, v_dd=.5, one_over_q=5, w_bits=5,
        weight_quant_factor_bits=1, thermal_noise=False, enob=None,
        enable_measured_activation=measured,
        activation_curve_path=str(ROOT/'hardware_data/mc_45_corners/0906_RELU_Voltage/fs_25_2.csv'),
        activation_corner='MC18', activation_interpolation='piecewise_linear',
        activation_curve_sharing='per_spin', activation_curve_seed=4096,
        activation_normalize_positive_endpoint=False, fuse_measured_activation=False,
        nonlinear_R=nonlinear,
        nonlinear_R_table=str(ROOT/'hardware_data/mc_45_corners/coupler_full_range/fs_25_2.csv'),
        nonlinear_R_train_mode='exact_curve' if nonlinear and representation == 'dense' else 'none',
        nonlinear_R_curve_sharing='shared' if representation == 'dense' else 'per_coupler',
        nonlinear_R_mc_quantity='conductance', nonlinear_R_curve_sampling='empirical_with_replacement',
        nonlinear_R_curve_seed=4096)
    model.eval()
    if scene == 'ideal':
        from types import MethodType
        for block in model.PcConvs:
            block.project_state = MethodType(lambda self, state: state, block)
            block.act_fn = torch.nn.ReLU()
    inputs = torch.load(o.batch, map_location='cpu', weights_only=False)[0]
    assert inputs.shape[1:] == (4, 16, 16), inputs.shape
    validator = None
    if representation == 'expanded':
        assert not training, 'Expanded training is outside the supported production path'
        directory = Path(o.output)/'cache'/dynamics/scene
        directory.mkdir(parents=True, exist_ok=True)
        validator = Validator(model, str(directory), 'cuda',
            [(inputs[:2], torch.zeros(2, dtype=torch.long))], str(directory), wrapper=wrappers)
    return model, inputs, validator, initial_hash


def s2_assignments(model):
    """Hash actual sampled identities, not just their RNG seeds."""
    assignments = {}
    for name, module in model.named_modules():
        for attr in ('_nonlinear_R_training_curve_indices', '_curve_assignments',
                     'nonlinear_R_curve_assignment', '_sampled_curve_indices', 'active_corner'):
            if hasattr(module, attr):
                assignments[name + '.' + attr] = getattr(module, attr)
    return digest(assignments)


def s2_record(model, x, dynamics):
    import torch
    from types import MethodType
    traces, hooks, originals = {}, [], []
    for i, block in enumerate(model.PcConvs):
        original = block._s2_dtc_interval
        originals.append((block, original))
        counter = [0]
        def interval(self, *a, _original=original, _counter=counter, _i=i, **kw):
            state = _original(*a, **kw)
            _counter[0] += 1
            slices = 15 if dynamics == 'pulse' else 1
            if _counter[0] % slices == 0:
                frame = _counter[0] // slices
                for label, value in zip(('y', 'z'), state):
                    traces[f'L{_i+1}/frame{frame}/{label}'] = value.detach().cpu().clone()
            return state
        block._s2_dtc_interval = MethodType(interval, block)
        def capture(module, inputs, output, index=i):
            traces[f'L{index+1}/output'] = output.detach().cpu().clone()
        hooks.append(block.register_forward_hook(capture))
    try:
        with torch.no_grad():
            traces['logits'] = model(x).detach().cpu()
        assert all(torch.isfinite(t).all() for t in traces.values()), 'Nonfinite trajectory'
        return traces
    finally:
        for block, original in originals:
            block._s2_dtc_interval = original
        for hook in hooks:
            hook.remove()


def s2_set_solver(model, method, substeps, tol):
    for block in model.PcConvs:
        block.method, block.s2_dtc_substeps = method, substeps
        block.option_aca.update(method=method, rtol=tol, atol=tol)


def s2_error(actual, reference):
    import torch
    result = {}
    for key in reference:
        a, b = actual[key].double(), reference[key].double()
        diff = a - b
        result[key] = dict(relative_l2=float(diff.norm()/b.norm().clamp_min(1e-30)),
                           absolute_rms=float(diff.square().mean().sqrt()),
                           max_abs=float(diff.abs().max()))
    a = torch.cat([v.flatten().double() for k, v in actual.items() if '/frame' in k])
    b = torch.cat([v.flatten().double() for k, v in reference.items() if '/frame' in k])
    result['all_frames'] = dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                               absolute_rms=float((a-b).square().mean().sqrt()))
    result['prediction_disagreements'] = int((actual['logits'].argmax(1) != reference['logits'].argmax(1)).sum())
    return result


def s2_validate_reference(reference_row, initial_hash, batch_hash, assignments,
                          reset_z_per_cycle):
    """Reject cached outputs from different inputs, hardware or equations."""
    if (reference_row['initial_hash'] != initial_hash or
            reference_row['batch_hash'] != batch_hash or
            reference_row['assignments'] != assignments):
        raise ValueError('Saved reference weights/batch/curve assignments do not match.')
    if reference_row.get('reset_z_per_cycle', False) != reset_z_per_cycle:
        raise ValueError('Saved reference reset_z_per_cycle does not match.')


def s2_worker(o):
    import torch
    import gc
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, inputs, validator, initial_hash = s2_build_model(o, o.dynamics, o.scene, o.representation)
    out = Path(o.output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{o.representation}_{o.scene}_{o.dynamics}'
    if o.ordering_pairs:
        prefix += f'_pairs{o.ordering_pairs}'
    if o.s2_worker == 'numerical':
        x = inputs[:2].cuda()
        # Materialize curve assignments once and retain them for all solvers.
        with torch.no_grad():
            model(x)
        assignments = s2_assignments(model)
        if o.selective_reference:
            if o.dynamics != 'pulse':
                raise ValueError('Selective landing diagnostic requires pulse dynamics.')
            reference_dir = Path(o.selective_reference)
            reference_rows = json.loads((reference_dir / (prefix + '.json')).read_text())
            reference_row = next(r for r in reference_rows
                                 if r['method'] == 'dopri5' and r['tol'] == 1e-6)
            s2_validate_reference(reference_row, initial_hash, digest(x), assignments,
                                  o.reset_z_per_cycle)
            reference = torch.load(reference_dir / (reference_row['name'] + '.pt'), weights_only=False)
            if o.selective_timing:
                # Time plain evaluation forwards for BOTH paths. s2_record
                # copies intermediate frame states only for the legacy solver,
                # so it is not a symmetric end-to-end timing harness.
                configs = [None, (), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]
                if o.selective_timing_rk2:
                    configs = [None, 'rk2_n8', (1, 2, 3, 4)]
                rows = []
                for repeat in range(4):
                    order = configs if repeat % 2 == 0 else list(reversed(configs))
                    for slices in order:
                        method, substeps = ('rk2', 8) if slices == 'rk2_n8' else ('dopri5', 1)
                        s2_set_solver(model, method, substeps, 1e-6)
                        for block in model.PcConvs:
                            block.s2_dtc_landing_slices = None if method == 'rk2' else slices
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        with torch.no_grad():
                            logits = model(x)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        logits = logits.detach().cpu()
                        if s2_assignments(model) != assignments:
                            raise ValueError('Hardware assignments changed during timing.')
                        row = dict(slices=slices, repeat=repeat, warmup=repeat == 0,
                                   method=method, substeps=substeps,
                                   seconds=elapsed, batch_size=len(x), tol=1e-6,
                                   mode='eval_forward_no_grad_no_trace_hooks',
                                   initial_hash=initial_hash, batch_hash=digest(x),
                                   assignments=assignments,
                                   logit_relative_l2=float((logits.double()-reference['logits'].double()).norm()/
                                       reference['logits'].double().norm().clamp_min(1e-30)),
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        rows.append(row)
                        write(out / (prefix + '_selective_timing.json'), rows)
                        print('SELECTIVE_TIMING', row, flush=True)
                return
            rows = []
            for slices in ((), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)):
                for tol in (1e-4, 1e-6):
                    s2_set_solver(model, 'dopri5', 1, tol)
                    for block in model.PcConvs:
                        block.s2_dtc_landing_slices = slices
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = s2_record(model, x, 'pulse')
                    torch.cuda.synchronize()
                    row = dict(slices=slices, tol=tol, seconds=time.perf_counter()-start,
                               errors={k: dict(relative_l2=float((v.double()-reference[k].double()).norm()/
                                            reference[k].double().norm().clamp_min(1e-30)))
                                       for k, v in actual.items()},
                               prediction_flips=int((actual['logits'].argmax(1) != reference['logits'].argmax(1)).sum()),
                               accuracy=float((actual['logits'].argmax(1) ==
                                   torch.load(o.batch, weights_only=False)[1][:len(x)].cpu()).float().mean()),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               solvers=[b._last_s2_dtc_solver_stats for b in model.PcConvs])
                    rows.append(row)
                    write(out / (prefix + '_selective.json'), rows)
                    print('SELECTIVE_PASS', slices, tol, row['errors'], row['seconds'], flush=True)
            return
        configs = [('dopri5', 1, t) for t in (1e-4, 1e-6)]
        configs += [(m, n, 1e-6) for m in ('euler', 'rk2') for n in (1, 2, 4, 8)]
        if o.reset_study and o.dynamics == 'average':
            configs.append(('rk4', 1, 1e-6))
        if o.ordering_study:
            configs = [('rk2', 1, 1e-6)]
            if o.dynamics == 'pulse':
                configs.insert(0, ('dopri5', 1, 1e-6))
        rows = []
        for method, n, tol in configs:
            s2_set_solver(model, 'dopri5' if method == 'rk4' else method, n, tol)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if method == 'rk4':
                # Diagnostic-only substitution at the existing solver boundary:
                # unchanged physical RHS, midpoint projection and frame reset.
                from unittest.mock import patch
                from TorchDiffEqPack.odesolver.fixed_grid_solver import RK4
                def rk4_solve(func, state, options):
                    state, _ = RK4.step(None, func, 0., 1., state)
                    project = options.get('proj_fn')
                    return tuple(project(v) for v in state) if project else state
                with patch('ode_pc.aca_ode_solve', rk4_solve):
                    trace = s2_record(model, x, o.dynamics)
            else:
                trace = s2_record(model, x, o.dynamics)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            assert s2_assignments(model) == assignments, 'Curve assignments changed between solvers'
            name = f'{prefix}_{method}_{n}_{tol:g}'
            torch.save(trace, out/(name+'.pt'))
            row = dict(name=name, method=method, substeps=n, tol=tol, seconds=elapsed,
                reset_z_per_cycle=o.reset_z_per_cycle,
                assignments=assignments, initial_hash=initial_hash, batch_hash=digest(x))
            rows.append(row)
            write(out/(prefix+'.json'), rows)
            print('NUMERICAL_PASS', name, elapsed, flush=True)
    else:
        s2_set_solver(model, o.method, o.substeps, 1e-6)
        if o.train_step:
            model.train()
        attempts = []
        # The parent retries smaller sizes in fresh processes after OOM. CUDA
        # libraries and partial autograd graphs may retain failed allocations.
        for bs in (o.batch_size,):
            try:
                x = inputs[:bs].cuda()
                assert len(x) == bs
                torch.cuda.reset_peak_memory_stats()
                times = []
                for iteration in range(3):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.set_grad_enabled(o.train_step):
                        logits = model(x)
                        if o.train_step:
                            logits.square().mean().backward()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    if iteration:
                        times.append(elapsed)
                    del logits
                result = dict(status='ok', batch_size=bs, median_seconds=statistics.median(times),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), attempts=attempts,
                    dynamics=o.dynamics, scene=o.scene, representation=o.representation,
                    method=o.method, substeps=o.substeps, training=o.train_step,
                    checkpoint=o.checkpoint, initial_hash=initial_hash)
                break
            except RuntimeError as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and 'CUBLAS_STATUS_ALLOC_FAILED' not in str(exc):
                    raise
                attempts.append(dict(batch_size=bs, status='oom', error=str(exc)))
                model.zero_grad(set_to_none=True)
                if 'logits' in locals():
                    del logits
                gc.collect()
                torch.cuda.empty_cache()
        else:
            result = dict(status='oom', attempts=attempts)
        tag = f'{prefix}_{o.method}_{o.substeps}_train{o.train_step}_checkpoint{o.checkpoint}'
        write(out/('benchmark_'+tag+'.json'), result)
        print('BENCHMARK', tag, result, flush=True)
    if validator is not None:
        validator.close_curve_stager()


def s2_report(out):
    import torch
    def setting(row, dynamics):
        if row['method'] == 'dopri5':
            return f"dopri5 tol={row['tol']:g}"
        steps = (75 if dynamics == 'pulse' else 5) * row['substeps']
        return f"{row['method']}, {steps} total steps"

    comparisons = [
        ('Pulse Euler/RK2 with \\(75N\\) steps versus pulse Dopri5', 'pulse', 'pulse'),
        ('**Averaged five-step Euler/RK2 versus pulse reference**', 'average', 'pulse'),
        ('Averaged five-step Euler/RK2 versus accurately integrated averaged ODE', 'average', 'average'),
        ('Accurately integrated averaged ODE versus pulse reference', 'average_ref', 'pulse')]
    records = []
    lines = ['# S2 DTC fixed-batch numerical diagnostics', '',
        'Random initialization (seed 1907), not a trained checkpoint. Six PC blocks: '
        '24→48→96, stage repeats 1:1:1, pool before expansion; one fixed two-image '
        'CiFAIR-100 batch. Five simultaneous 10 ns frames. Ideal final linear head '
        'and pooling isolate recurrent dynamics. No noise or DTC nonidealities. '
        'Dopri5 1e-6 is a provisional reference, not proof of convergence.', '',
        'Dense nonlinear-R uses fixed per-tensor curves; expanded uses fixed per-coupler curves. '
        'Measured ReLU uses fixed per-spin assignments from 0906 FS_V2_T1. '
        'Assignments are hashed and checked across solvers and average/pulse within each representation.', '',
        'Scenarios are cumulative: ideal (no state clipping), clipped, clipped + measured ReLU, '
        'and clipped + measured ReLU + nonlinear-R. Spin variation is disabled. '
        'Enabling clipping does not guarantee this batch reaches the rails.', '',
        'Relative L2 = norm(actual − reference) / norm(reference); values below are ratios, '
        'not percentages or accuracy drops. State error concatenates y/z at all five frame '
        'boundaries in every layer. comparisons.json also contains each layer/frame error, '
        'absolute RMS errors, and logit errors. Refining the averaged solver reduces its '
        'integration error, not necessarily its discrepancy from pulse dynamics.', '']
    for rep in ('dense', 'expanded'):
        for scene in ('ideal', 'clipped', 'measured', 'nonlinear'):
            rows = {d: json.loads((out/f'{rep}_{scene}_{d}.json').read_text()) for d in ('pulse', 'average')}
            assert rows['pulse'][0]['initial_hash'] == rows['average'][0]['initial_hash']
            assert rows['pulse'][0]['batch_hash'] == rows['average'][0]['batch_hash']
            assert rows['pulse'][0]['assignments'] == rows['average'][0]['assignments'], (rep, scene)
            def load(row):
                return torch.load(out/(row['name']+'.pt'), map_location='cpu', weights_only=False)
            refs = {d: load(rows[d][1]) for d in rows}
            lines += ['', f'## {rep}: {scene}', '',
                      '| Comparison | Setting | State relative L2 | Logit relative L2 |',
                      '|---|---|---:|---:|']
            for title, source, target in comparisons:
                selected = ([rows['average'][1]] if source == 'average_ref' else
                            [r for r in rows[source] if r['method'] != 'dopri5' and
                             (source == 'pulse' or r['substeps'] == 1)])
                for row in selected:
                    metrics = s2_error(load(row), refs[target])
                    records.append(dict(representation=rep, scene=scene, comparison=title,
                                        source=row['name'], reference=target, metrics=metrics))
                    lines.append(f"| {title} | {setting(row, source)} | "
                                 f"{metrics['all_frames']['relative_l2']:.6g} | {metrics['logits']['relative_l2']:.6g} |")
            lines += ['', '### Supplement: averaged refinement and reference checks', '',
                      '| Setting | Reference | State relative L2 | Logit relative L2 |',
                      '|---|---|---:|---:|']
            for source in rows:
                for row in rows[source]:
                    if row['method'] == 'dopri5' and row['tol'] == 1e-4:
                        targets = [source]
                    elif source == 'average' and row['method'] != 'dopri5' and row['substeps'] > 1:
                        targets = ['average', 'pulse']
                    else:
                        continue
                    for target in targets:
                        metrics = s2_error(load(row), refs[target])
                        records.append(dict(representation=rep, scene=scene, comparison='supplement',
                                            source=row['name'], reference=target, metrics=metrics))
                        lines.append(f"| {source} {setting(row, source)} | {target} | "
                                     f"{metrics['all_frames']['relative_l2']:.6g} | {metrics['logits']['relative_l2']:.6g} |")
    write(out/'comparisons.json', records)
    lines += ['', '## Runtime and memory', '',
              'RTX 4090; one warmup and two measured iterations, median wall time with CUDA '
              'synchronization. Training means model forward/backward only: no teacher, '
              'optimizer step, or data-loading overhead. Each configuration starts at batch 128; '
              'OOM retries halve the batch in fresh processes. The reported batch is the largest '
              'successful tested size, not an exact capacity limit.', '',
              'Batch-128 benchmarks cover five-step averaged RK2, pulse RK2 with two substeps '
              'per slice (150 total), and pulse Dopri5 at 1e-6. Dense RK2 additionally tests '
              'forward/backward with and without RHS checkpointing. Expanded benchmarks are '
              'evaluation-only. Other Euler/RK2 resolutions are covered by the two-image '
              'numerical runs, not batch-128 memory benchmarks.', '',
              '| Configuration | Batch | Seconds | Allocated GiB | Reserved GiB |',
              '|---|---:|---:|---:|---:|']
    for path in sorted(out.glob('benchmark_*.json')):
        r = json.loads(path.read_text())
        if r['status'] == 'ok':
            lines.append(f"| {path.stem} | {r['batch_size']} | {r['median_seconds']:.4f} | "
                         f"{r['peak_allocated_bytes']/2**30:.3f} | {r['peak_reserved_bytes']/2**30:.3f} |")
        else:
            lines.append(f'| {path.stem} | OOM | — | — | — |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def s2_reset_report(out):
    """Three pinned comparison tables, with matching reset-enabled references."""
    import torch
    records = {}
    def trajectory(rep, scene, dynamics, method, n=1, tol=1e-6):
        key = (rep, scene, dynamics, method, n, tol)
        if key not in records:
            rows = json.loads((out/f'{rep}_{scene}_{dynamics}.json').read_text())
            row = next(r for r in rows if (r['method'], r['substeps'], r['tol']) == (method, n, tol))
            assert row['reset_z_per_cycle'] is True
            records[key] = row, torch.load(out/(row['name']+'.pt'), map_location='cpu', weights_only=False)
        return records[key]
    def error(rep, scene, dynamics, method, n=1, target='pulse', tol=1e-6):
        a, actual = trajectory(rep, scene, dynamics, method, n, tol)
        b, reference = trajectory(rep, scene, target, 'dopri5')
        for key in ('initial_hash', 'batch_hash', 'assignments'):
            assert a[key] == b[key], key
        return 100*s2_error(actual, reference)['logits']['relative_l2']
    def timing(rep, scene, dynamics, n=1, train=False, cp=False):
        tag = f'{rep}_{scene}_{dynamics}_rk2_{n}_train{train}_checkpoint{cp}'
        r = json.loads((out/('benchmark_'+tag+'.json')).read_text())
        if r['status'] != 'ok':
            return 'OOM'
        return f"{r['median_seconds']:.4f}" + (f" @ {r['batch_size']}" if train else '')
    labels = dict(ideal='Ideal', clipped='+ Clipping', measured='+ Measured ReLU', nonlinear='+ Nonlinear-R')
    lines = ['# Cycle-reset S2 DTC: toy-model comparison', '',
        'Randomly initialized six-block C24→48→96, seed 1907; same cached two error images '
        'and 128 timing images as the original persistent-z study. Five 10 ns cycles, '
        'R=50 kΩ, C=500 fF, VDD=0.5 V, inverse-q=5, QF1, 5-bit weights. '
        'Only z is zeroed between complete cycles, four resets per layer; y is retained. '
        'No reset occurs between PWM slices or integration substeps. No trained checkpoint.', '',
        'Errors are final-logit relative L2 percentages. References are freshly generated '
        'reset-enabled slice-wise pulse Dopri5 at atol=rtol=1e-6. Dense and expanded each '
        'use matching weights, scales and sampled curves within their comparisons. '
        'Scenarios cumulatively add clipping, fixed per-spin 0906 FS_V2_T1 measured ReLU, '
        'then full-range FS_V2_T1 nonlinear-R (dense per-tensor; expanded per-coupler). '
        'Noise, spin and DTC variation are disabled. Pooling and final head are ideal.', '',
        'Timings: strict FP32/TF32 off, one warmup and two measured forwards or '
        'forward/backward passes, no teacher/optimizer/data-loading. Forward timings use '
        'batch 128; training starts at 128 and halves after OOM in fresh processes. '
        'Training cells report seconds @ batch. CP means RHS checkpointing.', '',
        '## 1. Pulse RK2 versus pulse Dopri5', '',
        '| Scenario | N | Final-logit relative L2 error dense / expanded (%) | Dense forward (s) | Dense training (s @ batch) | Dense training + CP (s @ batch) | Expanded forward (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    numerical = []
    for scene in ('ideal', 'clipped', 'nonlinear'):
        for n in (1, 2, 4, 8):
            errors = [error(rep, scene, 'pulse', 'rk2', n) for rep in ('dense', 'expanded')]
            times = [timing('dense', scene, 'pulse', n), timing('dense', scene, 'pulse', n, True),
                     timing('dense', scene, 'pulse', n, True, True), timing('expanded', scene, 'pulse', n)]
            numerical.append(dict(table=1, scene=scene, substeps=n, logit_errors_percent=errors))
            lines.append(f"| {labels[scene]} | {n} | {errors[0]:.4f} / {errors[1]:.4f} | " + ' | '.join(times) + ' |')
    lines += ['', '## 2. Averaged five-step RK2 versus pulse reference', '',
        '| Scenario | Method | Final-logit relative L2 error dense / expanded (%) | Dense forward (s) | Dense training (s @ batch) | Dense training + CP (s @ batch) | Expanded forward (s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scene in labels:
        errors = [error(rep, scene, 'average', 'rk2') for rep in ('dense', 'expanded')]
        times = [timing('dense', scene, 'average'), timing('dense', scene, 'average', train=True),
                 timing('dense', scene, 'average', train=True, cp=True), timing('expanded', scene, 'average')]
        numerical.append(dict(table=2, scene=scene, logit_errors_percent=errors))
        lines.append(f"| {labels[scene]} | RK2 | {errors[0]:.4f} / {errors[1]:.4f} | " + ' | '.join(times) + ' |')
    lines += ['', '## 3. Averaged refinement: expanded, clipping + measured ReLU + nonlinear-R', '',
        '| Averaged method | Error against accurate reset-averaged solution (%) | Error against reset-pulse reference (%) |',
        '|---|---:|---:|']
    for method, n in [('euler', 1), ('rk2', 1), ('rk2', 2), ('rk2', 4), ('rk2', 8), ('rk4', 1), ('dopri5', 1)]:
        errors = [error('expanded', 'nonlinear', 'average', method, n, target=t) for t in ('average', 'pulse')]
        numerical.append(dict(table=3, method=method, steps=5*n, logit_errors_percent=errors))
        label = 'Accurate averaged, Dopri5 1e-6' if method == 'dopri5' else f'{method.upper()}, {5*n} steps'
        lines.append(f'| {label} | {errors[0]:.6f} | {errors[1]:.6f} |')
    reference_checks = []
    baseline_checks = []
    for rep in ('dense', 'expanded'):
        for scene in labels:
            for dynamics in ('average', 'pulse'):
                reference_checks.append(dict(representation=rep, scene=scene, dynamics=dynamics,
                    tol_1e4_vs_1e6_logit_error_percent=error(rep, scene, dynamics, 'dopri5', target=dynamics, tol=1e-4)))
                original = ROOT/'artifacts/s2_dtc_validation_20261104/numerical'/f'{rep}_{scene}_{dynamics}.json'
                if original.exists():
                    old = json.loads(original.read_text())[0]
                    new, _ = trajectory(rep, scene, dynamics, 'dopri5')
                    checks = {k: old[k] == new[k] for k in ('initial_hash', 'batch_hash', 'assignments')}
                    assert all(checks.values()), (rep, scene, dynamics, checks)
                    baseline_checks.append(dict(representation=rep, scene=scene, dynamics=dynamics, **checks))
    write(out/'results.json', dict(tables=numerical, reference_tolerance_checks=reference_checks,
                                  original_study_identity_checks=baseline_checks))
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('RESET_STUDY_COMPLETE', out/'summary.md', flush=True)


def s2_ordering_report(out):
    """Every candidate compared to the same pulse Dopri5, never to one another."""
    import torch
    rows = []
    lines = ['# RK2 ordering-correction study', '',
        'Dense 24→48→96, 1:1:1 repeats, fixed input batch and initialization. '
        'Five 10 ns frames; R=50 kΩ, C=500 fF, VDD=0.5 V, q=0.1. '
        'Cumulative deterministic scenarios; no noise, spin variation or DTC variation. '
        '0906 per-spin ReLU and full-range per-tensor coupler curves use FS_V2_T1. '
        'The reference for every accuracy row is pulse Dopri5 (atol=rtol=1e-6). '
        'Errors are logit relative L2 percentages, not classification accuracy. '
        'Timing is synchronized forward or forward/backward, excluding optimizer and teacher; '
        'one warmup, median of two measured iterations. Batch starts at 128 and halves after OOM.', '',
        '| Scenario | Method | Logit error (%) | State error (%) | Eval s / batch | F+B s / batch | F+B checkpoint s / batch |',
        '|---|---|---:|---:|---|---|---|']
    for scene in ('ideal', 'clipped', 'measured', 'nonlinear'):
        refrow = json.loads((out/f'dense_{scene}_pulse.json').read_text())[0]
        reference = torch.load(out/(refrow['name']+'.pt'), weights_only=False)
        for dynamics, pairs in [('pulse', 0)] + [('average', r) for r in (0, 1, 2, 4, 7)]:
            prefix = f'dense_{scene}_{dynamics}' + (f'_pairs{pairs}' if pairs else '')
            record = json.loads((out/(prefix+'.json')).read_text())[-1]
            for key in ('assignments', 'initial_hash', 'batch_hash'):
                assert record[key] == refrow[key], (scene, prefix, key)
            actual = torch.load(out/(record['name']+'.pt'), weights_only=False)
            errors = s2_error(actual, reference)
            times, benchmarks = [], []
            for train, cp in ((False, False), (True, False), (True, True)):
                path = out/f'benchmark_{prefix}_rk2_1_train{train}_checkpoint{cp}.json'
                b = json.loads(path.read_text()) if path.exists() else {'status': 'not_run'}
                benchmarks.append(b)
                times.append(f"{b['median_seconds']:.4f} / {b['batch_size']}" if b['status']=='ok' else b['status'])
            label = 'Pulse RK2 N=1' if dynamics == 'pulse' else ('Averaged RK2' if not pairs else f'RK2 + {pairs} pairs')
            lines.append(f"| {scene} | {label} | {100*errors['logits']['relative_l2']:.5f} | "
                         f"{100*errors['all_frames']['relative_l2']:.5f} | " + ' | '.join(times) + ' |')
            rows.append(dict(scene=scene, method=label, pairs=pairs, metrics=errors, benchmarks=benchmarks))
    write(out/'ordering_results.json', rows)
    (out/'ordering_summary.md').write_text('\n'.join(lines)+'\n')


def s2_pretrained_model(path, dynamics, pairs, substeps, device):
    """Pinned 1022/cascode PT recipe, but ideal hardware apart from PC clipping.

    Keep the fixed measured activation as the common activation definition.
    Use the production loader and physical quantizing wrapper; do not repeat
    PT's initialization compensation on already trained weights.
    """
    import torch
    from inference_utils import load_and_prepare_model
    from ode_pc import S2DTCFFFB, S2DTCPulseFFFB, S2DTCWrapper2State
    cls = S2DTCFFFB if dynamics == 'average' else S2DTCPulseFFFB
    flags = dict(enable_spin_variation=False, enable_summing_current_noise=False,
                 enable_coupler_noise=False, enable_slow_summing_current=False,
                 enable_slow_coupler_noise=False, enable_dtc_nonideality=False)
    # The production inference loader performs in-place weight quantization.
    with torch.no_grad():
        model = load_and_prepare_model(str(path), device, pc_conv_layer=None,
            conv_only=True, fuse_bn=False, noise_level=0.0,
            final_head_config=dict(final_head_quantize=False, final_head_clamp=False,
                                   final_adc_noise_lsb=0.0),
            ode_params=dict(ode_block=cls, method='rk2', t_end=1.75, n_steps=5,
                tol=1e-6, toggle_n_cycles=5, toggle_timing_mode='fixed',
                toggle_y_time=1e-8, z_over_y_time=1, toggle_timing_R=67e3,
                toggle_timing_C=500e-15, y_init_with='x', z_init_with='0',
                s2_dtc_substeps=substeps, s2_dtc_ordering_pairs=pairs, **flags),
            ode_wrapper_params=dict(ode_wrapper=S2DTCWrapper2State,
                R=67e3, C=500e-15, v_dd=0.6, one_over_q=6, w_bits=5,
                weight_quant_factor_bits=1, thermal_noise=False, enob=None,
                nonlinear_R=False, nonlinear_R_train_mode='none',
                enable_measured_activation=True,
                activation_curve_path=str(ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv'),
                activation_corner='MC18', activation_interpolation='piecewise_linear',
                activation_curve_sharing='per_model',
                activation_normalize_positive_endpoint=False, fuse_measured_activation=False,
                **flags))
    model.eval()
    assert model.measured_activation_scope == 'all'
    assert model.final_head_type == 'digital'
    signature = {}
    for i, block in enumerate(model.PcConvs):
        assert type(block) is cls and block.physical
        assert block.s2_dtc_substeps == substeps and block.toggle_n_cycles == 5
        for key in flags:
            assert not getattr(block, key, False), (i, key)
        assert not getattr(block, '_nonlinear_R_training_pkg', None)
        for key in ('FFconv', 'FBconv'):
            w = getattr(block, key).weight.detach().cpu()
            levels = w * 15 if dynamics == 'average' else w
            assert torch.allclose(levels, levels.round(), atol=2e-6, rtol=0)
            signature[f'{i}/{key}'] = levels.round().to(torch.int32)
        for key in ('s_ff', 's_fb', 'R', 'C', 'v_dd', 'toggle_y_time'):
            signature[f'{i}/{key}'] = torch.as_tensor(getattr(block, key)).cpu()
    signature['classifier'] = model.linear.weight.detach().cpu()
    signature['q'] = model.state_q
    # Check actual fixed curves, including the final activation, not just flags.
    sites = {name: mod for name, mod in model.named_modules()
             if hasattr(mod, 'active_corner')}
    assert len(sites) == model.num_layers + 1, list(sites)
    for name, mod in sites.items():
        assert mod.active_corner == 'MC18', (name, mod.active_corner)
        signature[f'activation/{name}'] = mod.state_dict()
    from measured_pooling import GlobalAvgPool2d, MeasuredAvgPool2d
    assert isinstance(model.global_avg_pool2d, GlobalAvgPool2d)
    assert not any(isinstance(mod, MeasuredAvgPool2d) for mod in model.modules())
    assert model.linear.head_config['final_head_quantize'] is False
    assert model.linear.head_config['final_head_clamp'] is False
    assert model.linear.head_config['final_adc_noise_lsb'] == 0
    return model, digest(signature)


def s2_pretrained_diagnostic():
    """One cached test batch, four dense physical methods, no training writes."""
    import gc
    import numpy as np
    import torch
    from inference_utils import get_test_data
    from pc_model import logits_for_loss
    from launch_scripts.find_tc_pretrain import find_pretrain
    p = argparse.ArgumentParser()
    p.add_argument('--s2-pretrained', action='store_true')
    p.add_argument('--pretrain-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=128,
                   help='Number of fixed test images, not a full dataset sweep.')
    p.add_argument('--microbatch-size', type=int, default=16)
    p.add_argument('--gpu-memory-fraction', type=float, default=0.2,
                   help='Cap this diagnostic CUDA allocator; leave room for concurrent FT.')
    o = p.parse_args()
    if min(o.batch_size, o.microbatch_size) < 1:
        p.error('Batch sizes must be positive')
    if not 0 < o.gpu_memory_fraction <= 1:
        p.error('GPU memory fraction must be in (0, 1]')
    out = Path(o.output).resolve()
    # Production discovery rejects incomplete, ambiguous or mismatched models.
    path = find_pretrain(o.pretrain_root,
        inp=[4, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 96, 96, 96, 96],
        out=[24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 96, 96, 96, 96, 96],
        pool=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        task='cifar100', img_type='CiFAIR', block='S2DTCFFFB', bits=12,
        center=False, selection='last')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['init_args']['s2_dtc']['method'] == 'rk2'
    assert saved['init_args']['s2_dtc']['s2_dtc_substeps'] == 1
    # Completed PT checkpoints retain constructor metadata here; unlike interim
    # recovery checkpoints, they need not duplicate these fields at top level.
    model_args = saved['init_args']['model_args']
    assert model_args['measured_activation_scope'] == 'all'
    assert model_args['final_head_type'] == 'digital'
    assert saved.get('measured_activation_scope', 'all') == 'all'
    assert saved.get('final_head', {'type': 'digital'})['type'] == 'digital'
    del saved
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(o.gpu_memory_fraction)
    torch.set_num_threads(4)
    torch.manual_seed(4096)
    np.random.seed(4096)
    random.seed(4096)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['DATALOADER_NUM_WORKERS'] = '0'
    loader = get_test_data(test_bs=o.batch_size, img_type='CiFAIR', task='cifar100',
                           shuffle=False, input_quant_bits=12, center_student_input=False)
    batch = next(iter(loader))
    inputs, labels = batch[0].cpu(), batch[1].cpu()
    assert len(inputs) == o.batch_size and inputs.shape[1:] == (4, 16, 16)
    del loader, batch
    torch.save((inputs, labels), out/'fixed_test_batch.pt')
    curve = ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv'
    metadata = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        batch_hash=digest((inputs, labels)), test_images=len(inputs),
        microbatch_size=o.microbatch_size, gpu_memory_fraction=o.gpu_memory_fraction,
        curve=str(curve),
        curve_sha256=hashlib.sha256(curve.read_bytes()).hexdigest(),
        reference='pulse_rk2_n8', representation='dense',
        settings=dict(R=67e3, C=500e-15, VDD=.6, one_over_q=6, frames=5,
            frame_seconds=1e-8, weight_bits=5, weight_quant_factor_bits=1,
            activation='fixed MC18, all sites', clipping=True,
            noise=False, variation=False, nonlinear_R=False, measured_pooling=False,
            head_quantization=False, head_clipping=False, head_noise=False),
        note='Inputs are cached once from the production test loader, including its input preprocessing. '
             'No resampling between methods. Logits restored to model units; no FT/optimizer updates.')
    write(out/'configuration.json', metadata)
    variants = [('pulse_rk2_n8', 'pulse', 0, 8), ('averaged_rk2', 'average', 0, 1),
                ('averaged_rk2_pairs4', 'average', 4, 1), ('pulse_rk2_n1', 'pulse', 0, 1)]
    rows, reference, common_signature = [], None, None
    for name, dynamics, pairs, substeps in variants:
        print('START', name, flush=True)
        model, signature = s2_pretrained_model(path, dynamics, pairs, substeps, 'cuda')
        if common_signature is None:
            common_signature = signature
        assert signature == common_signature, 'Quantized levels/scales/fixed activations differ'
        parts = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for offset in range(0, len(inputs), o.microbatch_size):
                raw = model(inputs[offset:offset+o.microbatch_size].cuda())
                parts.append(logits_for_loss(raw, model).cpu())
                print(name, min(offset+o.microbatch_size, len(inputs)), '/', len(inputs),
                      'elapsed_seconds=', round(time.perf_counter()-start, 2), flush=True)
        torch.cuda.synchronize()
        logits = torch.cat(parts)
        assert torch.isfinite(logits).all(), name
        if reference is None:
            reference = logits
        delta = logits.double() - reference.double()
        flips = int((logits.argmax(1) != reference.argmax(1)).sum())
        row = dict(method=name, logit_relative_l2_percent=float(100*delta.norm()/reference.double().norm().clamp_min(1e-30)),
            prediction_flips=flips, flip_percent=100*flips/len(inputs),
            accuracy_percent=float(100*(logits.argmax(1) == labels).double().mean()),
            seconds=time.perf_counter()-start,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            common_hardware_hash=signature)
        rows.append(row)
        torch.save(dict(logits=logits, labels=labels), out/(name+'.pt'))
        write(out/'results.json', rows)
        print('RESULT', row, flush=True)
        del model, parts, raw
        gc.collect()
        torch.cuda.empty_cache()
    lines = ['# S2 pretrained deterministic dynamics comparison', '',
        f'Checkpoint: `{path}`', '',
        f'{len(inputs)} fixed CiFAIR-100 test images; microbatch {o.microbatch_size}; dense convolutions.',
        'Common reference: pulse RK2 N=8 (600 steps over five 10 ns frames).',
        'Identical quantized weights, scales, fixed 1022 TT MC18 activation and state clipping.',
        'Noise, variation, nonlinear-R, measured pooling and head nonidealities disabled.',
        'Relative logit L2 = 100 ||logits - reference|| / ||reference||, after model-unit restoration.', '',
        '| Method | Logit error (%) | Prediction flips | Flip rate (%) | Accuracy (%) | Seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['logit_relative_l2_percent']:.6f} | "
            f"{row['prediction_flips']} / {len(inputs)} | {row['flip_percent']:.4f} | "
            f"{row['accuracy_percent']:.4f} | {row['seconds']:.2f} |")
    lines += ['', 'This is a one-batch diagnostic, not full-test or corner accuracy. '
              'N=8 is a numerical reference, not an analytic solution. No FT was launched.', '']
    (out/'summary.md').write_text('\n'.join(lines))
    print('COMPLETE', out/'summary.md', flush=True)


def s2_slice_reference_identity(model, settings, curve_paths):
    """Fingerprint actual transient assignments and physical hardware, not seeds."""
    blocks = []
    for block in model.PcConvs:
        physical = {key: getattr(block, key, None) for key in (
            'R', 'C', 'v_dd', 'beta_c', 'q', 's_ff', 's_fb',
            'toggle_y_time', 'toggle_n_cycles',
            's2_dtc_reset_z_per_cycle', 'pulse_weight_encoding',
            'enable_spin_variation', 'enable_summing_current_noise',
            'enable_coupler_noise', 'enable_slow_summing_current',
            'enable_slow_coupler_noise', 'enable_dtc_nonideality')}
        package = getattr(block, '_nonlinear_R_training_pkg', None)
        if package is not None:
            physical['nonlinear_bank'] = {key: package.get(key) for key in (
                'R', 'v_grid', 'R_left', 'R_slope', 'lengths')}
        blocks.append(physical)
    hardware = dict(settings=settings, blocks=blocks, state_q=getattr(model, 'state_q', None),
        head=getattr(model.linear, 'head_config', None),
        curves={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in curve_paths.items()})
    return dict(assignments_sha256=s2_assignments(model), hardware_sha256=digest(hardware))


def s2_validate_slice_reference(previous, current):
    for key in ('checkpoint_sha256', 'batch_hash', 'model_hash', 'microbatch',
                'settings', 'hardware_identity'):
        if key not in previous:
            raise ValueError(f'Cached reference lacks {key}; verify/regenerate its identity before reuse.')
        if previous[key] != current[key]:
            raise ValueError(f'Cached reference {key} does not match this run.')


def s2_slice_study():
    """Matched trained-model pulse schedule study; no training/checkpoint writes."""
    import numpy as np
    import torch
    from inference_utils import load_and_prepare_model
    from ode_pc import S2DTCPulseFFFB, S2DTCQATTester2State
    from pc_model import logits_for_loss
    p = argparse.ArgumentParser()
    p.add_argument('--s2-slice-study', action='store_true')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--microbatch-size', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cycle-sampling-study', action='store_true')
    p.add_argument('--branch-sampling-study', action='store_true')
    p.add_argument('--baseline-ode-file', help='Pre-optimization ode_pc.py snapshot for matched timing.')
    p.add_argument('--reference-dir', help='Completed trained-model slice study supplying cached reference logits.')
    o = p.parse_args()
    out = Path(o.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.manual_seed(4096)
    np.random.seed(4096)
    random.seed(4096)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, labels = torch.load(o.batch, map_location='cpu', weights_only=False)
    inputs, labels = inputs[:16], labels[:16]
    assert inputs.shape == (16, 4, 16, 16)
    from input_preprocessing import prepare_student_input
    saved = torch.load(o.checkpoint, map_location='cpu', weights_only=False)
    preprocessing = saved['student_preprocessing']
    assert preprocessing['input_quant_bits'] == 12
    assert not preprocessing['center_student_input']
    inputs = prepare_student_input(inputs, bits=12, center=False)
    torch.save((inputs, labels), out/'fixed_test_batch.pt')
    flags = dict(enable_spin_variation=False, enable_summing_current_noise=False,
                 enable_coupler_noise=False, enable_slow_summing_current=False,
                 enable_slow_coupler_noise=False, enable_dtc_nonideality=False)
    with torch.no_grad():
        model = load_and_prepare_model(o.checkpoint, 'cuda', pc_conv_layer=None,
            conv_only=True, fuse_bn=False, noise_level=0.,
            final_head_config=dict(final_head_quantize=False, final_head_clamp=False,
                                   final_adc_noise_lsb=0.),
            ode_params=dict(ode_block=S2DTCPulseFFFB, method='rk2', t_end=1.75,
                n_steps=5, tol=1e-6, toggle_n_cycles=5, toggle_timing_mode='fixed',
                toggle_y_time=1e-8, z_over_y_time=1, toggle_timing_R=67e3,
                toggle_timing_C=500e-15, s2_dtc_substeps=1, s2_dtc_ordering_pairs=0,
                s2_dtc_slice_policy='full', **flags),
            ode_wrapper_params=dict(ode_wrapper=S2DTCQATTester2State,
                R=67e3, C=500e-15, v_dd=.6, one_over_q=6, w_bits=5,
                weight_quant_factor_bits=2, thermal_noise=False, enob=None,
                enable_measured_activation=True,
                activation_curve_path=str(ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/fs_25_2.csv'),
                activation_corner='MC18', activation_interpolation='piecewise_linear',
                activation_curve_sharing='per_model', fuse_measured_activation=False,
                activation_normalize_positive_endpoint=False,
                nonlinear_R=True, nonlinear_R_train_mode='exact_curve',
                nonlinear_R_table=str(ROOT/'hardware_data/mc_45_corners/coupler_low_volt_cascode/fs_25_2.csv'),
                nonlinear_R_curve_sharing='shared', nonlinear_R_mc_quantity='conductance',
                nonlinear_R_curve_sampling='empirical_with_replacement', nonlinear_R_curve_seed=4096,
                **flags))
    model.eval()
    assert saved['epoch'] == 140 and len(model.PcConvs) == 16
    for i, block in enumerate(model.PcConvs):
        for key in ('s_ff', 's_fb'):
            torch.testing.assert_close(getattr(block, key).cpu(), saved['net'][f'PcConvs.{i}.{key}'], rtol=0, atol=0)
        for module in (block.FFconv, block.FBconv):
            torch.testing.assert_close(module.weight, module.weight.round(), rtol=0, atol=2e-6)
    torch.testing.assert_close(model.linear.weight.cpu(), saved['net']['linear.weight'],rtol=0,atol=0)
    del saved
    microbatch = min(16, o.microbatch_size)
    while True:
        try:
            with torch.no_grad():
                model(inputs[:microbatch].cuda())
            break
        except torch.cuda.OutOfMemoryError:
            if microbatch == 1:
                raise
            microbatch //= 2
            torch.cuda.empty_cache()
    assignments = s2_assignments(model)
    initial_hash = digest(model.state_dict())
    settings = dict(R=67e3,C=500e-15,VDD=.6,inverse_q=6,cycles=5,cycle_ns=10,
        activation='1022 FS_V2_T1 MC18 all sites', nonlinear_R='cascode FS_V2_T1 shared/conv',
        current_noise=False,spin_variation=False,dtc_variation=False,
        measured_pooling=False,head_quantization=False,head_noise=False,
        weight_encoding='baked QF2; no requantization; normalized->integer x15')
    reference_config = dict(
        checkpoint_sha256=hashlib.sha256(Path(o.checkpoint).read_bytes()).hexdigest(),
        model_hash=initial_hash, batch_hash=digest((inputs,labels)), microbatch=microbatch,
        settings=settings, hardware_identity=s2_slice_reference_identity(model, settings, dict(
            relu=ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/fs_25_2.csv',
            coupler=ROOT/'hardware_data/mc_45_corners/coupler_low_volt_cascode/fs_25_2.csv')))
    configs = [dict(name='slice_dopri5', method='dopri5', n=1, landing=None),
               dict(name='pulse_rk2_n8', method='rk2', n=8, landing=None),
               dict(name='pulse_rk2_n1', method='rk2', n=1, landing=None)]
    for landing in ((), (1,), (1,2), (1,2,3), (1,2,3,4)):
        configs.append(dict(name='selective_'+('_'.join(map(str,landing)) or 'none'),
                            method='dopri5', n=1, landing=landing))
    for policy in ('drop', 'merge'):
        for label, keep, k in [('first4', (1,2,3,4), 0),
                               ('alternating8', tuple(range(1,16,2)), 0),
                               ('first4_random2', (1,2,3,4), 2),
                               ('first4_random4', (1,2,3,4), 4),
                               ('first4_random7', (1,2,3,4), 7)]:
            configs.append(dict(name=policy+'_'+label, method='rk2', n=1,
                                landing=None, policy=policy, keep=keep, k=k))
    reference = None
    if o.cycle_sampling_study or o.branch_sampling_study:
        if not o.reference_dir:
            p.error('Cycle sampling study requires --reference-dir.')
        previous = Path(o.reference_dir)
        previous_config = json.loads((previous/'configuration.json').read_text())
        s2_validate_slice_reference(previous_config, reference_config)
        reference = torch.load(previous/'slice_dopri5_repeat0.pt', map_location='cpu', weights_only=False)
        configs = [dict(name='pulse_rk2_n8', method='rk2', n=8, landing=None),
                   dict(name='pulse_rk2_n1', method='rk2', n=1, landing=None)]
        for policy in ('drop','merge'):
            configs.append(dict(name=policy+'_first4',method='rk2',n=1,landing=None,
                                policy=policy,keep=(1,2,3,4),k=0))
            for k in (2,4):
                for sampling in ('per_forward','per_cycle'):
                    configs.append(dict(name=f'{policy}_random{k}_{sampling}',method='rk2',n=1,
                                        landing=None,policy=policy,keep=(1,2,3,4),k=k,sampling=sampling))
    from ode_pc import _S2DTCMixin
    optimized_reduced = _S2DTCMixin._s2_dtc_reduced_schedule
    optimized_interval = _S2DTCMixin._s2_dtc_interval
    baseline_methods = None
    if o.branch_sampling_study:
        if not o.baseline_ode_file:
            p.error('Branch optimization study requires --baseline-ode-file.')
        # Compile only the two pre-optimization methods; use the same live
        # model, hardware realization, libraries and benchmark process.
        import ast
        import ode_pc
        tree = ast.parse(Path(o.baseline_ode_file).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_S2DTCMixin')
        names = ('_s2_dtc_reduced_schedule', '_s2_dtc_interval')
        functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(vars(ode_pc))
        exec(compile(ast.Module(body=functions, type_ignores=[]), o.baseline_ode_file, 'exec'), namespace)
        baseline_methods = [namespace[name] for name in names]
        configs = [dict(name='pulse_rk2_n8',method='rk2',n=8,landing=None),
                   dict(name='pulse_rk2_n1',method='rk2',n=1,landing=None)]
        for policy in ('drop', 'merge'):
            for count, k in ((4,0), (4,2), (4,4), (5,2), (6,2)):
                for implementation, branches in (('baseline','shared'), ('optimized','shared'),
                                                  ('optimized','independent')):
                    if k == 0 and branches == 'independent':
                        continue
                    configs.append(dict(name=f'{policy}_keep{count}_random{k}_{implementation}_{branches}',
                                        method='rk2',n=1,landing=None,policy=policy,
                                        keep=tuple(range(1,count+1)),k=k,sampling='per_cycle',
                                        branches=branches,implementation=implementation))
    write(out/'configuration.json', dict(checkpoint=o.checkpoint, **reference_config,
        student_preprocessing=preprocessing,
        reference_dir=o.reference_dir,
        baseline_ode_file=o.baseline_ode_file,
        baseline_sha256=(hashlib.sha256(Path(o.baseline_ode_file).read_bytes()).hexdigest()
                         if o.baseline_ode_file else None),
        samples=16, repeats=o.repeats, configs=configs, representation='dense'))
    rows = []
    for repeat in range(o.repeats+1):
        order = configs if repeat % 2 == 0 else list(reversed(configs))
        for config in order:
            if config.get('implementation') == 'baseline':
                _S2DTCMixin._s2_dtc_reduced_schedule, _S2DTCMixin._s2_dtc_interval = baseline_methods
            else:
                _S2DTCMixin._s2_dtc_reduced_schedule = optimized_reduced
                _S2DTCMixin._s2_dtc_interval = optimized_interval
            seed = 8100 + max(0,repeat-1)
            # Reset selection RNG per microbatch below: one realization across
            # all sixteen images, identical between paired drop/merge methods.
            s2_set_solver(model, config['method'], config['n'], 1e-6)
            for block in model.PcConvs:
                block.s2_dtc_landing_slices = config['landing']
                block.s2_dtc_slice_policy = config.get('policy','full')
                block.s2_dtc_keep_slices = config.get('keep',())
                block.s2_dtc_random_k = config.get('k',0)
                block.s2_dtc_slice_sampling = config.get('sampling','per_forward')
                block.s2_dtc_branch_sampling = config.get('branches','shared')
            parts, selections = [], []
            torch.cuda.reset_peak_memory_stats()
            elapsed = 0.
            with torch.no_grad():
                for offset in range(0,16,microbatch):
                    torch.manual_seed(seed)
                    batch = inputs[offset:offset+microbatch].cuda()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    logits = logits_for_loss(model(batch),model)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter()-start
                    parts.append(logits.cpu())
                    if config.get('policy'):
                        selections.append([(b._last_s2_dtc_selected_slices_by_cycle
                                            if config.get('implementation') == 'baseline' else
                                            b._last_s2_dtc_branch_slices_by_cycle) for b in model.PcConvs])
            logits = torch.cat(parts)
            assert torch.isfinite(logits).all()
            if reference is None:
                reference = logits
            assert s2_assignments(model) == assignments
            assert digest(model.state_dict()) == initial_hash, 'Weights/buffers changed'
            delta = logits.double()-reference.double()
            row = dict(name=config['name'], repeat=repeat, warmup=repeat==0, seed=seed,
                       seconds=elapsed, logit_relative_l2=float(delta.norm()/reference.double().norm()),
                       prediction_flips=int((logits.argmax(1)!=reference.argmax(1)).sum()),
                       accuracy=float((logits.argmax(1)==labels).float().mean()),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), selections=selections)
            rows.append(row)
            torch.save(logits,out/f"{config['name']}_repeat{repeat}.pt")
            write(out/'results.json',rows)
            print('SLICE_STUDY', {k:v for k,v in row.items() if k!='selections'},flush=True)
    report = []
    baseline = 'pulse_rk2_n1' if (o.cycle_sampling_study or o.branch_sampling_study) else 'slice_dopri5'
    base = statistics.median(r['seconds'] for r in rows if r['name']==baseline and not r['warmup'])
    for config in configs:
        subset = [r for r in rows if r['name']==config['name'] and not r['warmup']]
        report.append(dict(name=config['name'], median_seconds=statistics.median(r['seconds'] for r in subset),
            speedup=base/statistics.median(r['seconds'] for r in subset),
            error_mean_percent=100*statistics.mean(r['logit_relative_l2'] for r in subset),
            error_std_percent=100*statistics.pstdev(r['logit_relative_l2'] for r in subset),
            flips=[r['prediction_flips'] for r in subset], accuracy=[r['accuracy'] for r in subset]))
    write(out/'summary.json',report)
    print('SLICE_STUDY_FINISHED',out,flush=True)


def s2_pulse_ft_cache_diagnostic():
    """Small-batch production pulse FT; compare worktrees without runtime flags.

    Reuse the actual FT argv, checkpoint, trainer, teacher and SRRL loss. Only
    the batch size, pulse selection, checkpoint policy and diagnostic outputs
    are overridden. No trained checkpoint is saved or live process modified.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--s2-pulse-ft-cache', action='store_true')
    p.add_argument('--source-selection', choices=('full_param_last', 'latest'))
    p.add_argument('--source-root', required=True)
    p.add_argument('--ft-args', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-cache', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--substeps', type=int, default=1)
    p.add_argument('--checkpoint-portion', type=float, default=1.0)
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--memory-fraction', type=float, default=.2)
    p.add_argument('--phase', choices=('parity', 'parity-production', 'benchmark'), default='benchmark')
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--merge-training-suite', action='store_true')
    p.add_argument('--slice-policy', choices=('full', 'merge'), default='full')
    p.add_argument('--keep-slices', default='1,2,3,4')
    p.add_argument('--random-k', type=int, default=0)
    p.add_argument('--branch-sampling', choices=('shared', 'independent'), default='shared')
    p.add_argument('--stable-timing', action='store_true')
    p.add_argument('--max-measured-steps', type=int, default=24)
    p.add_argument('--checkpoint-root', help='Override the continuation source, preserving latent QAT weights.')
    o = p.parse_args()
    if o.substeps < 1 or o.grad_accum_steps < 1:
        p.error('Substeps and gradient accumulation must be positive.')
    if o.phase != 'benchmark' and o.grad_accum_steps != 1:
        p.error('Parity captures require grad-accum-steps=1.')
    o.ft_args = str(Path(o.ft_args).resolve())
    o.output = str(Path(o.output).resolve())
    o.batch_cache = str(Path(o.batch_cache).resolve())
    if o.merge_training_suite:
        output = Path(o.output)
        output.mkdir(parents=True, exist_ok=False)
        cases = [('full_N1_before', 'full', 4, 'shared'),
                 ('merge14_random2_shared', 'merge', 4, 'shared'),
                 ('merge15_random2_shared', 'merge', 5, 'shared'),
                 ('merge14_random2_independent', 'merge', 4, 'independent'),
                 ('merge15_random2_independent', 'merge', 5, 'independent'),
                 ('full_N1_after', 'full', 4, 'shared')]
        results = []
        for name, policy, keep, sharing in cases:
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--s2-pulse-ft-cache', '--source-root', o.source_root,
                       '--ft-args', o.ft_args, '--output', str(output / name),
                       '--batch-cache', o.batch_cache, '--batch-size', str(o.batch_size),
                       '--substeps', '1', '--checkpoint-portion', str(o.checkpoint_portion),
                       '--grad-accum-steps', str(o.grad_accum_steps),
                       '--memory-fraction', str(o.memory_fraction), '--warmups', str(o.warmups),
                       '--steps', str(o.steps), '--max-measured-steps', str(o.max_measured_steps),
                       '--stable-timing', '--slice-policy', policy,
                       '--keep-slices', ','.join(str(i) for i in range(1, keep + 1)),
                       '--random-k', '2' if policy == 'merge' else '0',
                       '--branch-sampling', sharing, '--phase', 'benchmark']
            if o.source_selection is not None:
                command.extend(['--source-selection', o.source_selection])
            if o.checkpoint_root:
                command.extend(['--checkpoint-root', o.checkpoint_root])
            print('TRAIN_BENCHMARK_START', name, flush=True)
            with (output / (name + '.log')).open('w') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                write(output / 'failure.json', dict(case=name, returncode=completed.returncode))
                raise RuntimeError(f'Benchmark {name} failed; inspect {output / (name + ".log")}')
            result = json.loads((output / name / 'result.json').read_text())
            result['case'] = name
            results.append(result)
            assert result['initial_model_hash'] == results[0]['initial_model_hash']
            assert result['initial_auxiliary_hash'] == results[0]['initial_auxiliary_hash']
            assert result['batch_hash'] == results[0]['batch_hash']
            baseline = results[0]['median_seconds']
            for row in results:
                row['time_change_vs_full_N1_percent'] = 100 * (row['median_seconds'] / baseline - 1)
                row['speedup_vs_full_N1'] = baseline / row['median_seconds']
            write(output / 'summary.json', results)
            print('TRAIN_BENCHMARK_FINISHED', name, result['median_seconds'], flush=True)
        return
    global ROOT
    ROOT = Path(o.source_root).resolve()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    import numpy as np
    import torch
    import train_ode_cifar as training
    from trainer_timm import TrainerCiFarTimmStyle
    assert Path(training.__file__).resolve().parent == ROOT
    out = Path(o.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    argv = json.loads(Path(o.ft_args).read_text())
    if isinstance(argv, dict):
        argv = argv['argv']
        if argv[0] == 'train_ode_cifar.py':
            argv = argv[1:]

    def option(key, value):
        if key in argv:
            argv[argv.index(key) + 1] = str(value)
        else:
            argv.extend((key, str(value)))

    for key, value in {
        '--batch_size': o.batch_size, '--num_workers': 0,
        '--ode_block': 'S2DTCPulseFFFB', '--s2_dtc_ordering_pairs': 0,
        '--s2_dtc_substeps': o.substeps, '--method': 'rk2',
        '--s2_dtc_slice_policy': o.slice_policy,
        '--s2_dtc_keep_slices': o.keep_slices,
        '--s2_dtc_random_k': o.random_k,
        '--s2_dtc_slice_sampling': 'per_cycle',
        '--s2_dtc_branch_sampling': o.branch_sampling,
        '--grad_accum_steps': o.grad_accum_steps,
        '--checkpoint_ode_rhs_training': 'true',
        '--checkpoint_ode_rhs_portion': o.checkpoint_portion,
        '--fuse_toggle_nonlinear_R_input': 'false',
        '--output_save_path': out / 'initialization', '--mem_frac': o.memory_fraction,
    }.items():
        option(key, value)
    if o.source_selection is not None:
        option('--ckpt', o.source_selection)
    if o.checkpoint_root:
        checkpoint_root = Path(o.checkpoint_root).resolve()
        matches = list(checkpoint_root.glob('*/*_full_param_last_ckpt.pth'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one latent last checkpoint in {checkpoint_root}')
        option('--save_path', checkpoint_root)
        option('--model_name', matches[0].parent.name)
        option('--ckpt', 'full_param_last')
    if o.stable_timing:
        random.seed(31337)
        np.random.seed(31337)
        torch.manual_seed(31337)
        torch.cuda.manual_seed_all(31337)
    sys.argv = ['train_ode_cifar.py', *argv]
    write(out / 'command.json', dict(source_root=str(ROOT), argv=argv, options=vars(o)))
    training.evaluate_teacher = lambda *args, **kw: None

    def run(trainer):
        model = trainer.model
        blocks = list(model.PcConvs)
        assert len(blocks) == 16
        assert trainer.grad_accum_steps == o.grad_accum_steps
        assert all(type(b).__name__ == 'S2DTCPulseFFFB'
                   and b.method == 'rk2' and b.s2_dtc_substeps == o.substeps
                   and b.s2_dtc_ordering_pairs == 0 for b in blocks)
        assert all(b.s2_dtc_slice_policy == o.slice_policy and
                   b.s2_dtc_branch_sampling == o.branch_sampling and
                   b.s2_dtc_slice_sampling == 'per_cycle' and
                   b.s2_dtc_random_k == o.random_k for b in blocks)
        cache = Path(o.batch_cache).resolve()
        if cache.exists():
            batch = torch.load(cache, map_location='cpu', weights_only=False)
        else:
            batch = next(iter(trainer.train_dataloader))
            batch = tuple(v.detach().cpu() if torch.is_tensor(v) else v for v in batch)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(batch, cache)
        assert batch[0].shape == (o.batch_size, 4, 16, 16)
        trainer.train_dataloader = [batch] * o.grad_accum_steps
        metadata = dict(source_root=str(ROOT), gpu=torch.cuda.get_device_name(),
            batch_size=o.batch_size, batch_hash=digest(batch),
            initial_model_hash=digest(model.state_dict()),
            initial_auxiliary_hash=digest(trainer._feature_kd_loss.state_dict()),
            checkpointed_layers=sum(bool(b.checkpoint_ode_rhs_training) for b in blocks),
            checkpoint_portion=o.checkpoint_portion,
            grad_accum_steps=o.grad_accum_steps,
            method='rk2', substeps=o.substeps, ordering_pairs=0,
            slice_policy=o.slice_policy, keep_slices=o.keep_slices,
            random_k=o.random_k, cycle_sampling='per_cycle', branch_sampling=o.branch_sampling,
            model_layers=16, fuse_nonlinear_R_input=False,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            memory_fraction=o.memory_fraction,
            timing_scope='One accumulation window: actual teacher, student, SRRL loss, '
                         'backward and optimizer; cached batch, no loading/saving.')
        random.seed(31337)
        np.random.seed(31337)
        torch.manual_seed(31337)
        torch.cuda.manual_seed_all(31337)
        if o.phase != 'benchmark':
            if o.phase == 'parity':
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.use_deterministic_algorithms(True)
            metadata.update(precision='strict' if o.phase == 'parity' else 'production',
                            cudnn_benchmark=torch.backends.cudnn.benchmark,
                            cudnn_deterministic=torch.backends.cudnn.deterministic,
                            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                            optimizer_updates=False)
            captured = {}
            original = trainer._student_forward_feature_kd

            def forward(inputs):
                inputs = inputs.detach().requires_grad_(True)
                inputs.register_hook(lambda g: captured.update(input_gradient=g.detach().cpu().clone()))
                logits, features = original(inputs)
                captured['logits'] = logits.detach().cpu().clone()
                captured['features'] = [feature.detach().cpu().clone() for feature in features]
                return logits, features

            trainer._student_forward_feature_kd = forward
            trainer.optimizer.step = lambda *args, **kw: None
            loss = trainer.train_one_epoch(0)
            captured['model_gradients'] = {n: None if v.grad is None else v.grad.detach().cpu().clone()
                                           for n, v in model.named_parameters()}
            captured['auxiliary_gradients'] = {n: None if v.grad is None else v.grad.detach().cpu().clone()
                                               for n, v in trainer._feature_kd_loss.named_parameters()}
            metadata.update(loss=float(loss), assignments_hash=digest(hardware(model)),
                end_rng_hash=digest((random.getstate(), np.random.get_state(), torch.get_rng_state(),
                                     torch.cuda.get_rng_state_all())),
                module_rng_hash=digest({n: {k: {str(i): g.get_state() for i, g in v.items()}
                    for k, v in vars(m).items() if k.endswith('_generators') and isinstance(v, dict)}
                    for n, m in model.named_modules()}))
            torch.save(captured, out / 'outputs_gradients.pt')
            write(out / 'result.json', metadata)
            print('S2_PULSE_FT_PARITY_SAVED', out, flush=True)
            return

        rows = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stable = False
        limit = o.max_measured_steps if o.stable_timing else o.steps
        for step in range(o.warmups + limit):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = trainer.train_one_epoch(0)
            torch.cuda.synchronize()
            rows.append(dict(step=step, warmup=step < o.warmups,
                             seconds=time.perf_counter() - start, loss=float(loss)))
            print('S2_PULSE_FT_STEP', rows[-1], flush=True)
            write(out / 'progress.json', dict(metadata=metadata, steps=rows))
            measured_now = [r['seconds'] for r in rows if not r['warmup']]
            if o.stable_timing and len(measured_now) >= max(8, o.steps):
                recent = measured_now[-8:]
                median_shift = abs(statistics.median(recent[:4]) / statistics.median(recent[4:]) - 1)
                relative_spread = statistics.pstdev(recent) / statistics.mean(recent)
                stable = median_shift <= .05 and relative_spread <= .10
                if stable:
                    break
        measured = [r['seconds'] for r in rows if not r['warmup']]
        timing_window = measured[-8:] if o.stable_timing else measured
        metadata.update(steps=rows, median_seconds=statistics.median(timing_window),
            stable=stable if o.stable_timing else None,
            stability_definition='Last eight measured steps: adjacent four-step median shift <=5%; CV <=10%.',
            measurements=len(measured), reported_window=len(timing_window),
            median_microbatch_seconds=statistics.median(measured) / o.grad_accum_steps,
            min_seconds=min(measured), max_seconds=max(measured),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out / 'result.json', metadata)
        print('S2_PULSE_FT_BENCHMARK_SAVED', out, flush=True)

    TrainerCiFarTimmStyle.train = run
    training.main()


def s2_main():
    p = argparse.ArgumentParser()
    p.add_argument('--s2-dtc', action='store_true')
    p.add_argument('--reset-z-per-cycle', action='store_true')
    p.add_argument('--reset-study', action='store_true',
                   help='Reproduce the three pinned toy-model tables with cycle-boundary resets.')
    p.add_argument('--ordering-study', action='store_true', help='Dense RK2 ordering correction study only.')
    p.add_argument('--selective-reference', help='Reuse existing numerical output directory; run only selective Dopri5 cases.')
    p.add_argument('--selective-timing', action='store_true',
                   help='Benchmark reference and selective plain eval forwards together, with warmup and alternating order.')
    p.add_argument('--selective-timing-rk2', action='store_true',
                   help='With selective timing, compare slice-wise Dopri5, pulse RK2 N=8 and selective slices 1-4.')
    p.add_argument('--ordering-pairs', type=int, choices=(0, 1, 2, 4, 7), default=0)
    p.add_argument('--s2-worker', choices=('numerical', 'benchmark'))
    p.add_argument('--output', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--dynamics', choices=('average', 'pulse'), default='average')
    p.add_argument('--scene', choices=('ideal', 'clipped', 'measured', 'nonlinear'), default='ideal')
    p.add_argument('--representation', choices=('dense', 'expanded'), default='dense')
    p.add_argument('--method', default='rk2')
    p.add_argument('--substeps', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--start-worker', type=int, default=0,
                   help='Resume completed diagnostic cases without rerunning them.')
    p.add_argument('--train-step', action='store_true')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--report-only', action='store_true')
    o = p.parse_args()
    if o.selective_timing and not o.selective_reference:
        p.error('--selective-timing requires --selective-reference')
    if o.selective_timing_rk2 and not o.selective_timing:
        p.error('--selective-timing-rk2 requires --selective-timing')
    if o.selective_reference and (o.reset_study or o.ordering_study or o.report_only):
        p.error('Selective reference reuse cannot be combined with another study/report mode.')
    if o.selective_reference and o.s2_worker not in (None, 'numerical'):
        p.error('Selective reference reuse requires a numerical pulse worker.')
    if o.s2_worker:
        return s2_worker(o)
    out = Path(o.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if o.report_only:
        if o.reset_study:
            return s2_reset_report(out)
        return s2_ordering_report(out) if o.ordering_study else s2_report(out)
    commands = []
    base = [sys.executable, '-u', str(Path(__file__).resolve()), '--s2-dtc',
            '--output', str(out), '--batch', str(Path(o.batch).resolve())]
    if o.reset_z_per_cycle:
        base.append('--reset-z-per-cycle')
    if o.reset_study:
        if not o.reset_z_per_cycle:
            p.error('--reset-study requires --reset-z-per-cycle')
        base.append('--reset-study')
    for rep in ('dense', 'expanded'):
        for scene in ('ideal', 'clipped', 'measured', 'nonlinear'):
            for dynamics in ('average', 'pulse'):
                common = ['--representation', rep, '--scene', scene, '--dynamics', dynamics]
                commands.append(base + ['--s2-worker', 'numerical'] + common)
                n = '1' if dynamics == 'average' else '2'
                commands.append(base + ['--s2-worker', 'benchmark', '--substeps', n] + common)
                if rep == 'dense':
                    for checkpoint in (False, True):
                        commands.append(base + ['--s2-worker', 'benchmark', '--substeps', n,
                            '--train-step'] + (['--checkpoint'] if checkpoint else []) + common)
            commands.append(base + ['--s2-worker', 'benchmark', '--method', 'dopri5',
                '--representation', rep, '--scene', scene, '--dynamics', 'pulse'])
    # Finish all correctness/numerical comparisons before memory-intensive benchmarks.
    if o.reset_study:
        commands = []
        for rep in ('dense', 'expanded'):
            for scene in ('ideal', 'clipped', 'measured', 'nonlinear'):
                common = ['--representation', rep, '--scene', scene]
                for dynamics in ('average', 'pulse'):
                    commands.append(base + ['--s2-worker', 'numerical', '--dynamics', dynamics] + common)
                variants = [('average', 1)]
                if scene != 'measured':
                    variants += [('pulse', n) for n in (1, 2, 4, 8)]
                for dynamics, n in variants:
                    for flags in ([[], ['--train-step'], ['--train-step', '--checkpoint']]
                                  if rep == 'dense' else [[]]):
                        commands.append(base + ['--s2-worker', 'benchmark', '--dynamics', dynamics,
                                                '--substeps', str(n)] + common + flags)
    if o.ordering_study:
        commands = []
        for scene in ('ideal', 'clipped', 'measured', 'nonlinear'):
            for dynamics, pairs in [('pulse', 0)] + [('average', r) for r in (0, 1, 2, 4, 7)]:
                common = ['--ordering-study', '--representation', 'dense', '--scene', scene,
                          '--dynamics', dynamics, '--ordering-pairs', str(pairs)]
                commands.append(base + ['--s2-worker', 'numerical'] + common)
                for flags in ([], ['--train-step'], ['--train-step', '--checkpoint']):
                    commands.append(base + ['--s2-worker', 'benchmark', '--substeps', '1'] + common + flags)
    if o.selective_reference:
        # Reuse exactly the selected scene/representation, never launch the
        # ordinary sweep or regenerate its Dopri5 references/timing benchmarks.
        command = base + ['--s2-worker', 'numerical', '--dynamics', 'pulse',
            '--scene', o.scene, '--representation', o.representation,
            '--selective-reference', str(Path(o.selective_reference).resolve())]
        if o.selective_timing:
            command.append('--selective-timing')
        if o.selective_timing_rk2:
            command.append('--selective-timing-rk2')
        commands = [command]
    commands.sort(key=lambda c: c[c.index('--s2-worker')+1] != 'numerical')
    write(out/'commands.json', commands)
    for i, command in enumerate(commands):
        if i < o.start_worker:
            continue
        print('START', i, command, flush=True)
        benchmark = command[command.index('--s2-worker')+1] == 'benchmark'
        attempts = []
        for bs in ((128, 64, 32, 16, 8, 4, 2, 1) if benchmark else (128,)):
            current = command + ['--batch-size', str(bs)]
            log_path = out/f'worker_{i:03d}_bs{bs}.log'
            with log_path.open('w') as log:
                result = subprocess.run(current, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Worker {i} failed; see {log_path}')
            if not benchmark:
                break
            def value(key, default):
                return command[command.index(key)+1] if key in command else default
            pairs = int(value('--ordering-pairs', '0'))
            suffix = f'_pairs{pairs}' if pairs else ''
            tag = (f"{value('--representation','dense')}_{value('--scene','ideal')}_"
                   f"{value('--dynamics','average')}{suffix}_{value('--method','rk2')}_"
                   f"{value('--substeps','1')}_train{'--train-step' in command}_"
                   f"checkpoint{'--checkpoint' in command}")
            path = out/('benchmark_'+tag+'.json')
            measured = json.loads(path.read_text())
            attempts.extend(measured.get('attempts', []))
            measured['attempts'] = attempts
            write(path, measured)
            if measured['status'] == 'ok':
                break
        print('FINISHED', i, flush=True)
    if o.selective_reference:
        print('SELECTIVE_COMPLETE', out, flush=True)
    elif o.reset_study:
        s2_reset_report(out)
    elif o.ordering_study:
        s2_ordering_report(out)
    else:
        s2_report(out)


if __name__=='__main__':
    if '--s2-slice-study' in sys.argv:
        s2_slice_study()
    elif '--s2-pulse-ft-cache' in sys.argv:
        s2_pulse_ft_cache_diagnostic()
    elif '--s2-pretrained' in sys.argv:
        s2_pretrained_diagnostic()
    elif '--s2-dtc' in sys.argv:
        s2_main()
    else:
        main()
