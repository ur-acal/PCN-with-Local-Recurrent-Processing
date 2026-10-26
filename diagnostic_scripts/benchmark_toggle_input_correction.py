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


if __name__=='__main__':main()
