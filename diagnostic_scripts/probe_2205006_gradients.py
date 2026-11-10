"""Bounded no-update FT gradient comparison using job 2205006's PT weights.

Use --capture to obtain the real archived worker's FT command. --probe executes
one actual teacher/student backward per PVT corner on one fixed training batch.
No optimizer updates, production checkpoints or expanded caches are written.
"""
import argparse
import copy
import json
import os
import re
import statistics
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/PCN_C10_2205006_audit'
TAG = 'C10_C96L16_cc_randSpin_fullRange_1022ReLU_vdd0p5_analog_finalOnly_warmPT5'
SOURCE = ROOT/'saved_ckpt_runs'/f'{TAG}_iq12_toggle_odexinit'


def summarize():
    sys.path.insert(0,str(ROOT))
    import torch
    from ode_pc import _symmetric_qat_weight_scale, QuantizationImpl
    lines=(ROOT/'logs/slurm_jobs/slurm_2205006.out').read_bytes().decode(errors='replace').split('\n')
    rows=[];epoch=None;phase=None;health=[];evidence=[]
    for number,line in enumerate(lines,1):
        m=re.search(r'Training epoch (\d+) / (300|140)',line)
        if m:epoch=int(m[1]);phase='PT' if m[2]=='300' else 'FT'
        if '\r' in line:
            matches=re.findall(r'Iter=(\d+)/390, Loss=([\d.e+-]+), CE=([\d.e+-]+), SRRL=([\d.e+-]+)',line)
            if matches:
                i,loss,ce,kd=matches[-1]
                rows.append(dict(phase=phase,epoch_index=epoch,line=number,
                    running_epoch_mean_loss=float(loss),last_batch_ce=float(ce),last_batch_srrl=float(kd)))
        elif line.startswith('Training health check'):
            health.append(dict(line=number,text=line))
        if '\r' not in line and (line.startswith(('WARNING:root:', '-----', '[EVAL', 'Training health'))):
            evidence.append(f'{number}: {line}')
    (OUT/'training_evidence.txt').write_text('\n'.join(evidence)+'\n')
    pt=next(SOURCE.glob('TIMMPCNet*/*_last_ckpt.pth'))
    ft=next(SOURCE.glob('TIMMQAT*/*_full_param_last_ckpt.pth'))
    a=torch.load(pt,map_location='cpu',weights_only=False);b=torch.load(ft,map_location='cpu',weights_only=False)
    weights=[]
    for i in range(16):
        for branch in ('FFconv','FBconv'):
            key=f'PcConvs.{i}.{branch}'
            before=a['net'][key+'.weight'];after=b['net'][key+'.parametrizations.weight.original']
            w=after.detach().clone().requires_grad_();scale=_symmetric_qat_weight_scale(w,1)
            baked=QuantizationImpl.apply(w,scale,w.new_tensor(-15),w.new_tensor(15));baked.sum().backward()
            weights.append(dict(layer=i,branch=branch,pt_absmax=float(before.abs().max()),
                ft_master_absmax=float(after.abs().max()),scale=float(scale),
                gradient_blocked_percent=float((w.grad==0).float().mean()*100),
                quantized_at_rail_percent=float((baked.detach().abs()==1).float().mean()*100)))
    document=dict(remote_commit='20cdb14ee38866863e18b13e8ae2c7004e8c8977',
        pt_checkpoint=str(pt),ft_checkpoint=str(ft),pt_accuracy=a['acc'],ft_accuracy=b['acc'],
        epoch_rows=rows,health_checks=health,weight_stats=weights)
    probes={}
    for family in ('0906','1022'):
        p=OUT/f'gradients_{family}_RELU_Voltage.json'
        if p.exists():
            d=json.loads(p.read_text())
            probes[family]=dict(corners=len(d),
                gradient_norm_median=statistics.median(r['global_gradient_norm'] for r in d),
                gradient_norm_max=max(r['global_gradient_norm'] for r in d),
                training_batch_accuracy_range_percent=[min(r['training_batch_accuracy_percent'] for r in d),
                                                        max(r['training_batch_accuracy_percent'] for r in d)])
    document['frozen_pretrained_gradient_probe']=dict(
        description='Identical 1022-pretrained source and fixed training batch, one backward per PVT corner, no optimizer updates. Not an epoch-24 replay.',
        results=probes)
    (OUT/'collapse_evidence.json').write_text(json.dumps(document,indent=2)+'\n')
    print('Saved',OUT/'collapse_evidence.json')


def capture():
    env = dict(os.environ)
    env.update(TOGGLE_MODE='odexinit', TASK='cifar10', IMG_TYPE='CiFAIR',
        EXP_PREFIX=TAG, TEACHER_CKPT=str(ROOT/'checkpoint/efficientnet_v2_l_cifar10_CiFAIR_OldNoTimm_MatchDistill.pth'),
        ADAPT_PIL_TEACHER='false', DISTILL_METHOD='srrl', TIMM_RE_PROB='0.0',
        NONLINEAR_R_TABLE='coupler_full_range', MC_RELU_MONTE_CARLO_SOURCE='1022_RELU_Voltage',
        R_VAL='50e3', C_VAL='500e-15', V_DD='0.5', TOGGLE_ONE_OVER_Q='5',
        TOGGLE_TIMING_MODE='fixed', TOGGLE_Y_TIME='10e-9', Z_OVER_Y_TIME='1',
        SCALE_TRAIN_RECIPE='1', INPUT_QUANT_BITS='12', CENTER_STUDENT_INPUT='false',
        ENABLE_MEASURED_POOLING='true', WEIGHT_QUANT_FACTOR_BITS='1', ENOB='none',
        UNITLESS_MEASURED_PULLBACK_MODE='direct', MEASURED_ACTIVATION_SCOPE='all',
        PRETRAIN_ACTIVATION_CORNER_MODE='fixed', ACTIVATION_CORNER_MODE='random_per_forward',
        ACTIVATION_RANDOM_CURVE_SHARING='per_spin', FT_CORNER_COUPLED_SAMPLING='true',
        SEARCH_ARCH='default', PCN_CHAN_0_LIST='24', PCN_NUM_LAYERS_LIST='16',
        FINAL_HEAD_TYPE='analog', FINAL_ADC_NOISE_LSB='0.5', LINEAR_BIAS='false',
        WARMUP_PRETRAIN='5', WARMUP_FT='0', FINAL_EVAL_ONLY='true',
        NUM_COMB_PER_NUM_LAYER='2', COMB_SEL_SET='2', TC_DRY_RUN='true')
    r = subprocess.run(['bash','-c','source '+str(OUT/'remote_launch_scripts/slurm_search_config.sh')],
                       cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    (OUT/'replay_scheduler_capture.log').write_text(r.stdout+r.stderr)
    command = next(line for line in r.stdout.splitlines() if line.startswith('sbatch --parsable'))
    decoded = subprocess.check_output(['bash','-c',
        'sbatch() { for arg; do [[ "$arg" != --export=* ]] || printf "%s" "${arg#--export=}"; done; };\n'+command],
        cwd=ROOT, env=env, text=True)
    for field in decoded.split(','):
        if '=' in field:
            k,v = field.split('=',1); env[k]=v
    env.update(PRETRAIN_SAVE_PATH=str(SOURCE), FT_OUTPUT_SAVE_PATH=str(OUT/'probe_initialization'),
               PROBE_PYTHON=sys.executable, PROBE_SCRIPT=str(Path(__file__).resolve()))
    header = (OUT/'remote_launch_scripts/run_kdcrd_then_ft.sbatch').read_text().split('# PHASE 1:')[0]
    header = header.replace('source activate base', ':').replace('conda activate scanbase', ':')
    header = header.replace('/scratch/rzeng7/repos/PCN-with-Local-Recurrent-Processing/logs/', str(OUT/'capture_logs')+'/')
    name = next(SOURCE.glob('TIMMPCNet*')).name
    env['PROBE_MODEL_NAME'] = name
    tail = '''
IFS=$'\\t' read -r comb_tag params inp_str out_str pool_str <<< "${COMBS[0]}"
python() { "$PROBE_PYTHON" "$PROBE_SCRIPT" --capture-argv "$@"; }
finetune_one_combo_model "$comb_tag" "$PROBE_MODEL_NAME" "$FT_ODE_BLOCK"
'''
    r = subprocess.run(['bash','-c',header+tail], cwd=ROOT, env=env,
                       capture_output=True, text=True, check=True)
    (OUT/'replay_worker_capture.log').write_text(r.stdout+r.stderr)


def probe(options):
    # Prefer the exact remote commit, not newer local TC changes.
    code = Path(options.code_root).resolve()
    sys.path.insert(0, str(code)); os.chdir(code)
    import random
    import numpy as np
    import torch
    import train_ode_cifar as training
    from trainer_timm import TrainerCiFarTimmStyle
    from training_recovery import _runtime
    argv = json.loads((OUT/'ft_command.json').read_text())
    sys.argv = ['train_ode_cifar.py', *argv]
    args = training.get_args()
    args.save_path = str(SOURCE)
    args.teacher_ckpt = str(ROOT/'checkpoint/efficientnet_v2_l_cifar10_CiFAIR_OldNoTimm_MatchDistill.pth')
    args.output_save_path = str(OUT/f'gradient_{options.relu}_initialization')
    args.activation_curve_path = str(code/'hardware_data/mc_45_corners'/options.relu)
    args.nonlinear_R_table = str(code/'hardware_data/mc_45_corners/coupler_full_range')
    args.num_workers = 0
    training.get_args = lambda: args
    training.evaluate_teacher = lambda *unused: None
    training._configure_training_recovery = lambda *unused: None

    def run(trainer):
        from utils import mc45_corner_ids
        from pc_model import logits_for_loss
        file = OUT/'fixed_training_batch.pt'
        if file.exists():
            batch = torch.load(file, map_location='cpu', weights_only=False)
        else:
            batch = next(iter(trainer.train_dataloader)); torch.save(batch,file)
        trainer.train_dataloader = [batch]
        initial = copy.deepcopy(trainer.model.state_dict())
        aux = copy.deepcopy(trainer._feature_kd_loss.state_dict())
        runtime = copy.deepcopy(_runtime(trainer.model))
        rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        sampler = trainer.model._ft_corner_coupled_sampler
        original = trainer._student_forward_feature_kd
        captured = {}
        def forward(inputs):
            outputs, features = original(inputs)
            captured['training_batch_accuracy_percent'] = float(
                outputs.argmax(1).cpu().eq(batch[-1]).float().mean()*100)
            captured['logits_absmax'] = float(logits_for_loss(outputs,trainer.model).detach().abs().max())
            return outputs,features
        trainer._student_forward_feature_kd = forward
        def record(*unused):
            rows = []
            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        rows.append((p, float(group['lr'])))
            captured['global_gradient_norm'] = float(torch.stack([p.grad.detach().norm() for p,lr in rows]).norm())
            captured['first_layers'] = {}
            for name,p in trainer.model.named_parameters():
                if p.grad is not None and name.startswith(('PcConvs.0.','PcConvs.1.')):
                    lr = next(lr for param,lr in rows if param is p)
                    captured['first_layers'][name] = dict(
                        gradient_absmax=float(p.grad.detach().abs().max()),
                        parameter_absmax=float(p.detach().abs().max()),
                        lr=lr, first_step_update_absmax=float(p.grad.detach().abs().max())*lr)
        trainer.optimizer.step = record
        records = []
        for corner in mc45_corner_ids():
            trainer.model.load_state_dict(initial); trainer._feature_kd_loss.load_state_dict(aux)
            for name, attributes in copy.deepcopy(runtime).items():
                module=trainer.model.get_submodule(name)
                for k,v in attributes.items():setattr(module,k,v)
            sampler.clear()
            def sample(corner=corner):
                sampler.active_corner=corner
                trainer.model._last_ft_corner_coupled_corner=corner
                return corner
            sampler.sample=sample
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            captured={}
            loss=trainer.train_one_epoch(0)
            row=dict(corner=corner,loss=float(loss),**captured);records.append(row)
            (OUT/f'gradients_{options.relu}.json').write_text(json.dumps(records,indent=2)+'\n')
            print('PROBE',corner,'loss',loss,'gradient_norm',captured['global_gradient_norm'],flush=True)
    TrainerCiFarTimmStyle.train=run
    training.main()


if __name__ == '__main__':
    if '--capture-argv' in sys.argv:
        argv=sys.argv[sys.argv.index('--capture-argv')+1:]
        if argv[0]=='-u':argv=argv[1:]
        assert argv[0]=='train_ode_cifar.py'
        (OUT/'ft_command.json').write_text(json.dumps(argv[1:],indent=2)+'\n')
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('--capture',action='store_true')
        p.add_argument('--summarize',action='store_true')
        p.add_argument('--code-root')
        p.add_argument('--relu',choices=('1022_RELU_Voltage','0906_RELU_Voltage'))
        options=p.parse_args()
        if options.summarize:summarize()
        elif options.capture:capture()
        else:probe(options)
