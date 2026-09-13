"""Bounded real TC fine-tuning benchmark, using the production entry/trainer.

Uses the real train_one_epoch implementation, data loader, teacher, restored
SRRL module, losses and optimizer. Stops after a small number of batches;
does not evaluate/save model or recovery checkpoints. Updates are process-local.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import train_ode_cifar as entry
from trainer_timm import TrainerCiFarTimmStyle

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--batches',type=int,default=13)
p.add_argument('--warmup',type=int,default=3)
p.add_argument('--distill-alpha',type=float,default=None)
p.add_argument('--distill-temperature',type=float,default=None)
a=p.parse_args()
assert a.batches>a.warmup>=0
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,TC_DRY_RUN='true',OUTPUT_DIR=str(out/'scratch_output'))
command=shlex.split(subprocess.check_output(['bash','launch_scripts/run_tc_nonidealities.sh','ft'],
                                           cwd=ROOT,env=env,text=True))
argv=command[command.index('train_ode_cifar.py')+1:]
for flag,value in (('--distill_alpha',a.distill_alpha),('--distill_temperature',a.distill_temperature)):
    if value is not None:
        argv += [flag,str(value)]
        command += [flag,str(value)]
(out/'resolved_command.txt').write_text(shlex.join(command)+'\n')
report=dict(command=command,rows=[],warmup_batches=a.warmup,
            scope='Actual train_one_epoch: data/H2D, student, teacher, CE/KD/SRRL, backward, SGD. No checkpoint saves.',
            excluded='Model/teacher/data initialization, standalone teacher validation, epoch-end validation/checkpoint serialization.')
def save(): (out/'timing.json').write_text(json.dumps(report,indent=2,default=str))

class TimedLoader:
    def __init__(self,loader):self.loader=loader
    def __len__(self):return a.batches
    def __iter__(self):
        iterator=iter(self.loader)
        for i in range(a.batches):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            data=next(iterator)
            fetched=time.perf_counter()
            yield data
            torch.cuda.synchronize()
            row=dict(batch=i,warmup=i<a.warmup,seconds=time.perf_counter()-start,
                     data_wait_seconds=fetched-start,peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved())
            report['rows'].append(row);save()
            print('FT_BENCH '+json.dumps(row),flush=True)

def train_benchmark(self):
    report.update(options=self.recovery_config,trainer=type(self).__name__,
                  teacher=type(self.teacher_model).__name__,batch_size=self.batch_size,
                  optimizer=type(self.optimizer).__name__,train_batches_per_epoch=len(self.train_dataloader),
                  feature_kd_initialized=self._feature_kd_loss is not None)
    report['optimizer_groups']=[{k:v for k,v in g.items() if k!='params'} for g in self.optimizer.param_groups]
    assert self._feature_kd_loss is not None and self.teacher_model is not None
    original_loader=self.train_dataloader
    initial=next(self.model.parameters()).detach().clone()
    self.train_dataloader=TimedLoader(original_loader)
    try:
        report['average_loss']=float(self.train_one_epoch(0))
    finally:
        self.train_dataloader=original_loader
        save()
    report['first_parameter_changed']=not torch.equal(initial,next(self.model.parameters()).detach())
    report['finite_parameters']=all(bool(p.isfinite().all()) for p in self.model.parameters())
    rows=[r for r in report['rows'] if not r['warmup']]
    report['median_seconds']=statistics.median(r['seconds'] for r in rows)
    report['mean_seconds']=statistics.mean(r['seconds'] for r in rows)
    report['median_peak_allocated']=statistics.median(r['peak_allocated'] for r in rows)
    report['completed']=True;save()
    print('FT_BENCH_COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ('command','rows','options','optimizer_groups')}),flush=True)

# Instrument only this diagnostic process. Production files remain unchanged.
TrainerCiFarTimmStyle.train=train_benchmark
entry.evaluate_teacher=lambda *args,**kwargs: print('Benchmark: standalone teacher validation skipped.',flush=True)
sys.argv=['train_ode_cifar.py',*argv]
entry.main()
