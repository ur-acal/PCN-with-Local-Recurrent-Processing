"""Two small real FT batches from recovery weights; no checkpoint writes."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import train_ode_cifar as entry
from trainer_timm import TrainerCiFarTimmStyle

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
a=p.parse_args(); path=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
original=torch.load
def load(p,*args,**kwargs):
    if isinstance(p,(str,Path)) and Path(p).resolve()==path:kwargs['map_location']='cpu'
    return original(p,*args,**kwargs)
torch.load=load
recovery=path.parent/(path.parent.name+'_latest_ckpt.pth')
d=torch.load(recovery,map_location='cpu',weights_only=False)
config=d['training_recovery']['config'].copy();del d
suffix=path.name[len(path.parent.name)+1:-len('_ckpt.pth')]
config.update(model_name=path.parent.name,save_path=str(path.parent.parent),ckpt=suffix,
    batch_size=4,num_workers=0,output_save_path=str(out.parent/'unused_ft_output'))
report=dict(checkpoint=str(path),scope='real FT two batches; optimizer updates process-local only',config=config,rows=[])
class SmallLoader:
    def __init__(self,loader):self.loader=loader
    def __len__(self):return 2
    def __iter__(self):
        it=iter(self.loader)
        for i in range(2):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            yield next(it)
            torch.cuda.synchronize()
            report['rows'].append(dict(batch=i,seconds=time.perf_counter()-t,peak=torch.cuda.max_memory_allocated()))
            out.write_text(json.dumps(report,indent=2,default=str));print(report['rows'][-1],flush=True)
def run(self):
    self.train_dataloader=SmallLoader(self.train_dataloader)
    self.train_one_epoch(0)
entry.get_args=lambda:Namespace(**config)
entry.evaluate_teacher=lambda *args,**kwargs:None
TrainerCiFarTimmStyle.train=run
entry.main()
