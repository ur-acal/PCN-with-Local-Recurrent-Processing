"""Separate dropout-induced variance from systematic SRRL prediction error."""
import argparse
import hashlib
import json
import torch
import torch.nn.functional as F
from study_relu_srrl_gap import ROOT,seed,student,checkpoint,quantize_unit_interval


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--classes',type=int,required=True)
    p.add_argument('--repetitions',type=int,default=16)
    a=p.parse_args();folder=ROOT/'results/relu_srrl_gap'/f'cifar{a.classes}'
    cache=torch.load(folder/'matched_augmented.pt',weights_only=False)
    x=cache['inputs'];labels=cache['labels'];w=cache['w'].cuda();b=cache['b'].cuda()
    reports={}
    for family in ['ideal','0906','1022']:
        path=checkpoint(a.classes,family);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        bits=None if family=='ideal' else 12
        targets=F.linear(cache['targets'][str(bits)]['pool'].cuda(),w,b).cpu()
        for batch_bn in [True,False]:
            model,kd,_=student(a.classes,family);model.train();kd.train(batch_bn)
            predictions=[];ce=[]
            for rep in range(-1,a.repetitions):
                model.dropout=0. if rep==-1 else .25
                outputs=[];total_ce=0.
                for i in range(0,len(x),128):
                    seed(9000+max(rep,0)*10000+i)
                    f,logits=model(quantize_unit_interval(x[i:i+128].cuda(),bits),is_feat=True)
                    outputs.append(F.linear(kd.connector(f[-1]).mean((2,3)),w,b).cpu())
                    total_ce+=F.cross_entropy(logits,labels[i:i+128].cuda()).item()*len(logits)/len(x)
                predictions.append(torch.cat(outputs));ce.append(total_ce)
            without=predictions[0];stack=torch.stack(predictions[1:]);mean=stack.mean(0)
            row=dict(no_dropout_mse=F.mse_loss(without,targets).item(),
                dropout_expected_mse=(stack-targets).square().mean().item(),
                dropout_mean_prediction_mse=F.mse_loss(mean,targets).item(),
                dropout_prediction_variance=stack.var(0,unbiased=False).mean().item(),
                dropout_mean_shift_mse=F.mse_loss(mean,without).item(),
                no_dropout_ce=ce[0],dropout_expected_ce=sum(ce[1:])/a.repetitions,
                repetitions=a.repetitions,samples=len(x))
            assert abs(row['dropout_expected_mse']-row['dropout_mean_prediction_mse']-row['dropout_prediction_variance'])<1e-5
            key=family+('_batch_bn' if batch_bn else '_eval_bn')
            reports[key]=row
            torch.save(dict(without_dropout=without,with_dropout=stack,target=targets),folder/f'dropout_{key}.pt')
            print(a.classes,key,row,flush=True)
            (folder/'dropout_decomposition.json').write_text(json.dumps(reports,indent=2)+'\n')
            del model,kd;torch.cuda.empty_cache()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest


if __name__=='__main__':main()
