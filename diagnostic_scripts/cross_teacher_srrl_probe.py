"""Cross the two teachers with both datasets to diagnose target specificity.

Teachers on the other dataset supply regression targets, not accuracy baselines.
Frozen students; same ridge fitting protocol and held-out samples in all cells.
"""
import argparse
import json
import torch
import torch.nn.functional as F
from study_relu_srrl_gap import ROOT,teacher,teacher_cache


def ridge(train_x,test_x,train_y,test_y):
    train_x=train_x.double();test_x=test_x.double()
    train_y=train_y.double();test_y=test_y.double()
    mu,sd=train_x.mean(0),train_x.std(0).clamp_min(1e-8)
    x=(train_x-mu)/sd;z=(test_x-mu)/sd
    mean_y=train_y.mean(0)
    gram=x.T@x/len(x)
    coef=torch.linalg.solve(gram+.1*torch.eye(x.shape[1]),x.T@(train_y-mean_y)/len(x))
    pred=z@coef+mean_y
    return dict(mse=F.mse_loss(pred,test_y).item(),
        variance_normalized_mse=((pred-test_y).square().mean()/test_y.var(0,unbiased=False).mean()).item())


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher-classes',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    net=teacher(a.teacher_classes);w=net.classifier[-1].weight.detach().cpu();b=net.classifier[-1].bias.detach().cpu()
    report={}
    for dataset in [10,100]:
        folder=ROOT/'results/relu_srrl_gap'/f'cifar{dataset}'
        if dataset==a.teacher_classes:
            cache=torch.load(folder/'teacher.pt',weights_only=False)['targets']['12']
        else:
            data=torch.load(folder/'inputs.pt',weights_only=False)
            cache={part:teacher_cache(net,data[part+'_inputs'],12) for part in ['train','test']}
            torch.save(dict(targets=cache,w=w,b=b),folder/f'cross_teacher_{a.teacher_classes}.pt')
        row={}
        for family in ['ideal','0906','1022']:
            data=torch.load(folder/f'{family}_features.pt',weights_only=False)
            train=data['train_features'].mean((2,3));test=data['test_features'].mean((2,3))
            row[family]={kind:ridge(train,test,
                cache['train']['pool'] if kind=='feature' else F.linear(cache['train']['pool'],w,b),
                cache['test']['pool'] if kind=='feature' else F.linear(cache['test']['pool'],w,b))
                for kind in ['feature','logits']}
        report[str(dataset)]=row
        print('TEACHER',a.teacher_classes,'DATASET',dataset,row,flush=True)
    (ROOT/'results/relu_srrl_gap'/f'cross_teacher_{a.teacher_classes}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
