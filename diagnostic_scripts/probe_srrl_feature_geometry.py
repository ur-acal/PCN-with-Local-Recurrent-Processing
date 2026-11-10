"""Compare cached student/teacher feature geometry, without changing weights."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from study_relu_srrl_gap import PiecewiseLinearActivation

ROOT = Path(__file__).resolve().parents[1]


def geometry(x):
    x = x.double()
    centered = x-x.mean(0)
    eigen = torch.linalg.svdvals(centered).square()
    eigen = eigen[eigen > 1e-12]
    if eigen.numel() == 0:
        return dict(dimensions=x.shape[1], variance=0., participation_rank=0.,
            entropy_rank=0., leading_variance_fraction=None,
            negative_fraction=(x<0).double().mean().item())
    p = eigen/eigen.sum()
    return dict(dimensions=x.shape[1], variance=eigen.sum().item()/len(x),
        participation_rank=(1/p.square().sum()).item(),
        entropy_rank=torch.exp(-(p*p.log()).sum()).item(),
        leading_variance_fraction=p[0].item(),
        negative_fraction=(x<0).double().mean().item())


def main():
    torch.set_num_threads(4)
    report = {}
    for classes in [10,100]:
        folder = ROOT/'results/relu_srrl_gap'/f'cifar{classes}'
        teacher = torch.load(folder/'teacher.pt', weights_only=False)
        target = teacher['targets']['12']
        w, b = teacher['w'], teacher['b']
        row = {'teacher_feature':geometry(target['test']['pool']),
               'teacher_logits':geometry(F.linear(target['test']['pool'],w,b)),
               'student':{}}
        for family in ['0906','1022']:
            data = torch.load(folder/f'{family}_features.pt',weights_only=False)
            train = data['train_features'].mean((2,3)).double()
            test = data['test_features'].mean((2,3)).double()
            train_y = F.linear(target['train']['pool'],w,b).double()
            test_y = F.linear(target['test']['pool'],w,b).double()
            # Shared, explicit ridge grid. No test-set choice of a "best" fit.
            mean = train.mean(0); scale = train.std(0).clamp_min(1e-8)
            xx = (train-mean)/scale; zz = (test-mean)/scale
            ym = train_y.mean(0)
            gram = xx.T@xx/len(xx)
            rhs = xx.T@(train_y-ym)/len(xx)
            errors = {}
            for ridge in [.001,.01,.1,1.]:
                coef = torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0]),rhs)
                errors[str(ridge)] = F.mse_loss(zz@coef+ym,test_y).item()
            row['student'][family] = dict(geometry=geometry(test),
                linear_teacher_logit_probe_mse=errors)
        report[str(classes)] = row
    report['effective_curves'] = {}
    for family in ['0906','1022']:
        activation = PiecewiseLinearActivation(
            str(ROOT/'hardware_data/mc_45_corners'/f'{family}_RELU_Voltage'/'tt_25_1.csv'),
            v_dd=.5, corner='MC18', normalize_positive_endpoint=False,
            fuse_measured_activation=False)
        activation.set_coordinate_pullback_scale(.1)
        x = torch.tensor([-2.,-1.,-.5,-.1,0.,.1,1.,4.5,4.8,5.],requires_grad=True)
        y = activation(x)
        y.sum().backward()
        report['effective_curves'][family] = dict(unitless_input=x.detach().tolist(),
            unitless_output=y.detach().tolist(), derivative=x.grad.tolist(),
            corner='TT_V1_T1', curve='MC18', v_dd=.5, pullback_q=.1)
    path = ROOT/'results/relu_srrl_gap/geometry.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
