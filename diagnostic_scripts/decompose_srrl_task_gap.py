"""Decompose cached SRRL error into class-count, scale, and class directions."""
import json
import torch
import torch.nn.functional as F
from study_relu_srrl_gap import ROOT, SRRLLoss, checkpoint


def decompose(pred, target, labels):
    n,k = pred.shape
    err = pred-target
    square = err.square()
    top = target.argmax(1)
    top_sq = square.gather(1,top[:,None]).mean()
    true_sq = square.gather(1,labels[:,None]).mean()
    # Independent scalar per output, estimated on these same data: descriptive
    # diagnostic only, not a held-out fitted predictor.
    pc = pred-pred.mean(0)
    tc = target-target.mean(0)
    rho = (pc*tc).mean(0)/(pc.square().mean(0)*tc.square().mean(0)).sqrt().clamp_min(1e-12)
    metrics = dict(mse=square.mean().item(), sum_squared_error=square.sum(1).mean().item(),
        teacher_top_class_squared_error=top_sq.item(),
        teacher_top_class_mse_contribution=(top_sq/k).item(),
        other_class_mse_contribution=(square.mean()-top_sq/k).item(),
        true_class_squared_error=true_sq.item(),
        teacher_top_signed_error=err.gather(1,top[:,None]).mean().item(),
        teacher_top_logit=target.gather(1,top[:,None]).mean().item(),
        teacher_per_logit_energy=target.square().mean().item(),
        standardized_mse=(square/(target.var(0,unbiased=False)+1e-8)).mean().item(),
        average_logit_correlation=rho.mean().item(),
        prediction_teacher_argmax_agreement=pred.argmax(1).eq(top).float().mean().item(),
        prediction_accuracy=pred.argmax(1).eq(labels).float().mean().item(),
        output_error_mse=square.mean(0).tolist(),
        output_target_energy=target.square().mean(0).tolist())
    return metrics


def main():
    torch.set_num_threads(4)
    report = {}
    for c in [10,100]:
        folder = ROOT/'results/relu_srrl_gap'/f'cifar{c}'
        inputs = torch.load(folder/'inputs.pt',weights_only=False)
        tc = torch.load(folder/'teacher.pt',weights_only=False)
        row = {}; sample_errors={}; student_correct={}
        for family in ['ideal','0906','1022']:
            fc = torch.load(folder/f'{family}_features.pt',weights_only=False)
            bits = 'None' if family == 'ideal' else '12'
            targets = tc['targets'][bits]
            w,b = tc['w'],tc['b']
            state=torch.load(checkpoint(c,family),map_location='cpu',weights_only=False)['net']
            weight=next(v for k,v in state.items() if k.endswith('linear.weight'))
            biases=[v for k,v in state.items() if k.endswith('linear.bias')]
            own_logits=F.linear(fc['test_features'].mean((2,3)),weight,biases[0] if biases else None)
            student_correct[family]=own_logits.argmax(1).eq(inputs['test_labels'])
            for batch_bn in [False,True]:
                kd = SRRLLoss(96,1280)
                kd.load_state_dict(fc['kd']);kd.train(batch_bn)
                with torch.no_grad():
                    z = torch.cat([kd.connector(f).mean((2,3)) for f in fc['test_features'].split(128)])
                pred = F.linear(z,w,b)
                target = F.linear(targets['test']['pool'],w,b)
                record = decompose(pred,target,inputs['test_labels'])
                if batch_bn:sample_errors[family]=(pred-target).square().mean(1)
                residual=pred-target
                between=0.
                for label in inputs['test_labels'].unique():
                    selected=inputs['test_labels']==label
                    between+=residual[selected].mean(0).square().mean().item()*selected.float().mean().item()
                record['between_class_error_mse']=between
                record['within_class_error_mse']=record['mse']-between
                # Ridge: one fixed regularizer, same calibration size for all.
                # A refit to common teacher targets removes the stored connector.
                train = fc['train_features'].mean((2,3)).double()
                test = fc['test_features'].mean((2,3)).double()
                mu,sd=train.mean(0),train.std(0).clamp_min(1e-8)
                xx=(train-mu)/sd;zz=(test-mu)/sd
                train_y=F.linear(targets['train']['pool'],w,b).double()
                ym=train_y.mean(0)
                coef=torch.linalg.solve(xx.T@xx/len(xx)+.1*torch.eye(xx.shape[1]),
                    xx.T@(train_y-ym)/len(xx))
                record['ridge_heldout']=decompose(zz@coef+ym,target.double(),inputs['test_labels'])
                row[f'{family}_{"batch_bn" if batch_bn else "eval_bn"}']=record
        shared=student_correct['0906']&student_correct['1022']
        subsets={'both_students_correct':shared,
            '0906_correct_1022_wrong':student_correct['0906']&~student_correct['1022'],
            '1022_correct_0906_wrong':student_correct['1022']&~student_correct['0906'],
            'both_wrong':~student_correct['1022']&~student_correct['0906']}
        row['subset_comparison']={name:dict(n=int(mask.sum()),
            mse_0906=sample_errors['0906'][mask].mean().item(),
            mse_1022=sample_errors['1022'][mask].mean().item(),
            contribution_to_gap=(sample_errors['1022'][mask]-sample_errors['0906'][mask]).sum().item()/len(mask))
            for name,mask in subsets.items()}
        report[str(c)]=row
    path=ROOT/'results/relu_srrl_gap/task_decomposition.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for c,rows in report.items():
        for family in ['ideal','0906','1022']:
            r=rows[family+'_batch_bn']
            print(c,family,{k:round(r[k],5) for k in ['mse','sum_squared_error',
                'teacher_top_class_squared_error','teacher_top_class_mse_contribution',
                'other_class_mse_contribution','teacher_top_signed_error','standardized_mse',
                'average_logit_correlation','prediction_accuracy']})
            rr=r['ridge_heldout']
            print('RIDGE',{k:round(rr[k],5) for k in ['mse','standardized_mse',
                'teacher_top_class_squared_error','average_logit_correlation','prediction_accuracy']})


if __name__=='__main__':main()
