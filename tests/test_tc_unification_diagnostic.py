"""Report unfinished checkpoint searches honestly; never certify stale files."""
import pytest
from diagnostic_scripts.benchmark_tc_unification import performance_table


def row(count,status='ok'):
    return dict(state=1,mode='empirical_with_replacement',n=3,variant='on',
        checkpoint_layers=count,status=status,median_seconds=2.,
        peak_allocated_including_warmup_bytes=5*2**30,
        weight_coverage={'total':{'individually_covered_weights':1234}})


def test_only_full_checkpoint_result_is_not_a_proven_minimum(tmp_path):
    performance_table(tmp_path,[row(22)])
    result=(tmp_path/'performance.md').read_text()
    assert 'searching' in result
    assert '| 1234 | searching | — | — | 2.000 | 5.000 |' in result


def test_minimum_requires_all_lower_counts_to_be_tested(tmp_path):
    performance_table(tmp_path,[row(22),row(0,'OOM'),row(1,'OOM'),row(2)])
    assert '| 1234 | 2 | 2.000 | 5.000 |' in (tmp_path/'performance.md').read_text()
    performance_table(tmp_path,[row(22),row(0,'OOM'),row(2)])
    assert 'searching' in (tmp_path/'performance.md').read_text()


def test_all_counts_failing_is_not_reported_as_fitting(tmp_path):
    performance_table(tmp_path,[row(k,'OOM') for k in range(23)])
    result=(tmp_path/'performance.md').read_text()
    assert 'none fits' in result
    assert '| OOM | OOM |' in result


def test_existing_output_directory_is_rejected_before_gpu_check(tmp_path,monkeypatch):
    import sys
    from diagnostic_scripts import benchmark_tc_unification as diagnostic
    monkeypatch.setattr(sys,'argv',['benchmark_tc_unification.py','--phase','memory',
                                   '--output',str(tmp_path)])
    with pytest.raises(FileExistsError):diagnostic.main()


@pytest.mark.parametrize('precision_flags', [False, True])
def test_top_n_validation_flags_reach_worker(tmp_path, monkeypatch, precision_flags):
    import json
    import sys
    from pathlib import Path
    from diagnostic_scripts import benchmark_tc_top_n as diagnostic
    root = tmp_path / 'saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study_student_teacher_no_norm'
    checkpoint = root / 'student/student_last_ckpt.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: 'env FLAG=1 python train.py')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index('--report')+1]).write_text(json.dumps({'passed': True}))
    monkeypatch.setattr(diagnostic.subprocess, 'run', run)
    flags = ['--production-precision', '--ordered-groups', '--audit-fp64-convolution']
    monkeypatch.setattr(sys, 'argv', ['benchmark_tc_top_n.py', '--verify',
        '--states', '1', '--ns', '0', '--modes', 'multivariate_gaussian',
        '--output', str(tmp_path / 'reports'), *(flags if precision_flags else [])])
    diagnostic.main()
    assert len(commands) == 1
    for flag in flags:
        assert (flag in commands[0]) == precision_flags


def test_rhs_counter_preserves_module_and_plain_function_protocols():
    import torch
    from diagnostic_scripts.benchmark_tc_empirical import count_rhs_calls
    class RHS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('gain',torch.tensor(2.))
        def forward(self,t,y):return self.gain*y
    module=RHS()
    module.tc_context=object()
    context=module.tc_context
    calls=[0]
    counted=count_rhs_calls(module,calls)
    assert counted is module
    assert counted.tc_context is context
    assert list(counted.state_dict())==['gain']
    torch.testing.assert_close(counted(0.,torch.tensor(3.)),torch.tensor(6.),atol=0,rtol=0)
    counted.load_state_dict({'gain':torch.tensor(4.)})
    assert float(counted(0.,torch.tensor(3.)))==12.
    assert calls==[2]
    def function(t,y):return 2*y
    function.tc_context=context
    wrapped=count_rhs_calls(function,calls)
    assert wrapped.tc_context is context
    assert float(wrapped(0.,torch.tensor(3.)))==6.
    assert calls==[3]


@pytest.mark.parametrize('two',[False,True])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_rhs_counter_preserves_actual_tc_block_outputs_and_gradients(two,checkpoint):
    import torch
    from test_tc_dense_training import make_block,wrap
    from diagnostic_scripts.benchmark_tc_empirical import count_rhs_calls
    answers=[]
    calls=[0]
    for instrument in (False,True):
        block=make_block(two)
        block.option_aca['checkpoint_ode_rhs_training']=checkpoint
        model=wrap(block,nonlinear_R=False,enable_summing_current_noise=True,
                   enable_coupler_noise=True,summing_noise_seed=123,coupler_noise_seed=456)
        if instrument:
            original_make=block._make_ode_fn
            def counted_make(*a,**kw):return count_rhs_calls(original_make(*a,**kw),calls)
            block._make_ode_fn=counted_make
        x=torch.full((2,2,2,2),.02,requires_grad=True)
        y=model(x)
        gradients=torch.autograd.grad(y.square().sum(),(x,*model.parameters()),allow_unused=True)
        answers.append((y.detach(),*[g for g in gradients if g is not None]))
    assert calls[0]>0
    for actual,expected in zip(answers[1],answers[0]):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
