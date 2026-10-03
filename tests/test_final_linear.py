"""Equation-level and production-model tests for nonideal classifiers."""
import copy
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from final_linear import (AnalogLinear, DigitalLinear, add_final_head_args,
                          config_from_args, select_model_head, configure_feedforward_head)
from measured_activation import PiecewiseLinearActivation
from physical_feedforward import AveragedPhysicalBasicBlock, convert_wide_resnet_to_physical
from pc_model import logits_for_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_digital_rounding_rails_bias_and_overflow(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from final_linear import round_clip_ste
    # Ties go to the even integer; clipping gates the STE gradient.
    x = torch.tensor([-9., -2.5, -1.5, -.5, .5, 1.5, 2.5, 9.],
                     device=device, requires_grad=True)
    y = round_clip_ste(x, -4, 3)
    torch.testing.assert_close(y, x.new_tensor([-4, -2, -2, 0, 0, 2, 2, 3]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, x.new_tensor([0, 1, 1, 1, 1, 1, 1, 0]))
    head = DigitalLinear(1, 1, config=dict(final_repr_bits=3,
        final_weight_bits=3, final_bias_bits=4, final_accumulator_bits=4)).to(device)
    head.configure(physical=True, q=.1, v_dd=.5)
    with torch.no_grad():
        head.weight.fill_(1.)
        head.bias.zero_()
    # ADC [-4,3], weight code 3: unbounded accumulators [-12,9].
    v = torch.tensor([[-.5], [.5]], device=device)
    torch.testing.assert_close(head(v), v.double().new_tensor([[-8], [7]]))
    head.head_config['final_head_clamp'] = False
    torch.testing.assert_close(head(v), v.double().new_tensor([[-12], [9]]))
    with torch.no_grad():
        head.bias.fill_(.625)  # q*s_w*b/s_v = 1.5 -> nearest-even 2.
    torch.testing.assert_close(head(v * 0), v.double().new_full((2, 1), 2))
    with torch.no_grad():
        head.bias.fill_(100.)
    torch.testing.assert_close(head(v * 0), v.double().new_full((2, 1), 7))
    with torch.no_grad():
        head.bias.fill_(-100.)
    torch.testing.assert_close(head(v * 0), v.double().new_full((2, 1), -8))


def template():
    return AveragedPhysicalBasicBlock(nn.Conv2d(4, 3, 1, bias=False),
        R=1e4, C=49e-15, v_dd=.5, one_over_q=5, w_bits=5)


@pytest.mark.parametrize('kind', ['analog', 'digital'])
@pytest.mark.parametrize('family', ['toggle', 'tc'])
@pytest.mark.parametrize('timing', ['derived', 'fixed'])
@pytest.mark.parametrize('measured', [False, True])
@pytest.mark.parametrize('bias', [False, True])
def test_coordinates_gradients_and_update(kind, family, timing, measured, bias):
    torch.manual_seed(50)
    cls = AnalogLinear if kind == 'analog' else DigitalLinear
    head = cls(4, 3, bias=bias, config=dict(final_head_quantize=False, final_head_clamp=False))
    head.weight.data.mul_(.1)
    if bias:
        head.bias.data.mul_(.1)
    head.configure(physical=True, q=.1, v_dd=.5, template=template(), family=family,
                   R=1e4, C=49e-15, timing=timing, base_time=2e-10)
    W = head.weight.detach().clone().requires_grad_()
    b = None if not bias else head.bias.detach().clone().requires_grad_()
    x = torch.randn(2, 4, 2, 2, requires_grad=True) * .2
    if measured:
        act = PiecewiseLinearActivation(
            ROOT / 'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv',
            v_dd=.5, corner='MC18', normalize_positive_endpoint=False,
            fuse_measured_activation=False)
        h_v = act(.1*x)
        h_u = act(.1*x)/.1
    else:
        h_v, h_u = F.relu(.1*x), F.relu(x)
    gain = 2e-10/(1e4*49e-15) if kind == 'analog' and timing == 'fixed' else 1.
    expected = gain * F.linear(h_u.mean((2, 3)), W, b)
    actual = head.logits_for_loss(head(h_v.mean((2, 3))))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    labels = torch.tensor([0, 2])
    F.cross_entropy(actual, labels).backward(retain_graph=True)
    F.cross_entropy(expected, labels).backward()
    torch.testing.assert_close(head.weight.grad, W.grad, rtol=1e-5, atol=1e-6)
    if bias:
        torch.testing.assert_close(head.bias.grad, b.grad, rtol=1e-5, atol=1e-6)
    opt = torch.optim.SGD(head.parameters(), lr=.01, momentum=.9, weight_decay=.001)
    opt.step()
    torch.testing.assert_close(head.weight, W.detach()-.01*(W.grad+.001*W.detach()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('quantize', [False, True])
@pytest.mark.parametrize('clamp', [False, True])
@pytest.mark.parametrize('kind', ['analog', 'digital'])
def test_four_arithmetic_cases(kind, quantize, clamp):
    torch.manual_seed(6)
    head = (AnalogLinear if kind == 'analog' else DigitalLinear)(
        4, 3, config=dict(final_head_quantize=quantize, final_head_clamp=clamp))
    head.configure(physical=True, q=.1, v_dd=.5, template=template(), R=1e4, C=49e-15)
    x = torch.tensor([[.1, -.2, .3, -.4]])
    result = head.logits_for_loss(head(x))
    W, b = head.weight.detach(), head.bias.detach()
    if not quantize:
        expected = F.linear(x, W, .1*b)
    elif kind == 'analog':
        augmented = torch.cat([W, b[:, None]], 1)
        s = augmented.abs().max().reciprocal()
        encoded = (augmented*s*15).round().clamp(-15, 15)/15
        expected = F.linear(torch.cat([x, torch.tensor([[.1]])],1), encoded)/s
    else:
        sv = .5/128
        sw = 127/W.abs().max()
        a = (x.double()/sv).round().clamp(-128,127)
        K = (W.double()*sw).round().clamp(-127,127)
        bi = (.1*sw*b.double()/sv).round().clamp(-2**31,2**31-1)
        A = F.linear(a,K,bi)
        if clamp: A = A.clamp(-2**31,2**31-1)
        expected = A*sv/sw
    if kind == 'analog' and clamp:
        expected = expected.clamp(-.5,.5)
    torch.testing.assert_close(result, (expected/.1).float(), rtol=1e-5, atol=1e-6)
    result.sum().backward()
    assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize('kind', ['analog', 'digital'])
@pytest.mark.parametrize('family', ['toggle', 'tc1', 'tc2'])
def test_production_pcn_forward_and_checkpoint(tmp_path, kind, family):
    from test_toggle_physical_head import base_model, physical
    from test_tc_physical_head import build_tc
    from inference_utils import load_and_prepare_model
    from pc_conv import PCConvReLU6
    from pc_model import PCNetNoBatchNorm
    model = base_model(measured_activation_scope='all')
    select_model_head(model, kind, dict(final_repr_bits=6))
    if family == 'toggle':
        model, _ = physical(model, measured=True)
    else:
        model, _ = build_tc(int(family[-1]), model=model, scope='all', hardware=True)
    features, raw = model(torch.rand(2,3,4,4), is_feat=True)
    if kind == 'analog' and family.startswith('tc'):
        from physical_feedforward_tc import TCPhysicalBasicBlock
        circuit = model.linear._circuit
        assert isinstance(circuit, TCPhysicalBasicBlock)
        assert circuit.enable_spin_variation
        assert circuit.enable_coupler_noise
        assert circuit.enable_summing_current_noise
        assert circuit._tc_resistance_package is not None
    loss = F.cross_entropy(logits_for_loss(raw, model), torch.tensor([0,1]))
    loss.backward()
    assert torch.isfinite(model.linear.weight.grad).all()
    path = tmp_path/'model_full_param_best_ckpt.pth'
    torch.save(dict(net=model.state_dict(),init_args=model.init_args,net_type='PCNetNoBatchNorm'),path)
    loaded = load_and_prepare_model(str(path),'cpu',model_struct=PCNetNoBatchNorm,
                                   pc_conv_layer=PCConvReLU6,fuse_bn=False,noise_level=0)
    assert loaded.final_head_type == kind
    assert loaded.linear.head_config['final_repr_bits'] == 6
    torch.testing.assert_close(model.linear.bias,loaded.linear.bias)


@pytest.mark.parametrize('kind', ['analog', 'digital'])
@pytest.mark.parametrize('tc', [False, True])
def test_production_feedforward_forward(kind, tc):
    from baseline.cifar_resnet import WideResNetCIFAR
    args = SimpleNamespace(v_dd=.1,one_over_q=1,R=1e4,C=49e-15,w_bits=5,
                           weight_quant_factor_bits=None,tc_feedforward=tc,
                           toggle_timing_mode='derived',toggle_y_time=5e-9)
    model = WideResNetCIFAR(depth=10,widen_factor=1,num_classes=3,in_chans=3,
                           use_batchnorm=False,avgpool_downsample_shortcut=True)
    select_model_head(model,kind)
    convert_wide_resnet_to_physical(model,physical=True,qat=True,
        R=args.R,C=args.C,v_dd=args.v_dd,one_over_q=1,w_bits=5,
        tc_options=dict(one_shot_conv=True) if tc else None)
    configure_feedforward_head(model,args)
    features,raw=model(torch.rand(2,3,8,8),is_feat=True)
    logits_for_loss(raw,model).sum().backward()
    assert torch.isfinite(model.fc.weight.grad).all()


def test_cli_options_are_effective():
    parser=ArgumentParser()
    add_final_head_args(parser)
    args=parser.parse_args(['--final_head_type','digital','--final_repr_bits','6',
                           '--final_weight_bits','7','--final_adc_noise_lsb','0.5'])
    head=DigitalLinear(4,3,config=config_from_args(args))
    assert head.head_config['final_repr_bits']==6
    assert head.head_config['final_weight_bits']==7
    assert head.head_config['final_adc_noise_lsb']==.5


@pytest.mark.parametrize('family', ['toggle', 'tc'])
def test_analog_expanded_trials_and_dtype(family):
    head = AnalogLinear(4, 3, config=dict(final_head_clamp=False)).eval()
    head.configure(physical=True, q=.1, v_dd=.5, template=template(), family=family,
                   R=1e4, C=49e-15)
    x = torch.randn(2,4)*.03
    with torch.no_grad():
        dense = head(x)
        head.expanded = True
        expanded = head(x)
        torch.testing.assert_close(expanded, dense, atol=1e-6, rtol=1e-5)
        cached = head._expanded_module
        for _ in range(3):
            torch.testing.assert_close(head(x), expanded)
            assert head._expanded_module is cached
        head.weight.add_(.01)
        head(x)
        assert head._expanded_module is not cached
        assert set(head.state_dict()) == {'weight','bias'}
        head.double()
        assert head._circuit is None
        assert head(x.double()).dtype == torch.float64


def test_digital_controls_change_production_arithmetic():
    torch.manual_seed(2)
    head = DigitalLinear(4,3)
    head.configure(physical=True,q=.1,v_dd=.5)
    x = torch.rand(30,4)*.3
    reference = head.logits_for_loss(head(x))
    head.head_config['final_repr_bits'] = 3
    assert not torch.allclose(reference,head.logits_for_loss(head(x)))
    head.head_config['final_weight_bits'] = 3
    changed = head.logits_for_loss(head(x))
    assert not torch.allclose(reference, changed)
    head.head_config['final_adc_noise_lsb'] = .5
    assert not torch.equal(head(x),head(x))
    head.head_config['final_accumulator_bits'] = 3
    assert head(x).abs().max() <= 4


@pytest.mark.parametrize('kind',['analog','digital'])
def test_real_shell_to_python_parser_wiring(kind):
    from test_tc_launch_alignment import cnn_stages, pcn_stage
    options = dict(FINAL_HEAD_TYPE=kind, FINAL_REPR_BITS='6', FINAL_WEIGHT_BITS='7',
                   FINAL_ADC_NOISE_LSB='.5', FINAL_BIAS_BITS='24',
                   FINAL_ACCUMULATOR_BITS='30', FINAL_HEAD_CLAMP='false')
    for args in [*cnn_stages(**options), pcn_stage('ft',**options), pcn_stage('eval',**options)]:
        assert args.final_head_type == kind
        assert args.final_repr_bits == 6
        assert args.final_weight_bits == 7
        assert args.final_adc_noise_lsb == .5
        assert args.final_bias_bits == 24
        assert args.final_accumulator_bits == 30
        assert args.final_head_clamp is False


def test_recipe_optimizer_includes_analog_bias():
    from trainer_timm import TrainerCiFarTimmStyle
    model=nn.Module()
    model.linear=AnalogLinear(4,3)
    model.final_head_type='analog'
    model.linear_train_scale=.4
    trainer=object.__new__(TrainerCiFarTimmStyle)
    trainer.model=model
    trainer.scale_train_recipe=1
    trainer.ff_train_scale=trainer.fb_train_scale=1
    trainer.bn_weight_decay=None
    trainer.bias_weight_decay=None
    trainer.bias_lr_multiplier=1.
    trainer.bias_weight_decay_multiplier=1.
    groups=trainer._composed_optimizer_parameters(.01,.001)
    assert sum(len(group['params']) for group in groups)==2
    assert all(abs(group['lr']-.01/.4**2)<1e-10 for group in groups)


@pytest.mark.parametrize('family',['toggle','tc'])
def test_real_coupler_bank_is_inherited_and_fixed_for_trial(family):
    from physical_feedforward import AveragedFeedForwardPhysicalWrapper
    from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
    if family == 'tc':
        block=TCPhysicalBasicBlock(nn.Conv2d(4,3,1,bias=False),one_shot_conv=True,
            tc_covariance_table=str(ROOT/'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'))
        wrapper=TCFeedForwardPhysicalWrapper(block)
        path=ROOT/'hardware_data/res_vs_vin_10k_150k.csv'
    else:
        block=template()
        wrapper=AveragedFeedForwardPhysicalWrapper(block)
        path=ROOT/'hardware_data/mc_45_corners/coupler_full_range/tt_25_1.csv'
    wrapper.configure_nonlinear_R_inference(str(path),curve_seed=19)
    head=AnalogLinear(4,3).eval()
    head.configure(physical=True,q=.1,v_dd=block.v_dd,template=block,family=family,
                   R=block.R,C=block.C)
    head.expanded=True
    with torch.no_grad():
        x=torch.rand(2,4)*.03
        a=head(x)
        matrix=head._expanded_module
        assert matrix.csv_enabled
        torch.testing.assert_close(a,head(x),rtol=0,atol=0)
        assert matrix is head._expanded_module
        if family == 'tc':
            assert matrix._tc_curve_package is not None


@pytest.mark.parametrize('factor',[None,1,3])
def test_analog_scale_factor_and_spin_controls(factor):
    block=template()
    block.weight_quant_factor_bits=factor
    block.enable_spin_variation=True
    block.sigma_spin=0.
    block.spin_variation_mean=1.7
    head=AnalogLinear(4,3,config=dict(final_head_clamp=False))
    head.configure(physical=True,q=.1,v_dd=.5,template=block,R=block.R,C=block.C)
    x=torch.rand(2,4)*.01
    raw=head(x)
    from ode_pc import _symmetric_qat_weight_scale
    aug=torch.cat((head.weight,head.bias[:,None]),1)
    scale=_symmetric_qat_weight_scale(aug,factor)
    torch.testing.assert_close(head._circuit.scale1,scale)
    expected=F.linear(torch.cat((x,x.new_full((2,1),.1)),1),
                      (aug*scale*15).round()/15)/scale*1.7
    torch.testing.assert_close(raw,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('kind',['analog','digital'])
@pytest.mark.parametrize('family',['toggle','tc1','tc2'])
@pytest.mark.parametrize('scope',['pc_only','all'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_pcn_losses_features_and_all_gradients(kind,family,scope,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from test_toggle_physical_head import base_model,physical
    from test_tc_physical_head import build_tc
    reference=base_model(measured_activation_scope=scope).to(device)
    actual=base_model(measured_activation_scope=scope).to(device)
    reference.device=actual.device=torch.device(device)
    select_model_head(actual,kind,dict(final_head_quantize=False,final_head_clamp=False))
    if family=='toggle':
        reference,_=physical(reference,measured=True)
        actual,_=physical(actual,measured=True)
    else:
        reference,_=build_tc(int(family[-1]),model=reference,scope=scope,device=device)
        actual,_=build_tc(int(family[-1]),model=actual,scope=scope,device=device)
    # Derived analog MVM is the unit-gain reference. Fixed timing has its
    # separate independent equation/gradient tests above.
    if kind=='analog':
        actual.linear._runtime['timing']='derived'
    optimizers=[torch.optim.SGD(m.parameters(),lr=.001) for m in (reference,actual)]
    for batch in range(3):
        x=torch.rand(2,3,4,4,device=device)
        records=[]
        for model,optimizer in zip((reference,actual),optimizers):
            optimizer.zero_grad()
            torch.manual_seed(10+batch)
            features,raw=model(x,is_feat=True)
            logits=logits_for_loss(raw,model)
            loss=F.cross_entropy(logits,torch.tensor([0,2],device=device))
            # The SRRL-facing feature units must not depend on classifier kind.
            loss=loss+.01*features[-1].square().mean()
            loss.backward()
            records.append((logits.detach(),features[-1].detach(),
                {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(records[0][0],records[1][0],rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(records[0][1],records[1][1],rtol=3e-5,atol=3e-6)
        assert records[0][2].keys()==records[1][2].keys()
        for key in records[0][2]:
            torch.testing.assert_close(records[0][2][key],records[1][2][key],rtol=3e-5,atol=3e-6)
        for optimizer in optimizers: optimizer.step()


@pytest.mark.parametrize('kind',['analog','digital'])
@pytest.mark.parametrize('tc',[False,True])
@pytest.mark.parametrize('measured',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_feedforward_coordinate_and_gradient_reference(kind,tc,measured,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from baseline.cifar_resnet import WideResNetCIFAR
    args=SimpleNamespace(v_dd=.1,one_over_q=1,R=1e4,C=49e-15,w_bits=5,
                         weight_quant_factor_bits=None,tc_feedforward=tc,
                         toggle_timing_mode='derived',toggle_y_time=5e-9)
    original=WideResNetCIFAR(depth=10,widen_factor=1,num_classes=3,in_chans=3,
                            use_batchnorm=False,avgpool_downsample_shortcut=True)
    actual=copy.deepcopy(original)
    select_model_head(actual,kind,dict(final_head_quantize=False,final_head_clamp=False))
    for model in (original,actual):
        convert_wide_resnet_to_physical(model,physical=True,qat=True,
            R=args.R,C=args.C,v_dd=args.v_dd,one_over_q=1,w_bits=5,physical_level=2,
            tc_options=dict(one_shot_conv=True) if tc else None)
        if measured:
            from measured_activation import (configure_feedforward_measured_activation,
                                              feedforward_measured_activation_factory)
            factory=feedforward_measured_activation_factory(
                str(ROOT/'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv'),
                .1,corner='MC18',normalize_positive_endpoint=False,fuse_measured_activation=False)
            configure_feedforward_measured_activation(model,factory,scope='all')
    configure_feedforward_head(actual,args)
    original.to(device); actual.to(device)
    x=torch.rand(2,3,8,8,device=device)
    records=[]
    for model in (original,actual):
        torch.manual_seed(19)
        features,raw=model(x,is_feat=True)
        restored=logits_for_loss(raw,model)
        F.cross_entropy(restored,torch.tensor([0,2],device=device)).backward()
        records.append((features,restored,{n:p.grad for n,p in model.named_parameters()}))
    for a,b in zip(records[0][0],records[1][0]):
        torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(records[0][1],records[1][1],rtol=3e-5,atol=3e-6)
    for name,gradient in records[0][2].items():
        if gradient is not None:
            torch.testing.assert_close(gradient,records[1][2][name],rtol=3e-5,atol=3e-6)


def test_toggle_pipeline_all_three_production_commands(tmp_path):
    import shlex
    import subprocess
    from test_tc_launch_alignment import environment,parsed,pcn_train,pcn_eval
    from scripts.run_toggle_nonideality_ablation import parse_args,build_command
    prefix=(ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text().split('# PHASE 1:',1)[0]
    prefix=prefix.replace('source activate base','true').replace('conda activate scanbase','true')
    prefix='\n'.join('LOGDIR='+shlex.quote(str(tmp_path)) if line.startswith('LOGDIR=')
                     else line for line in prefix.splitlines())
    command=prefix+'''\n
python() { printf 'CAPTURE '; printf '%q ' "$@"; printf '\\n'; }
run_one_combo $'audit\\t1\\t3 4\\t4 4\\t0 0' ToggleODEXInitFFFB
finetune_one_combo_model audit TIMMPCNet_C100_CiFAIR_ToggleODEXInitFFFB ToggleODEXInitFFFB
eval_one_combo_model audit TIMMQATPCNet_C100_CiFAIR_ToggleODEXInitFFFB post_ft ToggleODEXInitFFFB
'''
    env=environment(TOGGLE_MODE='odexinit',FINAL_HEAD_TYPE='digital',FINAL_REPR_BITS='6',
                    FINAL_WEIGHT_BITS='7',FINAL_ADC_NOISE_LSB='.5',COMB_LIST='audit',
                    OUTPUT_SAVE_PATH=str(tmp_path))
    output=subprocess.check_output(['bash','-c',command],cwd=ROOT,env=env,text=True)
    calls=[shlex.split(line[len('CAPTURE '):]) for line in output.splitlines() if line.startswith('CAPTURE ')]
    assert len(calls)==3,output
    for index,argv in enumerate(calls):
        assert '--final_head_type' in argv
        assert '--final_repr_bits' in argv
        argv=argv[1:] if argv[0]=='-u' else argv
        if index<2:
            args=parsed(pcn_train,argv)
        else:
            runner=parsed(parse_args,argv)
            args=parsed(pcn_eval,build_command(runner,'all_known')[2:])
        assert args.final_head_type=='digital'
        assert args.final_repr_bits==6
        assert args.final_weight_bits==7
        assert args.final_adc_noise_lsb==.5


def test_analog_recovery_restores_private_hardware_rng(tmp_path):
    from training_recovery import HISTORY,save_latest,restore_latest,latest_path
    def make():
        model=nn.Module()
        model.linear=AnalogLinear(4,3)
        model.final_head_type='analog'
        model.final_head_config=model.linear.head_config
        block=template()
        block.enable_spin_variation=True
        block.spin_variation_seed=123
        block.enable_summing_current_noise=True
        block.summing_noise_seed=456
        model.linear.configure(physical=True,q=.1,v_dd=.5,template=block,R=block.R,C=block.C)
        return SimpleNamespace(model=model,device='cpu',save_path=str(tmp_path),model_name='head',
                               optimizer=torch.optim.SGD(model.parameters(),lr=.01))
    trainer=make()
    x=torch.rand(2,4)*.02
    trainer.model.linear(x)
    history=dict.fromkeys(HISTORY,None)
    history['val_acc']=.2
    save_latest(trainer,1,history)
    expected=trainer.model.linear(x)
    resumed=make()
    resumed.recovery_checkpoint=latest_path(trainer)
    restore_latest(resumed)
    torch.testing.assert_close(resumed.model.linear(x),expected,atol=0,rtol=0)


def test_analog_output_rail_saturates_and_gates_gradient():
    head=AnalogLinear(4,3,bias=False)
    head.weight.data.fill_(10.)
    block=template()
    head.configure(physical=True,q=.1,v_dd=.5,template=block,R=block.R,C=block.C)
    out=head(torch.full((2,4),.3))
    torch.testing.assert_close(out,torch.full_like(out,.5))
    out.sum().backward()
    assert torch.count_nonzero(head.weight.grad)==0


def test_old_ideal_does_not_require_a_supported_nonideal_head():
    model=nn.Sequential(nn.Linear(4,3))
    select_model_head(model)
    assert model.final_head_type=='old_ideal'
