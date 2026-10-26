"""Batched native-grid correction: packed layout, backward and full RHS replay."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tc_empirical import EmpiricalLevels, grouped_convolution
from tc_shared_correction import batched_empirical_correction, empirical_correction
import test_tc_multilevel_empirical as empirical_tests


def native_draw(device):
    grid=torch.full((15,7),float('nan'),device=device)
    left=torch.full((15,6),float('nan'),device=device)
    slope=torch.full_like(left,float('nan'))
    lengths=torch.tensor([2+k%6 for k in range(15)],device=device)
    nominal=torch.arange(15,device=device,dtype=torch.float32)*1000+20000
    for k,length in enumerate(lengths.tolist()):
        g=torch.linspace(-.08-k*.002,.08+k*.003,length,device=device)
        r=nominal[k]*(1+.2*g+.3*g.square())
        grid[k,:length]=g
        left[k,:length-1]=r[:-1]
        slope[k,:length-1]=r.diff()/g.diff()
    return EmpiricalLevels(dict(v_grid=grid,R_left=left,R_slope=slope,lengths=lengths),
                           torch.arange(15,device=device)),nominal


def scalar_reference(x,draw,nominal):
    return torch.cat([empirical_correction(x,*constants,.1)
                      for constants in draw.prepare(x,nominal)],dim=1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class BatchedCorrectionTests(unittest.TestCase):
    def test_native_knots_rails_noncontiguous_and_backward(self):
        draw,nominal=native_draw('cuda')
        knots=torch.cat([draw.bank['v_grid'][k,:n] for k,n in enumerate(draw.bank['lengths'].tolist())])
        values=torch.cat((knots,knots.new_tensor([-.2,-.1,0.,.1,.2])))
        values=values.repeat(8)[:384].reshape(2,3,8,8)
        for noncontiguous in (False,True):
            data=values.transpose(-1,-2) if noncontiguous else values
            expected=None
            for batched in (False,True):
                x=data.detach().requires_grad_()
                y=(batched_empirical_correction(x,draw,nominal,.1) if batched else scalar_reference(x,draw,nominal))
                self.assertIsNotNone(y)
                probe=torch.linspace(-1,1,y.numel(),device='cuda').reshape_as(y)
                answer=(y.detach(),torch.autograd.grad((y*probe).sum(),x)[0])
                if expected is None:expected=answer
                else:
                    torch.testing.assert_close(answer[0],expected[0],atol=0,rtol=0)
                    torch.testing.assert_close(answer[1],expected[1],atol=1e-5,rtol=1e-4)
        self.assertIsNone(batched_empirical_correction(values.double(),draw,nominal,.1))
        self.assertIsNone(batched_empirical_correction(values.cpu(),draw,nominal.cpu(),.1))
        self.assertIsNone(batched_empirical_correction(values,draw,nominal.clone().requires_grad_(),.1))

    def test_no_input_gradient_and_nonfinite_semantics(self):
        draw,nominal=native_draw('cuda')
        x=torch.tensor([float('nan'),float('inf'),-float('inf'),0.],device='cuda').reshape(1,1,1,4)
        a=batched_empirical_correction(x,draw,nominal,.1)
        b=scalar_reference(x,draw,nominal)
        torch.testing.assert_close(a,b,atol=0,rtol=0,equal_nan=True)
        self.assertIsNone(a.grad_fn)

    def test_convolution_and_transpose_weight_gradients(self):
        draw,nominal=native_draw('cuda')
        for transpose in (False,True):
            for groups in (1,2):
                cls=nn.ConvTranspose2d if transpose else nn.Conv2d
                options=dict(stride=2,padding=1,bias=False,groups=groups)
                if transpose:options['output_padding']=1
                module=cls(4,6,3,**options).cuda()
                codes=(module.weight.detach().abs()*60).round().long().clamp(0,15)
                data=torch.rand(2,4,5,5,device='cuda')*.3-.15
                answers=[]
                for batched in (False,True):
                    x=data.clone().requires_grad_()
                    packed=batched_empirical_correction(x,draw,nominal,.1) if batched else scalar_reference(x,draw,nominal)
                    y=grouped_convolution(module,packed,module.weight,codes,15)
                    answers.append((y.detach(),*torch.autograd.grad(y.square().mean(),(x,module.weight))))
                for actual,expected in zip(answers[1],answers[0]):
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)

    def test_complete_pcn_both_states_and_checkpoint_replay(self):
        helper=empirical_tests.EmpiricalTCTests()
        helper.setUpClass()
        for two in (False,True):
            for checkpoint in (False,True):
                answers=[]
                for batched in (False,True):
                    model=helper.wrapper(two,checkpoint).cuda()
                    # This CPU fixture stores its original time vector as a
                    # plain wrapper attribute, not a registered module buffer.
                    model.orig_integration_time=model.orig_integration_time.cuda()
                    model.ode_block._tc_conv_method='grouped'
                    model.ode_block._tc_batched_correction=batched
                    torch.manual_seed(37)
                    x=(torch.rand(4,2,3,3,device='cuda')*.08).requires_grad_()
                    with patch('tc_shared_correction.batched_empirical_correction',wraps=batched_empirical_correction) as called:
                        y=model(x)
                        gradients=torch.autograd.grad(y.square().mean(),(x,*model.parameters()),allow_unused=True)
                        self.assertEqual(called.call_count>0,batched)
                    answers.append((y.detach(),*[g for g in gradients if g is not None]))
                for actual,expected in zip(answers[1],answers[0]):
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)

    def test_analog_head_inherits_control_and_preserves_gradients(self):
        from final_linear import AnalogLinear
        from tc_empirical import prepare_empirical_curves
        helper=empirical_tests.EmpiricalTCTests()
        helper.setUpClass()
        package=prepare_empirical_curves(str(empirical_tests.BANK),
            levels=torch.arange(16,device='cuda')/15,R=13300.,R_max=200000.,
            dtype=torch.float32,device='cuda')
        for two in (False,True):
            answers=[]
            for batched in (False,True):
                template=helper.wrapper(two).ode_block
                template._tc_curve_package=package
                template._tc_conv_method='grouped'
                template._tc_batched_correction=batched
                torch.manual_seed(28)
                head=AnalogLinear(2,3,bias=False).cuda()
                head.configure(physical=True,q=.1,v_dd=.1,template=template,
                    family='tc',R=13300.,C=49e-15,timing='derived',base_time=1e-9)
                x=(torch.rand(4,2,device='cuda')*.01).requires_grad_()
                with patch('tc_shared_correction.batched_empirical_correction',wraps=batched_empirical_correction) as called:
                    y=head(x)
                    gradients=torch.autograd.grad(y.square().mean(),(x,head.weight))
                    self.assertEqual(called.call_count>0,batched)
                answers.append((y.detach(),*gradients))
            for actual,expected in zip(answers[1],answers[0]):
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)


if __name__=='__main__':unittest.main()
