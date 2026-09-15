"""TC CNN: physical current/noise parity and integration-mode comparisons."""
import copy
import tempfile
import unittest
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
from physical_feedforward import convert_wide_resnet_to_physical
from feedforward_validation import FeedForwardCNNValidator
from tc_nonidealities import TCResistanceCurves
from test_tc_dense_training import make_block, wrap


def block(**options):
    conv = nn.Conv2d(2, 2, 1, bias=False).double()
    conv.weight.data.copy_(torch.tensor([.2, -.4, 0., .8]).reshape_as(conv.weight))
    return TCPhysicalBasicBlock(conv, R=1e4, C=49e-15, v_dd=.1, **options).double()


def package(b, variation=True):
    levels=b._get_quant_magnitude_levels(b.conv1.weight)
    r=b.R/levels[1:]
    grid=torch.tensor([-.1,0,.1],dtype=torch.float64)
    return TCResistanceCurves(grid,levels,r,r,r[:,None].expand(-1,3).clone(),
        torch.eye(3,dtype=torch.float64)*(100 if variation else 0),'synthetic','synthetic')


class TCFeedForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_deterministic_outputs_and_gradients_both_timing_modes(self):
        for timing in ('fixed','derived'):
            a=block(one_shot_conv=True,toggle_timing_mode=timing,toggle_y_time=4.9e-10)
            b=copy.deepcopy(a);b.one_shot_conv=False
            x=torch.rand(4,2,3,3,dtype=torch.float64)*.01
            xa=x.clone().requires_grad_();xb=x.clone().requires_grad_()
            ya=a(xa);yb=b(xb)
            torch.testing.assert_close(ya,yb,rtol=2e-6,atol=2e-9)
            ya.sum().backward();yb.sum().backward()
            torch.testing.assert_close(xa.grad,xb.grad)
            torch.testing.assert_close(a.conv1.weight.grad,b.conv1.weight.grad)

    def test_qat_compensation(self):
        a=block(one_shot_conv=True)
        w=TCFeedForwardPhysicalWrapper(a,qat=True).double()
        x=torch.rand(2,2,3,3,dtype=torch.float64)*.01
        actual=w(x)
        expected=torch.nn.functional.conv2d(x,a.conv1.weight)/a.scale1
        torch.testing.assert_close(actual,expected)

    def test_pcn_ff_noise_coefficients_match(self):
        pcn=make_block();pw=wrap(pcn,enable_summing_current_noise=True,
            enable_coupler_noise=True,summing_current_p=.7e-12,coupler_noise_p=.4e-12)
        x=torch.ones(2,2,2,2)*.02
        pc,_,_=pcn._tc_prepare_noise(x)
        cnn=block(enable_summing_current_noise=True,enable_coupler_noise=True,
            summing_current_p=.7e-12,coupler_noise_p=.4e-12)
        cnn.conv1=copy.deepcopy(pcn.FFconv)
        state=torch.zeros_like(x)
        cc,_=cnn._noise_context(cnn.conv1,x,state,'z')
        for actual,expected in zip(cc.coefficients[0],pc.coefficients[0]):
            torch.testing.assert_close(actual,expected)

    def test_single_solver_step_exact_noise_replay(self):
        a=block(one_shot_conv=True,enable_summing_current_noise=True,enable_coupler_noise=True)
        b=copy.deepcopy(a);b.one_shot_conv=False;b.tc_method='euler';b.tc_step_size=a.R*a.C
        x=torch.ones(100,2,1,1,dtype=torch.float64)*.01
        torch.testing.assert_close(a(x),b(x),rtol=2e-6,atol=2e-9)

    def test_multistep_noise_distribution(self):
        a=block(one_shot_conv=True,enable_summing_current_noise=True,enable_coupler_noise=True)
        b=copy.deepcopy(a);b.one_shot_conv=False;b.tc_method='euler';b.tc_step_size=a.R*a.C/8
        x=torch.zeros(20000,2,1,1,dtype=torch.float64)
        _,eps=a._noise_context(a.conv1,x,x,'z')
        expected=eps.flatten().square()*a.R*a.C
        for model in (a,b):
            y=model(x).detach().reshape(-1,2)
            torch.testing.assert_close(y.var(0),expected,rtol=.04,atol=0)
            self.assertTrue(torch.all(y.mean(0).abs()<5*(expected/len(y)).sqrt()))

    def test_saturation_modes_are_not_claimed_identical(self):
        a=block(one_shot_conv=True,enable_coupler_noise=True,coupler_noise_p=1e-9)
        b=copy.deepcopy(a);b.one_shot_conv=False;b.tc_method='euler';b.tc_step_size=a.R*a.C/8
        x=torch.ones(1000,2,1,1,dtype=torch.float64)*.09
        ya,yb=a(x),b(x)
        self.assertTrue((ya.abs()<=.1).all() and (yb.abs()<=.1).all())
        self.assertGreater(float((ya-yb).detach().abs().max()),.01)

    def test_shared_curve_lifetime_and_mode_parity(self):
        a=block(one_shot_conv=True,enable_spin_variation=True,spin_variation_seed=3)
        a._tc_resistance_package=package(a)
        b=copy.deepcopy(a);b.one_shot_conv=False
        x=torch.ones(3,2,2,2,dtype=torch.float64)*.01
        torch.testing.assert_close(a(x),b(x))
        old=a._tc_samples['conv1'].resistance.clone();a(x)
        self.assertFalse(torch.equal(old,a._tc_samples['conv1'].resistance))
        a.eval();first=a(x);torch.testing.assert_close(first,a(x))
        a.reset_nonlinear_R_variation();self.assertFalse(a._tc_samples)

    def test_validator_uses_tc_per_coupler_not_pulses(self):
        for one in (False,True):
            a=block(one_shot_conv=one)
            a.conv1.weight.data.copy_(torch.tensor([3/15,-6/15,0.,12/15]).reshape_as(a.conv1.weight))
            w=TCFeedForwardPhysicalWrapper(a)
            pkg=package(a,variation=False)
            w.install_nonlinear_R_inference_package(dict(tc_curve_package=pkg,
                v_grid=None,R_codes=None,R_left=None,R_slope=None,proj_fn=w.proj_fn,
                R=a.R,nonlinear_R_curve_seed=7))
            model=nn.Sequential(w).eval()
            x=torch.ones(2,2,2,2,dtype=torch.float64)*.01
            expected=model(x)
            with tempfile.TemporaryDirectory() as path:
                loader=DataLoader(TensorDataset(x,torch.zeros(2)),batch_size=2)
                FeedForwardCNNValidator(model,path,'cpu',loader,path,wrapper=[w])
                torch.testing.assert_close(model(x),expected)
                curves=a.conv1.nonlinear_R_curve_gaussian_R_normalized.clone()
                model(x);torch.testing.assert_close(curves,a.conv1.nonlinear_R_curve_gaussian_R_normalized)

    def test_no_toggle_class_or_integer_weight_conversion(self):
        from baseline.cifar_resnet import WideResNetCIFAR
        model=WideResNetCIFAR(depth=10,widen_factor=1,num_classes=10,
                             use_batchnorm=False,conv_bias=False)
        converted=convert_wide_resnet_to_physical(model,physical_level=2,
            R=1e4,C=49e-15,v_dd=.1,tc_options=dict(one_shot_conv=True))
        self.assertIsInstance(converted.conv1,TCFeedForwardPhysicalWrapper)
        self.assertIsInstance(converted.conv1.block,TCPhysicalBasicBlock)

    def test_launchers_resolve_through_real_parsers(self):
        from baseline.train_baseline_cifar import parse_args as train_parser
        from baseline.evaluate_physical_feedforward_cifar import parse_args as eval_parser
        for stage,script,parser in (
            ('pretrain','run_feedforward_cifar_pretrain.sh',train_parser),
            ('ft','run_feedforward_physical_ft.sh',train_parser),
            ('eval','run_feedforward_physical_eval.sh',eval_parser)):
            env=dict(os.environ, TC_FEEDFORWARD='true', MODEL_NAME='wrn_28_2_cifar_nobn_no_bias_avgpool',
                     MODEL_CKPT='/tmp/checkpoint-not-read.pth',IMG_TYPE='rgb',TASK='cifar100')
            result=subprocess.run(['bash','-c',
                'python() { :; }; export -f python; bash launch_scripts/'+script],
                env=env,text=True,capture_output=True,check=True)
            line=next(x for x in result.stdout.splitlines() if x.startswith('Running:'))
            argv=shlex.split(line[len('Running:'):])[1:]
            with patch.object(sys,'argv',argv):
                args=parser()
            self.assertTrue(args.tc_feedforward)
            self.assertFalse(args.one_shot_conv)
            self.assertEqual(args.R,1e4)
            self.assertEqual(args.C,49e-15)
            self.assertEqual(args.physical_level,2)
            self.assertIsNone(args.enob)
            if stage!='pretrain':
                self.assertTrue(args.enable_coupler_noise and args.enable_measured_pooling)
                self.assertEqual(args.coupler_noise_p,.6e-12)
            if stage=='eval':
                self.assertEqual(args.activation_curve_sharing,'per_spin')
                self.assertTrue(args.activation_curve_path.endswith('0906_RELU_Voltage'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_qat_curves_noise_and_backward(self):
        torch.manual_seed(41)
        outputs=[]
        for one in (True,False):
            b=block(one_shot_conv=one,enable_coupler_noise=True,
                    enable_summing_current_noise=True)
            w=TCFeedForwardPhysicalWrapper(b,qat=True)
            w.install_nonlinear_R_training_package(dict(tc_curve_package=package(b),
                                                         nonlinear_R_curve_seed=7))
            w=w.cuda().float()
            x=torch.full((4,2,3,3),.01,device='cuda',requires_grad=True)
            y=w(x); y.square().sum().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(b.conv1.parametrizations.weight.original.grad).all())
            self.assertEqual(b._tc_resistance_package.means.device.type,'cuda')
            outputs.append(y.detach())

    def test_toy_wrn_real_curves_measured_components_and_backward(self):
        from baseline.cifar_resnet import WideResNetCIFAR
        from physical_feedforward import iter_physical_wrappers
        from measured_activation import configure_feedforward_measured_activation, feedforward_measured_activation_factory
        from tc_feedforward_cli import configure_pooling
        from types import SimpleNamespace
        torch.manual_seed(40)
        fp=WideResNetCIFAR(depth=10,widen_factor=1,base_width=2,num_classes=10,
                          use_batchnorm=False,conv_bias=False,avgpool_main_downsample=True,
                          avgpool_downsample_shortcut=True).double()
        means='./hardware_data/res_vs_vin_10k_150k.csv'
        cov='./hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'
        models=[]
        for one in (True,False):
            model=convert_wide_resnet_to_physical(copy.deepcopy(fp),qat=True,
                physical_level=2,R=1e4,C=49e-15,v_dd=.1,
                enable_spin_variation=True,spin_variation_seed=5,
                tc_options=dict(one_shot_conv=one,tc_covariance_table=cov))
            wrappers=list(iter_physical_wrappers(model))
            pkg=wrappers[0].configure_nonlinear_R_training(means,curve_seed=4)
            for w in wrappers[1:]:w.install_nonlinear_R_training_package(pkg)
            factory=feedforward_measured_activation_factory(
                './hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv',.1,
                corner='MC18',curve_sharing='per_model')
            configure_feedforward_measured_activation(model,factory)
            configure_pooling(model,SimpleNamespace(measured_pooling_curve_path=means,
                nonlinear_R_table=means,tc_covariance_table=cov,measured_pooling_nominal_R=1e4,
                nonlinear_R_curve_seed=3))
            models.append(model.double())
        x=torch.rand(2,3,8,8,dtype=torch.float64)*.01
        ys=[m(x) for m in models]
        torch.testing.assert_close(*ys,rtol=2e-5,atol=1e-7)
        for y in ys:y.square().sum().backward()
        for m in models:
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__=='__main__':
    unittest.main()
