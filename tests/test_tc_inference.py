"""Per-coupler TC software checks against an explicit current sum."""
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from validation import MVMConv, Validator, conv2d_to_matrix_fixed_padding
from test_tc_dense_training import make_block, wrap, synthetic_package
from ode_pc import ODEWrapper1State, ODEWrapper2State


def expanded(package, seed=12, stride=1):
    kernel = torch.tensor([1., -2/3, 0., 1/3, 1., -1/3, 0., 2/3, 1.],
                          dtype=torch.float64).reshape(1,1,3,3)
    mat, _, _ = conv2d_to_matrix_fixed_padding((1,3,3),kernel,stride=stride,padding=1)
    module = MVMConv(mat, dict(padding=1,stride=stride,ker_h=3,ker_w=3,
                               inp_chan=1,out_chan=1))
    module.enable_csv(None,None,None,None,None,1e4,tc_curve_package=package,
                      nonlinear_R_curve_seed=seed, nonlinear_R_curve_edge_chunk_size=3)
    return module


class TCInferenceTests(unittest.TestCase):
    def package(self):
        b=make_block(); b.q_hi=3; b.weight_scale=1.
        return synthetic_package(b)

    def test_explicit_edge_sum_and_static_reproducible_sampling(self):
        p=self.package()
        for stride in (1,2):
            m=expanded(p,stride=stride)
            x=torch.linspace(-.15,.15,18,dtype=torch.float64).reshape(2,1,3,3)
            curves=m.nonlinear_R_curve_gaussian_R_normalized.clone()
            rows=m.nonlinear_R_curve_row_ids; cols=m.mat.col_indices()
            expected=torch.zeros((m.mat.shape[0],2),dtype=x.dtype)
            flat=x.reshape(2,-1).T
            for edge,w in enumerate(m.mat.values()):
                if w == 0:
                    continue
                q=flat[cols[edge]].clamp(-.1,.1)
                interval=(q>=0).long()
                r=curves[edge,interval]+(curves[edge,interval+1]-curves[edge,interval])*(q-p.v_grid[interval])/.1
                expected[rows[edge]] += w.sign()*flat[cols[edge]]/r
            with patch.object(m,'_forward_pulse_per_edge',wraps=m._forward_pulse_per_edge) as helper:
                actual=m(x)
                self.assertEqual(helper.call_count,1)
            torch.testing.assert_close(actual.reshape(2,-1).T,expected)
            torch.testing.assert_close(m(x),actual)
            torch.testing.assert_close(m.nonlinear_R_curve_gaussian_R_normalized,curves)
            torch.testing.assert_close(expanded(p,stride=stride).nonlinear_R_curve_gaussian_R_normalized,curves)
            self.assertFalse(torch.equal(expanded(p,seed=13,stride=stride).nonlinear_R_curve_gaussian_R_normalized,curves))
            same=torch.nonzero(m.mat.values()==1).flatten()
            self.assertFalse(torch.equal(curves[same[0]],curves[same[1]]))
            # 3x3 on a 3x3 image has 49 genuine locations at stride1,
            # rather than 81 including virtual padding couplers.
            if stride==1:
                self.assertLessEqual(m.mat.values().numel(),49)
            with self.assertRaises(ValueError):
                m.add_noise(.1)

    def test_ideal_curves_recover_mvm_without_double_weight_magnitude(self):
        p=self.package(); p=replace(p,factor=torch.zeros_like(p.factor))
        m=expanded(p)
        x=torch.rand(2,1,3,3,dtype=torch.float64)*.08
        expected=torch.sparse.mm(m.mat,x.reshape(2,-1).T)
        torch.testing.assert_close(m(x).reshape(2,-1).T,expected)

    def test_zero_codes_consume_no_draws(self):
        p=self.package(); m=expanded(p)
        codes=m._values_to_code_idx(m.mat.values())+1
        expected=p.sample(codes,generator=torch.Generator().manual_seed(12),chunk_size=3)/1e4
        torch.testing.assert_close(m.nonlinear_R_curve_gaussian_R_normalized,expected)
        zeros=codes==0
        self.assertTrue(torch.isinf(expected[zeros]).all())

    def test_validator_wires_both_convolutions_and_cache_has_no_realization(self):
        with tempfile.TemporaryDirectory() as directory:
            samples=[]
            for trial in range(4):
                block=make_block(two=trial >= 2)
                # Standard Validator requires transpose convolutions converted
                # to equivalent Conv2d before wrapping (here 1x1).
                if isinstance(block.FBconv,nn.ConvTranspose2d):
                    old=block.FBconv
                    block.FBconv=nn.Conv2d(2,2,1,bias=False)
                    block.FBconv.weight.data.copy_(old.weight.data.transpose(0,1))
                wrapper_cls=ODEWrapper2State if trial >= 2 else ODEWrapper1State
                wrapper=wrapper_cls(ode_block=block,tc_nonidealities=True,
                    R=1e4,R_max=150e3,C=49e-15,k=1e3,v_dd=.1,state_bound=1.,
                    w_bits=5,thermal_noise=False,is_first=True,is_last=True)
                p=synthetic_package(block)
                block._tc_curve_package=p
                block._tc_curve_generator=None
                block._nonlinear_R_pkg=dict(v_grid=None,R_codes=None,R_left=None,R_slope=None,
                    proj_fn=wrapper.proj_fn,R=1e4,tc_curve_package=p,nonlinear_R_curve_seed=100+trial)
                class Model(nn.Module):
                    def __init__(self):
                        super().__init__(); self.PcConvs=nn.ModuleList([block])
                    def forward(self,x):
                        return self.PcConvs[0](x)
                model=Model()
                loader=DataLoader(TensorDataset(torch.ones(2,2,2,2)*.2,torch.zeros(2)),batch_size=2)
                Validator(model,directory,'cpu',loader,directory,wrapper=[wrapper])
                self.assertIsInstance(block.FFconv,MVMConv)
                self.assertIsInstance(block.FBconv,MVMConv)
                ff=block.FFconv.nonlinear_R_curve_gaussian_R_normalized.clone()
                self.assertNotEqual(block.FFconv.nonlinear_R_curve_seed,block.FBconv.nonlinear_R_curve_seed)
                out=model(next(iter(loader))[0])
                self.assertTrue(torch.isfinite(out).all())
                torch.testing.assert_close(ff,block.FFconv.nonlinear_R_curve_gaussian_R_normalized)
                samples.append(ff)
            self.assertFalse(torch.equal(samples[0],samples[1]))
            self.assertFalse(torch.equal(samples[2],samples[3]))


if __name__=='__main__':
    unittest.main()
