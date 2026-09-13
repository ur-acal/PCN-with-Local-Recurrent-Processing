"""Shared correction must match the prior expression, including clamp boundaries."""
import unittest
from unittest.mock import patch
import torch
from tc_nonidealities import TCSharedCurve
from tc_shared_correction import shared_correction


def reference(x,curve,grid,rail,floor):
    q=x.clamp(-rail,rail).clamp(grid[0],grid[-1])
    idx=(torch.bucketize(q,grid)-1).clamp(0,grid.numel()-2)
    s=(curve.resistance[1:]-curve.resistance[:-1])/(grid[1:]-grid[:-1])
    r=curve.resistance[idx]+s[idx]*(q-grid[idx])
    return x*curve.nominal_R/r.clamp_min(floor)


class CorrectionTests(unittest.TestCase):
    def check_device(self,device,dtype):
        for floor in (1e-6,10000.):
            grid=torch.tensor([-.1,-.07,-.02,0.,.01,.08,.1],device=device,dtype=dtype)
            curve=TCSharedCurve(torch.tensor(1,device=device),
                torch.tensor([8000,12000,11000,10000,9500,14000,9000],device=device,dtype=dtype),
                torch.tensor(10000.,device=device,dtype=dtype))
            x=torch.cat((grid,torch.tensor([-.2,-.09,.09,.2],device=device,dtype=dtype),
                         torch.linspace(-.2,.2,1000,device=device,dtype=dtype))).requires_grad_()
            for rail in (.09,.1,.2):
                y=shared_correction(x,curve,grid,rail,floor)
                ref=reference(x,curve,grid,rail,floor)
                torch.testing.assert_close(y,ref,rtol=2e-6,atol=2e-7)
                grad=torch.linspace(-2,3,x.numel(),device=device,dtype=dtype)
                actual=torch.autograd.grad(y,x,grad)[0]
                expected=torch.autograd.grad(ref,x,grad)[0]
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        # Non-contiguous input and convolution/QAT-weight gradient downstream.
        x=torch.randn(2,3,5,4,device=device,dtype=dtype).transpose(2,3).requires_grad_()
        w=torch.randn(4,3,3,3,device=device,dtype=dtype,requires_grad=True)
        y=torch.nn.functional.conv2d(shared_correction(x,curve,grid,.1,floor),w,padding=1)
        r=torch.nn.functional.conv2d(reference(x,curve,grid,.1,floor),w,padding=1)
        for actual,expected in zip(torch.autograd.grad(y.sum(),(x,w)),torch.autograd.grad(r.sum(),(x,w))):
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)

    def test_cpu(self):self.check_device('cpu',torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda(self):self.check_device('cuda',torch.float32)

    def test_cache_and_invalid_curve(self):
        grid=torch.tensor([-.1,0,.1]);x=torch.tensor([.03])
        curve=TCSharedCurve(torch.tensor(1),torch.tensor([10.,20.,30.]),torch.tensor(10.))
        first=curve.prepare(grid,x,1e-6)
        self.assertIs(first,curve.prepare(grid,x,1e-6))
        curve.resistance.add_(1)
        self.assertIsNot(first,curve.prepare(grid,x,1e-6))
        curve.resistance[0]=float('nan')
        with self.assertRaises(ValueError):curve.prepare(grid,x,1e-6)


if __name__=='__main__':unittest.main()
