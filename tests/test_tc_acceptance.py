"""Step-7 acceptance checks; seeded CPU software tests, not accuracy runs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from tc_nonidealities import TCNoiseLifecycle
from test_tc_dense_training import make_block, wrap, synthetic_package, ROOT
from TorchDiffEqPack.odesolver import odesolve


class TCAcceptanceTests(unittest.TestCase):
    def test_all_fifteen_codes_common_covariance_and_independent_residuals(self):
        block = make_block(); wrapper = wrap(block)
        p = synthetic_package(block)
        factor = torch.tensor([[20.,0.,0.],[5.,15.,0.],[2.,4.,10.]],dtype=torch.float64)
        p = replace(p,factor=factor)
        codes = torch.arange(1,16).expand(16000,15)
        residual = p.sample(codes,generator=torch.Generator().manual_seed(55))-p.means
        for code in range(15):
            torch.testing.assert_close(torch.cov(residual[:,code].T),p.covariance,rtol=.08,atol=3.)
        correlation = torch.corrcoef(residual[:,:,0].T)-torch.eye(15)
        self.assertLess(correlation.abs().max().item(),.035)

    def test_named_noise_streams_independent_across_layers_branches_sources(self):
        draws = []
        for layer in (0,1):
            block = make_block(); block.layer_idx = layer
            wrapper = wrap(block)
            ref = torch.zeros(24000)
            for name in ('spin:y','spin:z','0:sum','0:coupler','1:sum','1:coupler','fb:sum','fb:coupler'):
                draws.append(torch.randn(ref.shape, generator=block._tc_generator(ref,name,19)))
        correlation = torch.corrcoef(torch.stack(draws))
        off_diagonal = correlation - torch.eye(len(draws))
        self.assertLess(off_diagonal.abs().max().item(), .035)
        self.assertLess(torch.stack(draws).mean(1).abs().max().item(), .025)

    def test_empirical_two_state_diffusion_and_fb_current_units(self):
        for two in (False,True):
            block = make_block(two)
            wrapper = wrap(block, enable_summing_current_noise=True, enable_coupler_noise=True,
                           summing_current_p=.7e-12, coupler_noise_p=.4e-12,
                           summing_noise_seed=32, coupler_noise_seed=32)
            if two:
                wrapper.C_ff = 3*wrapper.C_fb
            x = torch.full((24000,2,2,2), .02)
            ctx,_,_ = block._tc_prepare_noise(x,two)
            cap_ff,cap_fb = block._tc_capacitances()
            if two:
                self.assertAlmostEqual(cap_ff/cap_fb,3.)
            y = block.init_y(x); y = y[0] if two else y
            fb = block._tc_nominal_sum(block.FBconv,y)
            ff = block._tc_nominal_sum(block.FFconv,fb)
            streams = []
            for branch,counts in enumerate((ff,fb) if two else (ff,)):
                cap = (cap_ff,cap_fb)[branch]
                expected = (.7e-12**2 + .4e-12**2*counts)*5/cap**2
                coeff = ctx.coefficients[branch]
                physical = ctx.normal(x,branch)*(coeff[0].square()+coeff[1].square()).sqrt()
                # Per-spin variance; repeated batch examples must receive independent draws.
                torch.testing.assert_close(physical.var(0),expected[0],rtol=.035,atol=0)
                for dt in (1e-9,4e-9):
                    torch.testing.assert_close((physical*dt**.5).var(0),expected[0]*dt,rtol=.035,atol=0)
                streams.append(physical[:,0,0,0])
            if two:
                self.assertLess(torch.corrcoef(torch.stack(streams))[0,1].abs().item(),.035)
            else:
                current = ctx.fb_current()
                expected = block._tc_fb_integral.variance_A2*5*((.7/.6)**2+(.4/.6)**2*fb)
                torch.testing.assert_close(current.var(0),expected[0],rtol=.035,atol=0)
                self.assertLess(torch.corrcoef(torch.stack([streams[0],current[:,0,0,0]]))[0,1].abs().item(),.035)

    def test_zero_couplers_have_no_diffusion_but_summing_noise_remains(self):
        for summing in (False,True):
            block = make_block()
            wrapper = wrap(block,enable_coupler_noise=True,enable_summing_current_noise=summing)
            with torch.no_grad():
                for conv in (block.FFconv,block.FBconv):
                    conv.parametrizations.weight.original.zero_()
            ctx,_,_ = block._tc_prepare_noise(torch.ones(2,2,2,2))
            self.assertEqual(ctx.coefficients[0][1].count_nonzero().item(),0)
            generator = ctx.generators[(0,'coupler')]
            before = generator.get_state().clone()
            value = ctx.normal(torch.ones(2,2,2,2),0)
            self.assertTrue(torch.equal(before,generator.get_state()))
            self.assertEqual(bool(value.count_nonzero()),summing)

    def test_real_wrapper_curve_seeds_lifetimes_and_layer_independence(self):
        saved = []
        for layer in (0,1,0):
            block = make_block(); block.layer_idx = layer
            wrapper = wrap(block,nonlinear_R=True,nonlinear_R_train_mode='exact_curve',
                nonlinear_R_table=ROOT/'hardware_data/res_vs_vin_10k_150k.csv',
                tc_covariance_table=ROOT/'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
                nonlinear_R_curve_seed=71)
            first = block._tc_curves_for_solve()
            saved.append(first['FFconv'].clone())
            self.assertFalse(torch.equal(first['FFconv'],first['FBconv']))
            block.eval()
            self.assertIs(first,block._tc_curves_for_solve())
            block.reset_tc_curves()
            self.assertFalse(torch.equal(first['FFconv'],block._tc_curves_for_solve()['FFconv']))
            block.train()
            previous = block._tc_curves_for_solve()
            self.assertFalse(torch.equal(previous['FBconv'],block._tc_curves_for_solve()['FBconv']))
        torch.testing.assert_close(saved[0],saved[2])
        self.assertFalse(torch.equal(saved[0],saved[1]))

    def test_all_enabled_qat_replay_outputs_and_gradients_both_states(self):
        for two in (False,True):
            results = []
            for regenerate in (False,True):
                block = make_block(two)
                wrapper = wrap(block,enable_spin_variation=True,enable_summing_current_noise=True,
                    enable_coupler_noise=True,spin_variation_seed=11,summing_noise_seed=12,
                    coupler_noise_seed=13,enable_measured_activation=True)
                package = synthetic_package(block)
                block._tc_curve_package = replace(package,
                    means=package.means*(1+package.v_grid),factor=package.factor*10)
                block._tc_curve_generator = torch.Generator().manual_seed(14)
                block.option_aca['regenerate_graph'] = regenerate
                x = torch.full((2,2,2,2),.2,requires_grad=True)
                out = block(x)
                params = [conv.parametrizations.weight.original for conv in (block.FFconv,block.FBconv)]
                grads = torch.autograd.grad(out.square().sum(),[x]+params)
                for grad in grads:
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(),0)
                results.append([out.detach()]+[g.detach() for g in grads])
            for a,b in zip(*results):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-8)

    def test_two_state_rejection_final_interval_and_predefined_replay(self):
        for endpoint in (False,True):
            ref = (torch.zeros(3),torch.zeros(3))
            amplitudes = (.02,.06)  # Unequal diffusion amplitudes for y/z.
            coeff = [(torch.full((3,),a),torch.zeros(3)) for a in amplitudes]
            generators = {(i,s):torch.Generator().manual_seed(19+10*i+j)
                          for i in range(2) for j,s in enumerate(('sum','coupler'))}
            ctx = TCNoiseLifecycle(coeff,generators)
            class RHS(torch.nn.Module):
                def forward(self,t,y):
                    return (torch.ones_like(y[0])*.1,torch.ones_like(y[1])*.2)
            rhs = RHS()
            rhs.tc_context = ctx
            solver = odesolve(rhs,ref,dict(method='dopri5',t0=0.,t1=1.,h=.4,
                t_eval=[1.],eps=amplitudes,noise_type='addi',end_point_mode=endpoint),return_solver=True)
            original = solver.adapt_stepsize
            calls = []
            def reject(y,yn,error,h,**kwargs):
                calls.append(ctx.index)
                if len(calls)==1:
                    return h/2,False,True
                return original(y,yn,error,h,**kwargs)
            with patch.object(solver,'adapt_stepsize',side_effect=reject):
                got,steps = solver.integrate(ref,0.,t_eval=[1.],return_steps=True)
            self.assertEqual(calls[:2],[0,0])
            self.assertAlmostEqual(float(steps[-1]),1.)
            for branch in range(2):
                expected = torch.ones(3)*(.1*(branch+1)); last = 0.
                for i,point in enumerate(steps):
                    dt = float(point)-last; last = float(point)
                    expected += ctx.tape[(i,branch)]*dt**.5
                torch.testing.assert_close(got[branch].reshape(-1,3)[-1],expected,rtol=1e-5,atol=1e-7)
            states = {key:g.get_state().clone() for key,g in generators.items()}
            ctx.restart()
            replay = solver.integrate_predefined_grids(ref,0.,predefine_steps=steps,t_eval=[1.])
            for a,b in zip(got,replay):
                torch.testing.assert_close(a,b)
            for key,g in generators.items():
                self.assertTrue(torch.equal(states[key],g.get_state()))


if __name__ == '__main__':
    unittest.main()
