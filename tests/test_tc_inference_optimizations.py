import unittest
from unittest.mock import patch
import torch
from TorchDiffEqPack.odesolver import odesolve
from test_tc_noise import tape
from test_tc_inference import expanded, TCInferenceTests


class AcceptedReuseTests(unittest.TestCase):
    def solve(self, reuse, grad=False, two=False, endpoint=False, project=False,
              meter=None, training_reuse=False, safe=True):
        ref = torch.full((2,), .08, requires_grad=grad)
        ctx = tape(ref, fb=not two)
        if two:
            ctx.coefficients = ctx.coefficients*2
            for source in ('sum', 'coupler'):
                ctx.generators[(1,source)] = torch.Generator().manual_seed(93)
        def rhs(t,y):
            if two: return (-y[0]+y[1], -2*y[1]+y[0])
            return -.3*y+ctx.fb_current()
        rhs.tc_context = ctx
        if two:
            class TwoStateRHS(torch.nn.Module):
                def forward(self, t, y):
                    return (-y[0]+y[1], -2*y[1]+y[0])
            rhs = TwoStateRHS()
            rhs.tc_context = ctx
        state = (ref, ref*.5) if two else ref
        options = dict(method='dopri5',t0=0.,t1=1.,h=.4,
            t_eval=[1.],eps=.036,noise_type='addi',end_point_mode=endpoint)
        options['reuse_accepted_step_training'] = training_reuse
        options['accepted_step_reuse_safe'] = safe
        if project: options['proj_fn'] = torch.nn.Hardtanh(-.03,.03)
        solver = odesolve(rhs, state, options,return_solver=True)
        solver.tc_reuse_accepted_step = reuse
        if project:
            solver.proj_fn = torch.nn.Hardtanh(-.03,.03)
        if meter is not None:
            solver.energy_meter = meter
        original = solver.adapt_stepsize
        attempts = [0]
        def reject_once(*args,**kwargs):
            attempts[0] += 1
            if attempts[0] == 1: return args[3]/2, False, True
            return original(*args,**kwargs)
        with torch.set_grad_enabled(grad), patch.object(solver,'adapt_stepsize',side_effect=reject_once), \
                patch.object(solver,'step',wraps=solver.step) as step:
            result, times = solver.integrate(state,0.,t_eval=[1.],return_steps=True)
        gradient = torch.autograd.grad(result.sum(), ref)[0] if grad else None
        return result, times, step.call_count, ctx.tape, gradient

    def test_same_states_steps_noise_with_rejection_and_endpoint(self):
        for two in (False,True):
            for endpoint in (False,True):
                a = self.solve(False,two=two,endpoint=endpoint)
                b = self.solve(True,two=two,endpoint=endpoint)
                torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
                torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
                self.assertLess(b[2],a[2])
                self.assertEqual(a[3].keys(),b[3].keys())
                for k in a[3]: torch.testing.assert_close(a[3][k],b[3][k],rtol=0,atol=0)

    def test_training_keeps_replay(self):
        a,b = self.solve(False,grad=True),self.solve(True,grad=True)
        self.assertEqual(a[2],b[2])
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)

    def test_training_reuses_accepted_graph_after_rejection(self):
        baseline = self.solve(False,grad=True)
        reused = self.solve(False,grad=True,training_reuse=True)
        torch.testing.assert_close(baseline[0],reused[0],rtol=0,atol=0)
        torch.testing.assert_close(baseline[1],reused[1],rtol=0,atol=0)
        torch.testing.assert_close(baseline[4],reused[4],rtol=0,atol=0)
        self.assertLess(reused[2],baseline[2])
        self.assertEqual(baseline[3].keys(),reused[3].keys())
        for key in baseline[3]:
            torch.testing.assert_close(baseline[3][key],reused[3][key],rtol=0,atol=0)

    def test_training_safety_gate_does_not_disable_existing_inference_reuse(self):
        baseline = self.solve(False, safe=False)
        reused = self.solve(True, safe=False)
        torch.testing.assert_close(baseline[0], reused[0], rtol=0, atol=0)
        self.assertLess(reused[2], baseline[2])

    def test_clamped_states_match(self):
        for two in (False,True):
            a,b = self.solve(False,two=two,project=True),self.solve(True,two=two,project=True)
            torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
            torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)

    def test_energy_observer_keeps_original_replay(self):
        from unittest.mock import Mock
        a,b = self.solve(False,meter=Mock()),self.solve(True,meter=Mock())
        self.assertEqual(a[2],b[2])
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FusedEdgeTests(unittest.TestCase):
    def test_gaussian_reference_signed_zero_padding_projection(self):
        from tc_edge_inference import triton, gaussian_edge_forward
        if triton is None: self.skipTest('Triton unavailable')
        m = expanded(TCInferenceTests().package())
        m.eval()
        for key,value in list(vars(m).items()):
            if isinstance(value,torch.Tensor):
                setattr(m,key,value.to(device='cuda',dtype=torch.float32 if value.is_floating_point() else value.dtype))
        for projection in (None,torch.nn.Hardtanh(-.1,.1)):
            m.proj_fn = projection
            for batch in (1,4,128):
                x = torch.linspace(-.3,.3,batch*9,device='cuda').reshape(batch,1,3,3)
                with torch.no_grad():
                    m._tc_fused_edges = False
                    expected = m(x)
                    m._tc_fused_edges = True
                    self.assertIsNotNone(gaussian_edge_forward(
                        m,x.reshape(batch,-1).T,m._tc_signed_mat,m.R))
                    actual = m(x)
                torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)


if __name__ == '__main__': unittest.main()
