"""TC empirical banks reuse the toggle interpolation without mixing codes."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from test_tc_inference import expanded, TCInferenceTests
from tc_nonidealities import load_tc_empirical_bank


class TCEmpiricalTests(unittest.TestCase):
    def bank(self, package, path, draws=100):
        codes = torch.arange(1, package.means.shape[0]+1).repeat_interleave(draws)
        curves = package.sample(codes, generator=torch.Generator().manual_seed(71))
        np.savez(path, v_grid=package.v_grid.numpy(),
                 programmed_resistances=package.programmed_resistances.numpy(),
                 curves=curves.reshape(-1, draws, curves.shape[-1]).numpy())
        return load_tc_empirical_bank(package, path)

    def test_code_assignment_static_seed_and_lookup_equivalence(self):
        p = TCInferenceTests().package()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.bank(p, Path(tmp)/'bank.npz')
        m = expanded(p)
        self.assertEqual(m.nonlinear_R_curve_sampling, 'empirical_with_replacement')
        self.assertIsNone(m.nonlinear_R_curve_gaussian_R_normalized)
        indices = m.nonlinear_R_curve_assignment.clone()
        active = m.mat.values() != 0
        torch.testing.assert_close(indices[active]//100,
                                   m._values_to_code_idx(m.mat.values())[active])
        torch.testing.assert_close(expanded(p).nonlinear_R_curve_assignment, indices)
        self.assertFalse(torch.equal(expanded(p, seed=99).nonlinear_R_curve_assignment, indices))
        x = torch.linspace(-.15, .15, 18, dtype=torch.float64).reshape(2,1,3,3)
        output = m(x)
        torch.testing.assert_close(m(x), output)
        torch.testing.assert_close(m.nonlinear_R_curve_assignment, indices)
        # Same physical curves, but evaluated through the original Gaussian helper.
        bank = p.empirical_bank
        grid = p.v_grid
        left = bank['R_left']
        curves = torch.cat((left, (left[:,-1] + bank['R_slope'][:,-1]*(grid[-1]-grid[-2]))[:,None]), 1)
        m.nonlinear_R_curve_gaussian_R_normalized = curves[indices]/m.R
        m.nonlinear_R_curve_sampling = 'multivariate_gaussian'
        torch.testing.assert_close(m(x), output)

    def test_ideal_bank_preserves_weight_and_zero_mapping(self):
        p = TCInferenceTests().package()
        p = replace(p, factor=torch.zeros_like(p.factor))
        with tempfile.TemporaryDirectory() as tmp:
            p = self.bank(p, Path(tmp)/'bank.npz', draws=2)
        m = expanded(p)
        x = torch.rand(2,1,3,3,dtype=torch.float64)*.08
        torch.testing.assert_close(m(x).reshape(2,-1).T,
                                  torch.sparse.mm(m.mat,x.reshape(2,-1).T))

    def test_invalid_bank_rejected(self):
        p = TCInferenceTests().package()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.npz'
            np.savez(path, v_grid=p.v_grid.numpy(),
                     programmed_resistances=p.programmed_resistances.numpy(),
                     curves=np.zeros((3,2,3)))
            with self.assertRaises(ValueError): load_tc_empirical_bank(p, path)


if __name__ == '__main__': unittest.main()
