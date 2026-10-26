"""Native-grid, level-conditioned empirical TC; never scientific accuracy tests."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tc_empirical import (prepare_curves, prepare_empirical_curves,
                          empirical_bank_on, sampling_mode)
from ode_pc import QATWrapper1State, QATWrapper2State
from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
from test_tc_dense_training import make_block

BANK = Path(__file__).resolve().parents[1] / 'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'


class EmpiricalTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.package = prepare_empirical_curves(str(BANK), levels=torch.arange(16)/15,
            R=13300., R_max=200000., dtype=torch.float32, device='cpu')

    def wrapper(self, two=False, checkpoint=False):
        block = make_block(two)
        block.option_aca['checkpoint_ode_rhs_training'] = checkpoint
        wrapper = (QATWrapper2State if two else QATWrapper1State)(
            ode_block=block, tc_nonidealities=True, R=13300., R_max=200000.,
            C=49e-15, k=1e3, v_dd=.1, state_bound=1., w_bits=5,
            thermal_noise=False, offset_eps=0., is_first=True, is_last=True,
            nonlinear_R=True, nonlinear_R_table='/not-used', tc_covariance_table='/not-used',
            nonlinear_R_curve_sampling='empirical_with_replacement',
            tc_empirical_curve_bank=str(BANK), nonlinear_R_curve_seed=19)
        return wrapper

    def test_unused_gaussian_inputs_and_shared_native_bank(self):
        with patch('tc_nonidealities.prepare_tc_resistance_curves', side_effect=AssertionError):
            package = prepare_curves('/missing-mean', '/missing-covariance',
                mode='empirical_with_replacement', empirical_path=str(BANK),
                levels=torch.arange(16)/15, R=13300., R_max=200000.,
                dtype=torch.float32, device='cpu')
        self.assertIs(package.empirical_bank, self.package.empirical_bank)
        bank = empirical_bank_on(package.empirical_bank, 'cpu', torch.float32)
        self.assertIs(bank, empirical_bank_on(package.empirical_bank, 'cpu', torch.float32))
        self.assertEqual(bank['v_grid'].shape, (67500, 52))
        self.assertFalse(torch.equal(bank['v_grid'][0], bank['v_grid'][1]))

    def test_independent_codes_layers_and_ff_fb(self):
        one, two = self.wrapper(), self.wrapper()
        two.ode_block.layer_idx = 1
        two.ode_block._tc_curve_generator.manual_seed(19+1009)
        first = one.ode_block._tc_curves_for_solve()
        second = two.ode_block._tc_curves_for_solve()
        expected = torch.arange(15)
        for draw in (*first.values(), *second.values()):
            self.assertTrue(torch.equal(draw.indices // 4500, expected))
        self.assertFalse(torch.equal(first['FFconv'].indices, first['FBconv'].indices))
        self.assertFalse(torch.equal(first['FFconv'].indices, second['FFconv'].indices))
        again = one.ode_block._tc_curves_for_solve()
        self.assertFalse(torch.equal(first['FFconv'].indices, again['FFconv'].indices))
        one.eval()
        fixed = one.ode_block._tc_curves_for_solve()
        self.assertIs(fixed, one.ode_block._tc_curves_for_solve())

    def test_checkpoint_forward_and_gradient_parity_both_states(self):
        for two in (False, True):
            a, b = self.wrapper(two), self.wrapper(two, True)
            x = torch.rand(2, 2, 2, 2)*.1
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya, yb = a(xa), b(xb)
            ya.square().sum().backward(); yb.square().sum().backward()
            torch.testing.assert_close(ya, yb, rtol=0, atol=0)
            torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-5, atol=1e-7)
            for pa, pb in zip(a.parameters(), b.parameters()):
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-7)
                    self.assertTrue(torch.isfinite(pa.grad).all())

    def test_corner_coupling_and_measured_pooling(self):
        from ft_corner_sampling import configure_ft_corner_coupled_sampling
        from measured_pooling import MeasuredAvgPool2d
        from tc_empirical import pooling_options
        from tc_feedforward_cli import configure_pooling
        model = nn.Sequential(self.wrapper(), nn.AvgPool2d(2))
        args = SimpleNamespace(nonlinear_R_curve_sampling='empirical_with_replacement',
            tc_empirical_curve_bank=str(BANK), measured_pooling_nominal_R=13300.,
            R=13300., nonlinear_R_curve_seed=13)
        model[1] = MeasuredAvgPool2d(kernel_size=2, seed=13,
            **pooling_options(args, None))
        self.assertEqual(Path(pooling_options(args, None)['curve_path']).name, 'pseudo_res_75uS.csv')
        participants = configure_ft_corner_coupled_sampling(model)
        self.assertIn('nonlinear-R', participants)
        self.assertIn('measured-pooling', participants)
        output = model(torch.rand(2, 2, 4, 4)*.1)
        self.assertTrue(torch.isfinite(output).all())
        corner = model._last_ft_corner_coupled_corner
        allowed = self.package.empirical_bank['corner_indices'][corner]
        for draw in model[0].ode_block._tc_curve_samples.values():
            for code, row in enumerate(draw.indices):
                self.assertIn(row.item(), allowed[code].tolist())
        self.assertTrue(any(isinstance(m, MeasuredAvgPool2d) for m in model.modules()))

    def test_cnn_training_and_native_expanded_assignment(self):
        from validation import MVMConv
        b = TCPhysicalBasicBlock(nn.Conv2d(2, 2, 1, bias=False),
            R=13300., C=49e-15, v_dd=.1,
            tc_method='euler', tc_num_steps=2,
            tc_distribution='empirical_with_replacement', tc_empirical_curve_bank=str(BANK))
        w = TCFeedForwardPhysicalWrapper(b, qat=True)
        w.configure_nonlinear_R_training('/unused', curve_seed=5)
        x = torch.rand(2, 2, 2, 2)*.02
        y = w(x); y.sum().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(any(p.grad is not None for p in w.parameters()))
        mat = torch.tensor([[0., 1., -2/15], [3/15, 0., 4/15]]).to_sparse_csr()
        meta = dict(inp_chan=3, out_chan=2, ker_h=1, ker_w=1, stride=1, padding=0)
        m = MVMConv(mat, meta)
        m.enable_csv(v_grid=None, R_codes=None, R_left=None, R_slope=None,
                     R=13300., proj_fn=None, tc_curve_package=self.package, nonlinear_R_curve_seed=12)
        indices = m.nonlinear_R_curve_assignment.clone()
        codes = m._values_to_code_idx(mat.values())
        self.assertTrue(torch.equal(indices//4500, codes))
        xx = torch.rand(2, 3, 1, 1)*.08
        yy = m(xx)
        torch.testing.assert_close(yy, m(xx), rtol=0, atol=0)
        self.assertTrue(torch.equal(indices, m.nonlinear_R_curve_assignment))
        # Independent scalar calculation, on each selected curve's native grid.
        from utils import interpolate_R_eff
        expected = torch.zeros(2, 2)
        bank = empirical_bank_on(self.package.empirical_bank, 'cpu', torch.float32)
        for row in range(2):
            for edge in range(mat.crow_indices()[row], mat.crow_indices()[row+1]):
                curve = indices[edge]; col = mat.col_indices()[edge]
                grid = bank['v_grid'][curve]
                v = xx[:, col, 0, 0]
                r = interpolate_R_eff(v, grid, torch.ones(1),
                    bank['R_left'][curve, :, None], bank['R_slope'][curve, :, None], 0)
                expected[:, row] += mat.values()[edge].sign()*v*13300./r
        torch.testing.assert_close(yy[:, :, 0, 0], expected)

    def test_mode_defaults_and_ignored_empirical_path_for_gaussian(self):
        for tc, expected in ((True, 'multivariate_gaussian'), (False, 'empirical_with_replacement')):
            args = SimpleNamespace(tc_nonidealities=tc, nonlinear_R_curve_sampling=None)
            self.assertEqual(sampling_mode(args), expected)
        with patch('tc_nonidealities.prepare_tc_resistance_curves', return_value='unchanged') as old:
            self.assertEqual(prepare_curves('mean', 'cov', empirical_path='/ignored', R=1), 'unchanged')
            old.assert_called_once_with('mean', 'cov', R=1)
        legacy = SimpleNamespace(tc_nonidealities=True,
            nonlinear_R_curve_sampling='empirical_with_replacement',
            _exact_training_recovery=True)
        self.assertEqual(sampling_mode(legacy), 'multivariate_gaussian')

    def test_real_pcn_launcher_selection_and_pooling_defaults(self):
        from test_tc_cli import resolved
        for stage in ('ft', 'eval'):
            args = resolved(stage, NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                TC_EMPIRICAL_CURVE_BANK=str(BANK), R_VAL='13.3e3', R_MAX='200e3',
                TC_MEAN_TABLE='/ignored-mean', TC_COVARIANCE_TABLE='/ignored-covariance')
            self.assertEqual(args.nonlinear_R_curve_sampling, 'empirical_with_replacement')
            self.assertEqual(args.measured_pooling_nominal_R, 13300.)
            self.assertEqual(args.tc_empirical_curve_bank, str(BANK))

    def test_recovery_rejects_distribution_or_bank_changes(self):
        import tempfile
        from test_training_recovery import make_trainer
        from training_recovery import save_latest, restore_latest, latest_path, HISTORY
        config = dict(tc_nonidealities=True, _tc_distribution_version=1,
            nonlinear_R_curve_sampling='empirical_with_replacement',
            tc_empirical_curve_bank='bank', _tc_empirical_identity='hash')
        for field, value in (('nonlinear_R_curve_sampling', 'multivariate_gaussian'),
                             ('tc_empirical_curve_bank', 'other'),
                             ('_tc_empirical_identity', 'changed')):
            with tempfile.TemporaryDirectory() as folder:
                trainer = make_trainer(folder)
                trainer.recovery_config = config
                save_latest(trainer, 1, dict.fromkeys(HISTORY, None))
                trainer.recovery_checkpoint = latest_path(trainer)
                trainer.recovery_config = dict(config, **{field: value})
                with self.assertRaisesRegex(ValueError, 'unchanged'):
                    restore_latest(trainer)

    def test_analog_head_inherits_empirical_package_both_states(self):
        from final_linear import AnalogLinear
        for two in (False, True):
            template = self.wrapper(two).ode_block
            head = AnalogLinear(2, 3, bias=False)
            head.configure(physical=True, q=.1, v_dd=.1, template=template,
                           family='tc', R=13300., C=49e-15,
                           timing='derived', base_time=1e-9)
            output = head(torch.rand(4, 2)*.01)
            output.sum().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertIs(head._circuit._tc_resistance_package.empirical_bank,
                          template._tc_curve_package.empirical_bank)

    def test_cnn_real_launchers_accept_empirical_selector(self):
        import os
        import shlex
        import subprocess
        import sys
        from baseline.train_baseline_cifar import parse_args as train_parser
        from baseline.evaluate_physical_feedforward_cifar import parse_args as eval_parser
        for script, parser in (('run_feedforward_physical_ft.sh', train_parser),
                               ('run_feedforward_physical_eval.sh', eval_parser)):
            env = dict(os.environ, TC_FEEDFORWARD='true', MODEL_NAME='resnet44_cifar',
                TASK='cifar100', IMG_TYPE='rgb', MODEL_CKPT='/not-loaded.pth',
                NONLINEAR_R_CURVE_SAMPLING='empirical_with_replacement',
                TC_EMPIRICAL_CURVE_BANK=str(BANK), R_VAL='13.3e3', R_MAX='200e3')
            result = subprocess.run(['bash', '-c',
                'python() { :; }; export -f python; bash launch_scripts/'+script],
                cwd=BANK.parents[2], env=env, text=True, capture_output=True, check=True)
            printed = next(line for line in result.stdout.splitlines() if line.startswith('Running:'))
            with patch.object(sys, 'argv', shlex.split(printed[len('Running:'):])[1:]):
                args = parser()
            self.assertEqual(args.nonlinear_R_curve_sampling, 'empirical_with_replacement')
            self.assertEqual(args.tc_empirical_curve_bank, str(BANK))
            self.assertEqual(args.measured_pooling_nominal_R, 13300.)


if __name__ == '__main__':
    unittest.main()
