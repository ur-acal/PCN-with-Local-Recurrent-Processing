"""Compact storage through the real CNN/PCN expansion and fused inference paths."""
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from validation import ExpandedGaussianCurveStager, MVMConv, Validator
from feedforward_validation import FeedForwardCNNValidator
from physical_feedforward_tc import TCFeedForwardPhysicalWrapper
from ode_pc import ODEWrapper1State, ODEWrapper2State
from test_tc_feedforward import block, package
from test_tc_dense_training import make_block, synthetic_package
from test_tc_inference import expanded, TCInferenceTests
from tc_nonidealities import prepare_tc_resistance_curves


def measured_package(levels):
    return prepare_tc_resistance_curves(
        'hardware_data/res_vs_vin_10k_150k.csv',
        'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv', levels=levels)


class CompactCurveTests(unittest.TestCase):
    def test_all_zero_and_all_active(self):
        for zero in (False, True):
            m = expanded(TCInferenceTests().package())
            m.mat.values().fill_(0. if zero else 1.)
            m.enable_csv(None, None, None, None, None, 1e4,
                         tc_curve_package=m._tc_curve_package,
                         nonlinear_R_curve_seed=12)
            n = m.mat.values().numel()
            self.assertEqual(m.nonlinear_R_curve_gaussian_R_normalized.shape[0],
                             0 if zero else n)
            x = torch.full((2, 1, 3, 3), .01, dtype=torch.float64)
            y = m(x)
            self.assertTrue(torch.isfinite(y).all())
            if zero:
                self.assertEqual(torch.count_nonzero(y), 0)

    def test_cpu_offload_round_trip_is_exact(self):
        m = expanded(TCInferenceTests().package())
        m.enable_csv(None, None, None, None, None, 1e4,
                     tc_curve_package=m._tc_curve_package,
                     nonlinear_R_curve_seed=12)
        expected = m.nonlinear_R_curve_gaussian_R_normalized.clone()
        x = torch.full((2, 1, 3, 3), .01, dtype=torch.float64)
        baseline = m(x)

        self.assertFalse(m.offload_gaussian_curves_to_cpu())
        self.assertIsNone(m.nonlinear_R_curve_gaussian_R_normalized)
        self.assertEqual(m._gaussian_curve_cpu.device.type, 'cpu')
        torch.testing.assert_close(
            m._gaussian_curve_cpu, expected, rtol=0, atol=0)
        self.assertFalse(m.stage_gaussian_curves())
        torch.testing.assert_close(
            m.nonlinear_R_curve_gaussian_R_normalized, expected,
            rtol=0, atol=0)
        torch.testing.assert_close(m(x), baseline, rtol=0, atol=0)
        m.release_staged_gaussian_curves()
        self.assertIsNone(m.nonlinear_R_curve_gaussian_R_normalized)
        torch.testing.assert_close(
            m._gaussian_curve_cpu, expected, rtol=0, atol=0)

    def test_expanded_analog_head_is_staged_by_owner(self):
        from final_linear import AnalogLinear
        from physical_feedforward import (
            PulseFeedForwardPhysicalWrapper, PulsePhysicalBasicBlock)

        block = PulsePhysicalBasicBlock(
            nn.Conv2d(4, 3, 1, bias=False), R=50e3, C=500e-15,
            v_dd=.5, one_over_q=1., w_bits=5).eval()
        wrapper = PulseFeedForwardPhysicalWrapper(block)
        wrapper.configure_nonlinear_R_inference(
            'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
            quantity='resistance', curve_sharing='per_coupler',
            curve_sampling='multivariate_gaussian', curve_seed=23)
        head = AnalogLinear(4, 3).eval()
        head.configure(
            physical=True, q=.5, v_dd=.5, template=block,
            family='toggle', R=50e3, C=500e-15)
        head.begin_evaluation_trial()

        class HeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = head

            def forward(self, x):
                return self.head(x)

        model = HeadModel().eval()
        x = torch.randn(2, 4) * .01
        stager = ExpandedGaussianCurveStager(model, [head], 'cpu')
        try:
            initialization_output = model(x)
            stager.finish_initialization()
            matrix = head._expanded_module
            self.assertIsNotNone(matrix)
            self.assertFalse(matrix.training)
            self.assertIsNone(
                matrix.nonlinear_R_curve_gaussian_R_normalized)
            self.assertIsNotNone(matrix._gaussian_curve_cpu)
            torch.testing.assert_close(
                model(x), initialization_output, rtol=0, atol=0)
        finally:
            stager.close()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CompactPipelineCudaTests(unittest.TestCase):
    def test_all_zero_fused(self):
        m = expanded(TCInferenceTests().package())
        m.mat.values().zero_()
        m.enable_csv(None, None, None, None, None, 1e4,
                     tc_curve_package=m._tc_curve_package)
        for key, value in list(vars(m).items()):
            if isinstance(value, torch.Tensor):
                setattr(m, key, value.to(device='cuda', dtype=(
                    torch.float32 if value.is_floating_point() else value.dtype)))
        m.eval()
        from tc_edge_inference import gaussian_edge_forward
        with torch.no_grad():
            out = gaussian_edge_forward(m, torch.ones(9, 2, device='cuda'),
                                        m._tc_signed_mat, m.R)
        self.assertIsNotNone(out)
        self.assertEqual(torch.count_nonzero(out), 0)

    def test_two_owner_prefetch_repeats_without_curve_growth(self):
        def build(path, offload):
            wrappers = []
            packages = []
            for index in range(3):
                b = block(one_shot_conv=False).float()
                w = TCFeedForwardPhysicalWrapper(b)
                pkg = measured_package(package(b).levels)
                w.install_nonlinear_R_inference_package(dict(
                    tc_curve_package=pkg, v_grid=None, R_codes=None,
                    R_left=None, R_slope=None, proj_fn=w.proj_fn, R=b.R,
                    nonlinear_R_curve_seed=17 + index))
                wrappers.append(w)
                packages.append(pkg)
            model = nn.Sequential(*wrappers).float().cuda().eval()
            x = torch.full((2, 2, 3, 3), .01, device='cuda')
            loader = DataLoader(
                TensorDataset(x, torch.zeros(2)), batch_size=2)
            validator = FeedForwardCNNValidator(
                model, path, 'cuda', loader, path, wrapper=wrappers,
                expanded_gaussian_curve_cpu_offload=offload)
            model.eval()
            return model, x, validator

        with tempfile.TemporaryDirectory() as base_path, \
                tempfile.TemporaryDirectory() as offload_path, \
                torch.no_grad():
            baseline, x, _ = build(base_path, False)
            expected = baseline(x)
            offloaded, _, validator = build(offload_path, True)
            try:
                stager = validator._gaussian_curve_stager
                self.assertEqual(len(stager._active_owners), 3)
                stale_peak = torch.cuda.get_device_properties('cuda').total_memory
                with patch(
                        'torch.cuda.max_memory_allocated',
                        side_effect=[stale_peak, 0]):
                    first = offloaded(x)
                torch.cuda.synchronize()
                self.assertEqual(
                    stager._peak_before_first_forward, stale_peak)
                self.assertEqual(stager._first_forward_peak, 0)
                self.assertTrue(stager._double_buffer)
                torch.testing.assert_close(first, expected, rtol=0, atol=0)
                resident = sum(
                    stager._owner_is_staged(owner)
                    for owner in stager._active_owners)
                self.assertEqual(resident, 2)
                allocated = torch.cuda.memory_allocated()
                for _ in range(4):
                    torch.testing.assert_close(
                        offloaded(x), expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(), allocated)
                self.assertEqual(sum(
                    stager._owner_is_staged(owner)
                    for owner in stager._active_owners), 2)
            finally:
                validator.close_curve_stager()

    def check_model(self, family):
        device = 'cuda'
        if family == 'cnn':
            b = block(one_shot_conv=False)
            w = TCFeedForwardPhysicalWrapper(b)
            pkg = measured_package(package(b).levels)
            w.install_nonlinear_R_inference_package(dict(
                tc_curve_package=pkg, v_grid=None, R_codes=None,
                R_left=None, R_slope=None, proj_fn=w.proj_fn,
                R=b.R, nonlinear_R_curve_seed=7))
            model = nn.Sequential(w)
            validator = FeedForwardCNNValidator
        else:
            two = family == 'pcn2'
            b = make_block(two=two)
            if isinstance(b.FBconv, nn.ConvTranspose2d):
                old = b.FBconv
                b.FBconv = nn.Conv2d(2, 2, 1, bias=False)
                b.FBconv.weight.data.copy_(old.weight.data.transpose(0, 1))
            cls = ODEWrapper2State if two else ODEWrapper1State
            w = cls(ode_block=b, tc_nonidealities=True, R=1e4,
                    R_max=150e3, C=49e-15, k=1e3, v_dd=.1,
                    state_bound=1., w_bits=5, thermal_noise=False,
                    is_first=True, is_last=True)
            b.FFconv.weight.data[0, 0, 0, 0] = 0.
            b.FBconv.weight.data[0, 0, 0, 0] = 0.
            pkg = measured_package(synthetic_package(b).levels)
            b._tc_curve_package = pkg
            b._tc_curve_generator = None
            b._nonlinear_R_pkg = dict(tc_curve_package=pkg,
                v_grid=None, R_codes=None, R_left=None, R_slope=None,
                proj_fn=w.proj_fn, R=1e4, nonlinear_R_curve_seed=7)
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.PcConvs = nn.ModuleList([b])
                def forward(self, x):
                    return self.PcConvs[0](x)
            model = Model()
            validator = Validator
        model.float().to(device)
        x = torch.full((2, 2, 3, 3), .01, device=device)
        loader = DataLoader(TensorDataset(x, torch.zeros(2)), batch_size=2)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            validator(model, directory, device, loader, directory, wrapper=[w])
            modules = [m for m in model.modules() if isinstance(m, MVMConv)]
            self.assertEqual(len(modules), 1 if family == 'cnn' else 2)
            for m in modules:
                n = m.mat.values().numel()
                active = int(torch.count_nonzero(m.mat.values()))
                self.assertLess(active, n)
                self.assertEqual(m.nonlinear_R_curve_gaussian_R_normalized.shape[0], active)
                self.assertFalse(m.training)
                curves = m.nonlinear_R_curve_gaussian_R_normalized
                storage = curves.numel() * curves.element_size() + m._tc_curve_row_index.numel() * 8
                self.assertLess(storage, n * curves.shape[1] * curves.element_size())
            from tc_edge_inference import gaussian_edge_forward
            hits = {id(m): 0 for m in modules}
            def audit(m, *args):
                out = gaussian_edge_forward(m, *args)
                self.assertIsNotNone(out, 'Fused path was silently bypassed')
                hits[id(m)] += 1
                return out
            with patch('tc_edge_inference.gaussian_edge_forward', side_effect=audit):
                compact = model(x)
            self.assertTrue(all(hits.values()))
            for m in modules:
                m._tc_measure_coupler_energy = True
            fallback = model(x)
            powers = [m._tc_last_coupler_power.clone() for m in modules]
            torch.testing.assert_close(compact, fallback, rtol=2e-5, atol=2e-7)
            # Old full layout, identical sampled active curves, no RNG changes.
            for m in modules:
                curves = m.nonlinear_R_curve_gaussian_R_normalized
                rows = m._tc_curve_row_index
                full = curves.new_full((rows.numel(), curves.shape[1]), float('inf'))
                full[rows >= 0] = curves[rows[rows >= 0]]
                m.nonlinear_R_curve_gaussian_R_normalized = full
                m._tc_curve_row_index = None
            dense_storage = model(x)
            torch.testing.assert_close(compact, dense_storage, rtol=2e-5, atol=2e-7)
            for m, power in zip(modules, powers):
                torch.testing.assert_close(m._tc_last_coupler_power, power)
            print(f'{family}: batch=2, compact used by every MVM; '
                  f'fused calls={sum(hits.values())}; full-layout outputs match', flush=True)

    def test_cnn(self):
        self.check_model('cnn')

    def test_pcn1(self):
        self.check_model('pcn1')

    def test_pcn2(self):
        self.check_model('pcn2')


if __name__ == '__main__':
    unittest.main()
