"""Resistance-coded CNN integration; toggle classes and PCN solvers are unchanged."""
import math
from dataclasses import fields, replace
import torch
import torch.nn.functional as F

from TorchDiffEqPack.odesolver import odesolve
from ode_pc import ODEBlockPC
from physical_feedforward import AveragedPhysicalBasicBlock, AveragedFeedForwardPhysicalWrapper
from tc_nonidealities import TCNoiseLifecycle, prepare_tc_resistance_curves
from tc_shared_correction import shared_correction


class TCPhysicalBasicBlock(AveragedPhysicalBasicBlock):
    """Fixed input per convolution, continuous current, no pulse duty cycles."""

    def __init__(self, *args, one_shot_conv=False, tc_method='dopri5',
                 tc_tol=1e-6, tc_step_size=None, tc_noise_reference_R=50e3,
                 tc_covariance_table=None, tc_curve_sampling='histogram', **kwargs):
        for key, value in dict(R=1e4, C=49e-15, v_dd=.1,
                               summing_current_p=.6e-12, coupler_noise_p=.6e-12).items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)
        if any(not math.isfinite(v) or v <= 0 for v in (self.R, self.C, self.v_dd)):
            raise ValueError('R, C and v_dd must be finite and positive.')
        for m in self.active_convolutions():
            if (m.groups != 1 or m.dilation != (1, 1) or m.padding_mode != 'zeros'
                    or m.stride[0] != m.stride[1] or m.padding[0] != m.padding[1]):
                raise ValueError('TC CNN unrolling supports ordinary undilated square-stride convolutions only.')
        if self.w_bits != 5 or self.weight_quant_factor_bits not in (None, -1):
            raise ValueError('TC CNN requires 5-bit weights and no scale-factor quantization.')
        if (self.enob is not None or self.noise_level not in (None, 0) or
                self.enable_dtc_nonideality or self.enable_slow_coupler_noise or
                self.enable_slow_summing_current):
            raise ValueError('TC CNN excludes ENOB, scalar mismatch, DTC and slow noise.')
        if not math.isclose(self.R, 1e4):
            raise ValueError('This TC CNN mapping uses R=10k and R_max=150k; other code grids are not implemented.')
        if tc_method not in ('dopri5', 'euler') or tc_tol <= 0:
            raise ValueError('Use dopri5 or euler with a positive tolerance.')
        if tc_step_size is not None and tc_step_size <= 0:
            raise ValueError('tc_step_size must be positive physical seconds.')
        if tc_method == 'euler' and tc_step_size is None:
            raise ValueError('Euler requires tc_step_size in physical seconds.')
        if tc_noise_reference_R <= 0:
            raise ValueError('tc_noise_reference_R must be positive.')
        self.one_shot_conv = one_shot_conv
        self.tc_method, self.tc_tol, self.tc_step_size = tc_method, tc_tol, tc_step_size
        self.tc_noise_reference_R = tc_noise_reference_R
        self.tc_covariance_table = tc_covariance_table
        self.tc_curve_sampling = tc_curve_sampling
        self._tc_generators = {}
        self._tc_samples = {}

    # Reuse TC's named, layer-separated RNG streams, not new distributions.
    _tc_generator = ODEBlockPC._tc_generator
    _tc_nominal_sum = ODEBlockPC._tc_nominal_sum

    @staticmethod
    def unroll_convolution(input_shape, kernel, stride=1, padding=0):
        # The existing trimmed builder has correct oc/spatial/ic ordering also
        # when padding=0; do not change the legacy padded builder for PCN/toggle.
        from validation import build_unrolled_csr_trimmed
        ci, h, w = input_shape
        co, _, kh, kw = kernel.shape
        ho, wo = (h+2*padding-kh)//stride+1, (w+2*padding-kw)//stride+1
        mat = build_unrolled_csr_trimmed(ci,h,w,co,kh,kw,ho,wo,ho*wo,
                                        stride,padding,kernel)
        return mat, None, None

    def reset_nonlinear_R_variation(self):
        super().reset_nonlinear_R_variation()
        self._tc_samples.clear()

    def _current(self, module, source):
        if hasattr(module, 'mat'):
            current = module(source)
        else:
            package = getattr(self, '_tc_resistance_package', None)
            if package is not None:
                if package.means.device != source.device or package.means.dtype != source.dtype:
                    package = replace(package, **{f.name: getattr(package, f.name).to(source)
                        for f in fields(package) if torch.is_tensor(getattr(package, f.name))})
                    self._tc_resistance_package = package
                    self._tc_samples.clear()
                key = self._module_key(module)
                if self.training or key not in self._tc_samples:
                    seed = getattr(self, '_tc_curve_seed', 4096)
                    self._tc_samples[key] = package.sample_shared(
                        self._values_to_level_idx(module.weight.detach()),
                        sampling=self.tc_curve_sampling,
                        generator=self._tc_generator(source, 'curve:'+key, seed))
                curve = self._tc_samples[key]
                if curve is not None:
                    source = shared_correction(source, curve, package.v_grid,
                                               self.v_dd, package.floor_ohms)
            current = F.conv2d(source, module.weight, None, module.stride,
                               module.padding, module.dilation, module.groups)
        # The bias circuit is ideal, but its fixed current enters the same spin
        # loop.  Its integrated physical-domain contribution is q*b.
        bias = self._module_bias(module)
        if bias is not None:
            stage = 'z' if module is self.conv1 else 'y'
            scale = self._effective_stage_scale(current, stage)
            current = current + (scale * self.q * bias).view(1, -1, 1, 1)
        return current

    def _noise_context(self, module, source, state, stage):
        cap = self._stage_capacitance(stage)
        ratio = math.sqrt(self.tc_noise_reference_R / self.R)
        summed = self._tc_nominal_sum(module, source)
        a = self.summing_current_p if self.enable_summing_current_noise else 0.
        b = self.coupler_noise_p if self.enable_coupler_noise else 0.
        coefficients = (torch.full_like(summed, a*ratio/cap), summed.sqrt()*b*ratio/cap)
        generators = {}
        for name, seed in (('sum', self.summing_noise_seed), ('coupler', self.coupler_noise_seed)):
            generators[(0, name)] = self._tc_generator(
                state, stage+':'+name, 4096 if seed is None else seed)
        return TCNoiseLifecycle([coefficients], generators), torch.hypot(*coefficients)

    def _run_stage(self, module, source, stage):
        if self._capture_dense_modules:
            return module(source)
        if not self.physical:
            return super()._run_stage(module, source, stage)
        duration = self._stage_duration(source, stage)
        state = self._output_zeros(module, source)
        drift = self._apply_spin_variation(stage, self._current(module, source))
        drift = drift / (self.R * self._stage_capacitance(stage))
        context, eps = self._noise_context(module, source, state, stage)
        if self.one_shot_conv:
            return self.project_state(duration*drift + duration.sqrt()*eps*context.normal(state, 0))
        def rhs(t, v):
            return drift
        rhs.tc_context = context
        # Same entry point, additive update, projection and accepted-step replay as TC PCN.
        options = dict(t0=0., t1=float(duration), t_eval=[float(duration)],
                       method=self.tc_method, rtol=self.tc_tol, atol=self.tc_tol,
                       h=self.tc_step_size, eps=eps, noise_type='addi',
                       proj_fn=self.project_state)
        return odesolve(rhs, state, options)


class TCPhysicalCIFARBasicBlock(TCPhysicalBasicBlock):
    """TC convolution stages with CIFAR ResNet-v1 post-activation ordering."""

    def forward(self, x, layer_idx=None):
        x = self._prepare_block_input(x)
        residual = x
        out = self._run_stage(self.conv1, x, 'z')
        out = self.act1(self.norm1(out))
        out = self._run_stage(self.conv2, out, 'y')
        out = self.norm2(out)
        if self.shortcut is not None:
            out = out + self.shortcut(residual)
            if self.physical and not self._capture_dense_modules:
                out = self.project_state(out)
        out = self.post_add_pool(out)
        out = self.act2(out)
        return self._finalize_block_output(out)


class TCFeedForwardPhysicalWrapper(AveragedFeedForwardPhysicalWrapper):
    """Reuse normalized QAT and checkpoint scales; replace curve preparation only."""

    def configure_nonlinear_R_training(self, nonlinear_R_table, mode='exact_curve',
                                     curve_seed=None, **kwargs):
        ref = self.block.conv1.weight
        package = prepare_tc_resistance_curves(
            nonlinear_R_table, self.block.tc_covariance_table,
            levels=self.block._get_quant_magnitude_levels(ref), R=self.R,
            R_max=self.R_max, device=ref.device, dtype=ref.dtype)
        result = dict(tc_curve_package=package, nonlinear_R_curve_seed=curve_seed)
        self.install_nonlinear_R_training_package(result)
        return result

    def install_nonlinear_R_training_package(self, package):
        self.block._tc_resistance_package = package['tc_curve_package']
        seed = package.get('nonlinear_R_curve_seed')
        self.block._tc_curve_seed = 4096 if seed is None else seed
        self.block.reset_nonlinear_R_variation()

    def configure_nonlinear_R_inference(self, nonlinear_R_table, curve_seed=None,
                                      curve_edge_chunk_size=65536, **kwargs):
        result = self.configure_nonlinear_R_training(nonlinear_R_table, curve_seed=curve_seed)
        result.update(v_grid=None, R_codes=None, R_left=None, R_slope=None,
                      proj_fn=self.proj_fn, R=self.R,
                      nonlinear_R_curve_edge_chunk_size=curve_edge_chunk_size)
        self.install_nonlinear_R_inference_package(result)
        return result
