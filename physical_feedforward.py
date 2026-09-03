"""Physical execution blocks for pre-activation feedforward CNNs.

This module is intentionally separate from the PCN conversion path.  It reuses
the level-2/level-3 physical convolution, noise, DTC, and nonlinear-resistance
machinery from :mod:`ode_pc`, but gives the stages feedforward residual-block
semantics rather than the recurrent FF/FB toggle semantics.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from ode_pc import (
    OutputQuantImpl,
    PulseSymQuantizeWeight,
    SymQuantizeWeight,
    ToggleAveragedPhysicalFFFB,
    TogglePulseBlk,
    TogglePulseWrapper1State,
    WrapQuantizeW,
    _symmetric_qat_weight_scale,
)
from utils import (
    load_mc_res_curve_bank,
    load_mc_res_curve_gaussian,
    load_mc_res_training_curve_bank,
    load_res_vs_vin,
)


class AvgPoolChannelPad(nn.Module):
    """Parameter-free residual shortcut: optional average pool, then zero-pad."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("AvgPoolChannelPad cannot reduce channel count.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.pool = (
            nn.Identity() if self.stride == 1 else
            nn.AvgPool2d(kernel_size=self.stride, stride=self.stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        channel_diff = self.out_channels - x.shape[1]
        if channel_diff:
            left = channel_diff // 2
            right = channel_diff - left
            x = F.pad(x, (0, 0, 0, 0, left, right))
        return x


class StateScale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return x * self.scale


class AveragedPhysicalBasicBlock(ToggleAveragedPhysicalFFFB):
    """Level-2 averaged-current feedforward basic block.

    The constructor and the methods that name or sequence convolutions are
    feedforward-specific.  Terminology-neutral current/noise kernels are
    inherited from :class:`ToggleAveragedPhysicalFFFB`.
    """

    reset_z = True
    supports_post_quant_mismatch = True
    physical_level = 2

    def __init__(
            self, conv1: nn.Conv2d, conv2: Optional[nn.Conv2d] = None,
            norm1: Optional[nn.Module] = None,
            act1: Optional[nn.Module] = None,
            norm2: Optional[nn.Module] = None,
            act2: Optional[nn.Module] = None,
            between: Optional[nn.Module] = None,
            shortcut: Optional[nn.Module] = None,
            *, layer_idx: int = 0,
            R: float = 50e3, C: float = 500e-15, v_dd: float = 0.5,
            one_over_q: float = 1.0, physical: bool = True,
            w_bits: int = 5, weight_quant_factor_bits: Optional[int] = None,
            toggle_timing_mode: str = "derived", toggle_y_time: float = 5e-9,
            z_over_y_time: float = 1.0, enob=None,
            noise_level=0.0, mismatch_type: str = "mul",
            enable_spin_variation: bool = False, sigma_spin: float = 0.10,
            spin_variation_mean: float = 1.0, spin_variation_seed=None,
            enable_summing_current_noise: bool = False,
            summing_current_p: float = 12.73e-12, summing_noise_seed=None,
            enable_coupler_noise: bool = False,
            coupler_noise_p: float = 0.6e-12, coupler_noise_seed=None,
            enable_slow_summing_current: bool = False,
            slow_summing_current: float = 2.47e-9,
            enable_slow_coupler_noise: bool = False,
            slow_coupler_noise: float = 2.47e-9,
            enable_dtc_nonideality: bool = False,
            dtc_leading_edge_variation_std: float = 0.0,
            dtc_width_variation_mean: float = 0.0,
            dtc_width_variation_std: float = 0.018,
            dtc_leading_edge_jitter_std: float = 0.005,
            dtc_falling_edge_jitter_std: float = 0.005,
            dtc_timing_seed=None):
        # The PCN constructors require recurrent PCConv state.  CNN blocks have
        # a separate constructor while inheriting their numerical kernels.
        nn.Module.__init__(self)
        if not isinstance(conv1, nn.Conv2d):
            raise TypeError("conv1 must be nn.Conv2d.")
        if conv2 is not None and not isinstance(conv2, nn.Conv2d):
            raise TypeError("conv2 must be nn.Conv2d or None.")
        if toggle_timing_mode not in ("derived", "fixed"):
            raise ValueError("toggle_timing_mode must be derived or fixed.")
        if mismatch_type not in ("mul", "add"):
            raise ValueError("mismatch_type must be mul or add.")

        self.conv1 = conv1
        self.conv2 = conv2
        self.norm1 = nn.Identity() if norm1 is None else norm1
        self.act1 = nn.Identity() if act1 is None else act1
        self.norm2 = nn.Identity() if norm2 is None else norm2
        self.act2 = nn.Identity() if act2 is None else act2
        self.between = nn.Identity() if between is None else between
        self.shortcut = shortcut

        self.layer_idx = int(layer_idx)
        self.physical = bool(physical)
        self.R = float(R)
        self.C = float(C)
        self.capacitance1 = self.C
        self.capacitance2 = self.C
        self.v_dd = float(v_dd)
        self.one_over_q = float(one_over_q)
        if self.one_over_q <= 0:
            raise ValueError("one_over_q must be positive.")
        self.q = self.v_dd / self.one_over_q
        self.input_scale = 1.0
        self.output_scale = 1.0
        self.enob = enob
        self.w_bits = int(w_bits)
        self.q_hi = (1 << (self.w_bits - 1)) - 1
        self.weight_scale = 1.0
        self.weight_quant_factor_bits = weight_quant_factor_bits
        self.toggle_timing_mode = str(toggle_timing_mode)
        self.toggle_y_time = float(toggle_y_time)
        self.z_over_y_time = float(z_over_y_time)
        if self.toggle_y_time <= 0 or self.z_over_y_time <= 0:
            raise ValueError("toggle_y_time and z_over_y_time must be positive.")
        self.toggle_fast_path = True
        self.noise_level = noise_level
        self.mismatch_type = mismatch_type
        self.tie_weights = False
        self.tie_bp = True
        self.bypass = None
        self._capture_dense_modules = False
        self._uses_quantized_weight_scale = False

        self.enable_spin_variation = bool(enable_spin_variation)
        self.sigma_spin = float(sigma_spin)
        self.spin_variation_mean = float(spin_variation_mean)
        self.spin_variation_seed = spin_variation_seed
        self.enable_summing_current_noise = bool(enable_summing_current_noise)
        self.summing_current_p = float(summing_current_p)
        self.summing_noise_seed = summing_noise_seed
        self.enable_coupler_noise = bool(enable_coupler_noise)
        self.coupler_noise_p = float(coupler_noise_p)
        self.coupler_noise_seed = coupler_noise_seed
        self.enable_slow_summing_current = bool(enable_slow_summing_current)
        self.slow_summing_current = float(slow_summing_current)
        self.enable_slow_coupler_noise = bool(enable_slow_coupler_noise)
        self.slow_coupler_noise = float(slow_coupler_noise)

        self.enable_dtc_nonideality = bool(enable_dtc_nonideality)
        self.dtc_leading_edge_variation_std = float(dtc_leading_edge_variation_std)
        self.dtc_width_variation_mean = float(dtc_width_variation_mean)
        self.dtc_width_variation_std = float(dtc_width_variation_std)
        self.dtc_leading_edge_jitter_std = float(dtc_leading_edge_jitter_std)
        self.dtc_falling_edge_jitter_std = float(dtc_falling_edge_jitter_std)
        self.dtc_timing_seed = dtc_timing_seed

        self.register_buffer("_spin_factor_y", None, persistent=False)
        self.register_buffer("_spin_factor_z", None, persistent=False)
        self.register_buffer("scale1", torch.tensor(1.0))
        self.register_buffer("scale2", torch.tensor(1.0))
        self._spin_variation_generators = {}
        self._summing_noise_generators = {}
        self._coupler_noise_generators = {}
        self._slow_summing_current_generators = {}
        self._slow_coupler_noise_generators = {}
        self._slow_summing_current_samples = {}
        self._slow_coupler_noise_samples = {}
        self._slow_coupler_count_cache = {}
        self._active_coupler_count_cache = {}
        self._nonlinear_R_training_generators = {}
        self._nonlinear_R_training_curve_indices = {}
        self._training_pulse_mismatch = None
        self._pulse_on_values = {}
        self._dtc_fixed_variation = {}
        self._dtc_timing_generators = {}
        self._last_dtc_window = {}
        self.clean_params = {
            "conv1": nn.Parameter(
                self.conv1.weight.detach().clone(), requires_grad=False)}
        if self.conv2 is not None:
            self.clean_params["conv2"] = nn.Parameter(
                self.conv2.weight.detach().clone(), requires_grad=False)

    def active_convolutions(self) -> Iterable[nn.Module]:
        yield self.conv1
        if self.conv2 is not None:
            yield self.conv2

    def _stage_scale(self, stage: str):
        return self.scale1 if stage == "z" else self.scale2

    def _stage_capacitance(self, stage: str):
        return self.capacitance1 if stage == "z" else self.capacitance2

    def _stage_duration(self, source: torch.Tensor, stage: str):
        scale = self._stage_scale(stage).to(source)
        if not self._uses_quantized_weight_scale:
            scale = torch.ones_like(scale)
        # TogglePulseBlk's reusable helpers call the first stored convolution
        # "z" and the second "y".  Feedforward semantics are the opposite:
        # conv1 is the y computation and conv2 is the z computation.  Keep the
        # inherited internal labels, but apply the external T_z/T_y ratio to
        # conv2 (internal "y"), not conv1.
        stage_ratio = 1.0 if stage == "z" else self.z_over_y_time
        if not self.physical:
            if self.toggle_timing_mode == "derived":
                return source.new_tensor(1.0) / scale
            return (source.new_tensor(self.toggle_y_time * stage_ratio) /
                    source.new_tensor(self.R * self._stage_capacitance(stage)) /
                    scale)
        if self.toggle_timing_mode == "fixed":
            return source.new_tensor(self.toggle_y_time * stage_ratio) / scale
        return source.new_tensor(self.R * self._stage_capacitance(stage)) / scale

    def _output_zeros(self, module, source):
        if hasattr(module, "mat"):
            meta = module.meta
            h = (source.shape[-2] + 2 * int(meta["padding"]) -
                 int(meta["ker_h"])) // int(meta["stride"]) + 1
            w = (source.shape[-1] + 2 * int(meta["padding"]) -
                 int(meta["ker_w"])) // int(meta["stride"]) + 1
            return source.new_zeros(
                source.shape[0], int(meta["out_chan"]), h, w)
        h = ((source.shape[-2] + 2 * module.padding[0] -
              module.dilation[0] * (module.kernel_size[0] - 1) - 1) //
             module.stride[0] + 1)
        w = ((source.shape[-1] + 2 * module.padding[1] -
              module.dilation[1] * (module.kernel_size[1] - 1) - 1) //
             module.stride[1] + 1)
        return source.new_zeros(source.shape[0], module.out_channels, h, w)

    def _module_key(self, module):
        if module is self.conv1:
            return "conv1"
        if module is self.conv2:
            return "conv2"
        raise ValueError("Unknown feedforward physical convolution.")

    def _nonlinear_R_training_module_key(self, module):
        return self._module_key(module)

    def _sample_nonlinear_R_training_curve(self, ref, module_key):
        package = self._nonlinear_R_training_pkg
        if not self._nonlinear_R_training_curve_indices:
            keys = ["conv1"]
            if self.conv2 is not None:
                keys.append("conv2")
            n_curves = int(package["v_grid"].shape[0])
            generator = self._nonlinear_R_training_generator(ref)
            if n_curves >= len(keys):
                sampled = torch.randperm(
                    n_curves, device=ref.device,
                    generator=generator)[:len(keys)].tolist()
            else:
                sampled = torch.randint(
                    n_curves, (len(keys),), device=ref.device,
                    generator=generator).tolist()
            self._nonlinear_R_training_curve_indices.update(
                zip(keys, map(int, sampled)))
        return self._nonlinear_R_training_curve_indices[module_key]

    def _clean_weight(self, module):
        key = self._module_key(module)
        if not hasattr(module, "mat") and P.is_parametrized(module, "weight"):
            return key, module.weight
        if hasattr(module, "mat"):
            if not hasattr(module, "clean_mat_values"):
                raise ValueError("Expanded modules require clean_mat_values.")
            return key, module.clean_mat_values
        return key, self.clean_params[key]

    def begin_training_pulse_mismatch(self, noise_level, mismatch_type):
        mismatch = {"type": mismatch_type}
        for module in self.active_convolutions():
            key = self._module_key(module)
            noise = torch.randn_like(module.weight) * float(noise_level)
            mismatch[key] = (
                1.0 + noise if mismatch_type == "mul" else noise)
        self._training_pulse_mismatch = mismatch

    def _apply_averaged_module(self, module, source):
        if hasattr(module, "mat"):
            return module(source)
        source = self._averaged_nonlinear_R_source(module, source)
        scale = self._stage_scale("z" if module is self.conv1 else "y")
        weight = module.weight
        mismatch = self._training_pulse_mismatch
        if mismatch is not None:
            perturbation = mismatch[self._module_key(module)]
            if mismatch["type"] == "mul":
                weight = weight * perturbation
            else:
                weight = weight + weight.abs() * perturbation
        bias = None if module.bias is None else module.bias * scale
        return F.conv2d(
            source, weight, bias, module.stride, module.padding,
            module.dilation, module.groups)

    def _run_averaged_stage(self, module, source, stage, duration):
        state = self._output_zeros(module, source)
        level_sum = None
        if self.enable_coupler_noise:
            level_sum = self._averaged_coupler_level_sum(
                module, source, state)
        updated = self._averaged_stage_update(
            state, self._apply_averaged_module(module, source), duration,
            stage, coupler_level_sum=level_sum,
            slow_current=self._slow_stage_current(
                state, stage, module, source))
        return self.project_state(updated) if self.physical else updated

    def _run_stage(self, module, source, stage):
        if self._capture_dense_modules:
            return module(source)
        duration = self._stage_duration(source, stage)
        return self._run_averaged_stage(module, source, stage, duration)

    def forward(self, x, layer_idx=None):
        self.reset_slow_current_noise()
        if self.training and self.enable_spin_variation:
            self.reset_spin_variation()
        if self.training and hasattr(self, "_nonlinear_R_training_pkg"):
            self.reset_nonlinear_R_variation()
        x = x * self.input_scale
        if self.physical and not self._capture_dense_modules:
            x = self.project_state(x)
        residual = x
        out = self._run_stage(
            self.conv1, self.act1(self.norm1(x)), "z")
        if self.conv2 is not None:
            out = self.between(self.act2(self.norm2(out)))
            out = self._run_stage(self.conv2, out, "y")
            if self.shortcut is not None:
                out = out + self.shortcut(residual)
                if self.physical and not self._capture_dense_modules:
                    out = self.project_state(out)
        if self.physical and not self._capture_dense_modules and self.enob is not None:
            if self.training:
                out = OutputQuantImpl.apply(out, self.v_dd, int(self.enob))
            else:
                intervals = (1 << int(self.enob)) - 1
                if intervals > 0:
                    step = 2.0 * self.v_dd / intervals
                    out = (((out.clamp(-self.v_dd, self.v_dd) + self.v_dd) /
                            step).round() * step - self.v_dd)
                    out = out.clamp(-self.v_dd, self.v_dd)
        return out / self.output_scale


class PulsePhysicalBasicBlock(AveragedPhysicalBasicBlock, TogglePulseBlk):
    """Level-3 pulse-sliced feedforward basic block."""

    physical_level = 3
    supports_post_quant_mismatch = True

    def _pulse_stage_rhs(
            self, module, source, state, pulse_weight, stage):
        summed_current = self._apply_pulse_module(
            module, source, pulse_weight)
        summed_current = self._apply_spin_variation(stage, summed_current)
        rhs = summed_current / (
            self.R * self._stage_capacitance(stage))
        slow_current = self._slow_stage_current(
            state, stage, module, source)
        return (rhs if slow_current is None else
                rhs + slow_current / self._stage_capacitance(stage))

    def _run_pulse_stage(self, module, source, stage, duration):
        if getattr(module, "bias", None) is not None:
            raise NotImplementedError(
                "Level-3 physical feedforward convolution bias requires a "
                "dedicated constant-input coupler stage. Use bias-free convs.")
        state = self._output_zeros(module, source)
        num_slices = self._num_slices(stage)
        dt = duration / float(num_slices)
        dtc_window = self._sample_dtc_window(module, stage)
        for slice_idx in range(num_slices):
            pulse_weight = self.get_pulse_matrix(
                module, slice_idx, num_slices, dtc_window=dtc_window)
            active_coupler_count = None
            if self.enable_coupler_noise:
                if dtc_window is None:
                    active_mask = self._active_pulse_mask(module, slice_idx)
                else:
                    _, clean_weight = self._clean_weight(module)
                    active_mask = self._dtc_slice_duty(
                        module, clean_weight, slice_idx, dtc_window)
                active_coupler_count = self._active_coupler_count(
                    module, source, active_mask, state,
                    slice_idx=slice_idx if dtc_window is None else None)
            constant_rhs = (
                self._pulse_stage_rhs(
                    module, source, state, pulse_weight, stage)
                if self.toggle_fast_path else None)
            rhs_fn = lambda t, current, pw=pulse_weight: self._pulse_stage_rhs(
                module, source, current, pw, stage)
            state = self.integrate_pulse_slice(
                state, dt, rhs_fn, stage, slice_idx,
                constant_rhs=constant_rhs,
                active_coupler_count=active_coupler_count)
        return state

    def _run_stage(self, module, source, stage):
        if self._capture_dense_modules:
            return module(source)
        duration = self._stage_duration(source, stage)
        return self._run_pulse_stage(module, source, stage, duration)


class FeedForwardPhysicalWrapper(WrapQuantizeW):
    """CNN wrapper reusing PCN table loading and quantization primitives.

    Its constructor is intentionally CNN-specific; PCN wrapper construction
    patches recurrent ODE methods that do not exist on a feedforward block.
    """

    def __init__(self, block, *, qat=False, quantize_weights=False):
        nn.Module.__init__(self)
        self.ode_block = block
        self.w_bits = block.w_bits
        self.q_hi = block.q_hi
        self.weight_quant_factor_bits = block.weight_quant_factor_bits
        self.R = block.R
        self.R_max = self.R * self.q_hi
        self.proj_fn = nn.Hardtanh(
            min_val=-self.block.v_dd, max_val=self.block.v_dd)
        self.mul_mismatch_mode = "static_mismatch"
        # Parametrization registers each quantizer below its convolution.  A
        # plain lookup avoids registering the same module a second time here.
        self.quantizers = {}
        self._qat_enabled = False
        self._refresh_scales_from_weights()
        if qat:
            self.enable_qat_()
        elif quantize_weights:
            self.quantize_weights_()

    @property
    def block(self):
        return self.ode_block

    def active_convolutions(self):
        return self.block.active_convolutions()

    quantizer_class = None

    def _install_qat(self):
        if self.quantizer_class is None:
            raise TypeError("Use an averaged or pulse feedforward wrapper.")
        for name, module in (("conv1", self.block.conv1),
                             ("conv2", self.block.conv2)):
            if module is None:
                continue
            quantizer = self.quantizer_class(
                w_bits=self.w_bits,
                weight_quant_factor_bits=self.weight_quant_factor_bits)
            quantizer.compute_s(module.weight)
            self.quantizers[name] = quantizer
            P.register_parametrization(module, "weight", quantizer)
            getattr(self.block, "scale1" if name == "conv1" else "scale2").copy_(
                quantizer.s_w)

    def enable_qat_(self):
        if not self._qat_enabled:
            self._install_qat()
            self._qat_enabled = True
        self.block._uses_quantized_weight_scale = True
        return self

    def _refresh_scales_from_weights(self):
        for name, module in (("conv1", self.block.conv1),
                             ("conv2", self.block.conv2)):
            if module is None:
                continue
            scale = _symmetric_qat_weight_scale(
                module.weight, self.weight_quant_factor_bits)
            getattr(self.block, "scale1" if name == "conv1" else "scale2").copy_(scale)

    def _refresh_qat_scales(self):
        for name, module in (("conv1", self.block.conv1),
                             ("conv2", self.block.conv2)):
            if module is None or name not in self.quantizers:
                continue
            if not P.is_parametrized(module, "weight"):
                continue
            quantizer = self.quantizers[name]
            quantizer.compute_s(module.parametrizations.weight.original)
            getattr(self.block, "scale1" if name == "conv1" else "scale2").copy_(
                quantizer.s_w)

    @torch.no_grad()
    def quantize_weights_(self):
        for name, module in (("conv1", self.block.conv1),
                             ("conv2", self.block.conv2)):
            if module is None:
                continue
            scale = self._quantize_module(module)
            getattr(self.block, "scale1" if name == "conv1" else "scale2").copy_(scale)
        self.block._uses_quantized_weight_scale = True
        return self

    def _quantize_module(self, module):
        raise NotImplementedError

    @torch.no_grad()
    def add_noise(self):
        self.block._pulse_on_values = {}
        sigma = self.block.noise_level
        if isinstance(sigma, dict):
            sigma = sigma.get(self.q_hi, max(sigma.values()))
        sigma = 0.0 if sigma is None else float(sigma)
        for module in self.active_convolutions():
            if hasattr(module, "add_noise"):
                module.add_noise(
                    noise_level=sigma, mismatch_type=self.block.mismatch_type,
                    q_hi=self.q_hi, weight_scale=self.block.weight_scale)
            elif sigma > 0.0:
                mismatch = torch.randn_like(module.weight) * sigma
                if self.block.mismatch_type == "mul":
                    module.weight.mul_(1.0 + mismatch)
                else:
                    module.weight.add_(mismatch * module.weight.abs().max())

    def _csv_prepare_table(self):
        """CNN adaptation of the PCN wrapper's nonlinear-R table loader."""
        device = self.block.conv1.weight.device
        if (self.nonlinear_R_curve_sharing != "shared" or
                self.nonlinear_R_curve_bank_indices is not None):
            if self.nonlinear_R_table is None:
                raise ValueError(
                    "Sampled nonlinear-R curves require nonlinear_R_table.")
            if self.nonlinear_R_curve_sampling == "multivariate_gaussian":
                self.nonlinear_R_curve_gaussian = load_mc_res_curve_gaussian(
                    self.nonlinear_R_table,
                    quantity=self.nonlinear_R_mc_quantity,
                    curve_indices=self.nonlinear_R_curve_bank_indices,
                    device=device)
            else:
                self.nonlinear_R_curve_bank = load_mc_res_curve_bank(
                    self.nonlinear_R_table,
                    quantity=self.nonlinear_R_mc_quantity,
                    curve_indices=self.nonlinear_R_curve_bank_indices,
                    device=device)
            shared_curve_index = (
                0 if self.nonlinear_R_curve_bank_indices is None else
                int(self.nonlinear_R_curve_bank_indices[0]))
        else:
            shared_curve_index = self.nonlinear_R_mc_curve_index

        self.v_grid, self.R_codes, self.R_table = load_res_vs_vin(
            R=self.R, R_max=self.R_max, device=device,
            nonlinear_R_table=self.nonlinear_R_table,
            nonlinear_R_mc_curve_index=shared_curve_index,
            nonlinear_R_mc_quantity=self.nonlinear_R_mc_quantity)
        voltage_order = torch.argsort(self.v_grid)
        self.v_grid = self.v_grid[voltage_order]
        self.R_table = self.R_table[voltage_order, :]
        resistance_order = torch.argsort(self.R_codes)
        self.R_codes = self.R_codes[resistance_order]
        self.R_table = self.R_table[:, resistance_order]

    def configure_nonlinear_R_training(
            self, nonlinear_R_table, mode="exact_curve", corner_range="all",
            quantity="conductance", curve_seed=None):
        """Prepare level-2 nonlinear-R data owned by this CNN wrapper."""
        if self.block.physical_level != 2:
            raise ValueError(
                "Nonlinear-R training requires an averaged physical block.")
        mode = str(mode).lower()
        if mode not in {"exact_curve", "mean"}:
            raise ValueError("mode must be exact_curve or mean.")
        bank = load_mc_res_training_curve_bank(
            nonlinear_R_table, mode=mode, corner_range=corner_range,
            quantity=quantity, dtype=self.block.conv1.weight.dtype,
            device=self.block.conv1.weight.device)
        package = {
            **bank,
            "R": self.R,
            "proj_fn": self.proj_fn,
            "nonlinear_R_train_mode": mode,
            "nonlinear_R_corner_range": corner_range,
            "nonlinear_R_curve_seed": curve_seed,
        }
        self.install_nonlinear_R_training_package(package)
        return package

    def install_nonlinear_R_training_package(self, package):
        self.block._nonlinear_R_training_pkg = dict(package, R=self.R)

    def configure_nonlinear_R_inference(
            self, nonlinear_R_table, quantity="conductance",
            curve_sharing="per_coupler",
            curve_sampling="empirical_with_replacement", curve_seed=None,
            curve_edge_chunk_size=65536, curve_indices=None,
            mul_mismatch_mode="static_mismatch"):
        """Prepare expanded-inference nonlinear-R data for this CNN wrapper."""
        self.nonlinear_R_table = nonlinear_R_table
        self.nonlinear_R_mc_quantity = quantity
        self.nonlinear_R_curve_sharing = curve_sharing
        self.nonlinear_R_curve_sampling = curve_sampling
        self.nonlinear_R_curve_seed = curve_seed
        self.nonlinear_R_curve_edge_chunk_size = int(curve_edge_chunk_size)
        self.nonlinear_R_curve_bank_indices = curve_indices
        self.nonlinear_R_mc_curve_index = None
        self.mul_mismatch_mode = mul_mismatch_mode
        self.nonlinear_R_curve_bank = None
        self.nonlinear_R_curve_gaussian = None
        self._csv_prepare_table()
        self._csv_build_interpolant()
        package = {
            "v_grid": self.v_grid,
            "R_codes": self.R_codes,
            "R": self.R,
            "R_left": self.R_left,
            "R_slope": self.R_slope,
            "proj_fn": self.proj_fn,
            "mul_mismatch_mode": self.mul_mismatch_mode,
            "curve_bank": self.nonlinear_R_curve_bank,
            "curve_gaussian": self.nonlinear_R_curve_gaussian,
            "nonlinear_R_curve_sampling": self.nonlinear_R_curve_sampling,
            "nonlinear_R_curve_sharing": self.nonlinear_R_curve_sharing,
            "nonlinear_R_curve_seed": self.nonlinear_R_curve_seed,
            "nonlinear_R_curve_edge_chunk_size": (
                self.nonlinear_R_curve_edge_chunk_size),
        }
        self.install_nonlinear_R_inference_package(package)
        return package

    def install_nonlinear_R_inference_package(self, package):
        self.block._nonlinear_R_pkg = dict(package, R=self.R)

    def forward(self, x, layer_idx=None):
        if self._qat_enabled:
            self._refresh_qat_scales()
        return self.block(x, layer_idx=layer_idx)


class AveragedFeedForwardPhysicalWrapper(FeedForwardPhysicalWrapper):
    """QAT/inference wrapper for a level-2 feedforward block."""

    quantizer_class = SymQuantizeWeight

    def _quantize_module(self, module):
        return self.cal_quant_factor_and_set(
            self.q_hi, module.weight, self.weight_quant_factor_bits)


class PulseFeedForwardPhysicalWrapper(FeedForwardPhysicalWrapper):
    """QAT/inference wrapper for a level-3 feedforward block."""

    quantizer_class = PulseSymQuantizeWeight

    def _quantize_module(self, module):
        return TogglePulseWrapper1State.cal_quant_factor_and_set(
            self.q_hi, module.weight, self.weight_quant_factor_bits)


def iter_physical_blocks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (AveragedPhysicalBasicBlock,
                               PulsePhysicalBasicBlock)):
            yield module


def iter_physical_wrappers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FeedForwardPhysicalWrapper):
            yield module


@torch.no_grad()
def prepare_flattened_qat_for_pulse_inference(model: nn.Module):
    """Convert baked level-2 QAT weights to level-3 signed pulse levels."""
    for wrapper in iter_physical_wrappers(model):
        block = wrapper.block
        for name, module in (("conv1", block.conv1),
                             ("conv2", block.conv2)):
            if module is None:
                continue
            module.weight.mul_(block.q_hi)
            block.clean_params[name].copy_(module.weight)
        block._uses_quantized_weight_scale = True
    return model


def _hardware_shortcut(conv1, conv2):
    in_channels = int(conv1.in_channels)
    out_channels = int(conv2.out_channels)
    stride = int(conv2.stride[0])
    if stride == 1 and in_channels == out_channels:
        return nn.Identity()
    return AvgPoolChannelPad(in_channels, out_channels, stride=stride)


def convert_wide_resnet_to_physical(
        model: nn.Module, activation_factory=None, **physical_kwargs):
    """Convert a registered pre-activation WideResNet in place.

    The floating-point checkpoint must be loaded before this function is
    called.  Learned projection shortcuts are intentionally replaced by the
    parameter-free hardware shortcut policy.
    """
    physical_level = int(physical_kwargs.pop("physical_level", 3))
    qat = bool(physical_kwargs.pop("qat", False))
    quantize_weights = bool(physical_kwargs.pop("quantize_weights", False))
    if physical_level == 2:
        block_cls = AveragedPhysicalBasicBlock
        wrapper_cls = AveragedFeedForwardPhysicalWrapper
    elif physical_level == 3:
        block_cls = PulsePhysicalBasicBlock
        wrapper_cls = PulseFeedForwardPhysicalWrapper
    else:
        raise ValueError("physical_level must be 2 or 3.")

    def wrap(block):
        return wrapper_cls(
            block, qat=qat, quantize_weights=quantize_weights)

    layer_idx = 0
    stem = getattr(model, "conv1", None)
    if not isinstance(stem, nn.Conv2d):
        raise TypeError("Expected a WideResNet-style model.conv1 stem.")
    model.conv1 = wrap(block_cls(
        stem, conv2=None, layer_idx=layer_idx, **physical_kwargs))
    layer_idx += 1

    for group_name in ("layer1", "layer2", "layer3", "layer4"):
        group = getattr(model, group_name, None)
        if group is None:
            continue
        for block_idx, block in enumerate(group):
            required = ("conv1", "conv2", "bn1", "bn2", "relu1", "relu2")
            if not all(hasattr(block, name) for name in required):
                raise TypeError(
                    "Only pre-activation two-convolution basic blocks are "
                    "currently supported; failed at {}[{}].".format(
                        group_name, block_idx))
            dropout_rate = float(getattr(block, "dropout_rate", 0.0))
            between = (
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity())
            act1 = (
                block.relu1 if activation_factory is None else
                activation_factory(layer_idx, "conv1"))
            act2 = (
                block.relu2 if activation_factory is None else
                activation_factory(layer_idx, "conv2"))
            replacement = block_cls(
                block.conv1, block.conv2,
                norm1=block.bn1, act1=act1,
                norm2=block.bn2, act2=act2,
                between=between,
                shortcut=_hardware_shortcut(block.conv1, block.conv2),
                layer_idx=layer_idx, **physical_kwargs)
            group[block_idx] = wrap(replacement)
            layer_idx += 1

    blocks = list(iter_physical_blocks(model))
    if physical_kwargs.get("physical", True):
        q = float(physical_kwargs.get("v_dd", 0.5)) / float(
            physical_kwargs.get("one_over_q", 1.0))
        blocks[0].input_scale = q
        model._physical_state_scale = q
        model.relu = nn.Sequential(model.relu, StateScale(1.0 / q))
    return model
