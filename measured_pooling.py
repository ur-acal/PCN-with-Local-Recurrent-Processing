import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_mc_res_curve_bank, load_mc_res_training_curve_bank


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.size(-1))


class MeasuredAvgPool2d(nn.Module):
    """Average pooling with one sampled curve per channel and pooling window."""

    uses_pool_id = True

    def __init__(self, kernel_size=None, stride=None, input_scale=1.0,
                 curve_path=None, quantity="conductance", curve_indices=None,
                 nominal_R=1.0, seed=None, training_curve_mode=None,
                 corner_range="all", curve_gaussian=None,
                 curve_sharing="per_window"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_scale = float(input_scale)
        self.nominal_R = float(nominal_R)
        self.curve_sharing = str(curve_sharing).lower()
        if self.curve_sharing not in {"per_window", "per_input"}:
            raise ValueError(
                "curve_sharing must be per_window or per_input.")
        self._curve_assignments = {}
        self._gaussian_curve_samples = {}
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(seed))

        if curve_path is not None and curve_gaussian is not None:
            raise ValueError(
                "Measured pooling accepts either a curve bank or a Gaussian.")
        self.curve_gaussian = curve_gaussian is not None
        self.enable_nonideality = (
            curve_path is not None or self.curve_gaussian)
        if self.curve_gaussian:
            required = {"v_grid", "mean", "factor", "value_scale", "quantity"}
            missing = required.difference(curve_gaussian)
            if missing:
                raise ValueError(
                    "Measured-pooling Gaussian is missing {}.".format(
                        sorted(missing)))
            self.register_buffer(
                "gaussian_v_grid", curve_gaussian["v_grid"], persistent=False)
            self.register_buffer(
                "gaussian_mean", curve_gaussian["mean"], persistent=False)
            self.register_buffer(
                "gaussian_factor", curve_gaussian["factor"], persistent=False)
            self.gaussian_value_scale = float(curve_gaussian["value_scale"])
            self.gaussian_quantity = str(curve_gaussian["quantity"]).lower()
        elif self.enable_nonideality:
            if training_curve_mode is None:
                curve_bank = load_mc_res_curve_bank(
                    curve_path, quantity=quantity, curve_indices=curve_indices)
            else:
                if curve_indices is not None:
                    raise ValueError(
                        "Training curve banks do not accept curve_indices.")
                curve_bank = load_mc_res_training_curve_bank(
                    curve_path, mode=training_curve_mode,
                    corner_range=corner_range, quantity=quantity)
            for name, value in curve_bank.items():
                self.register_buffer(name, value, persistent=False)

    def _pool_geometry(self, x):
        if self.kernel_size is None:
            kernel_size = x.shape[-2:]
            stride = kernel_size
        else:
            kernel_size = _pair(self.kernel_size)
            stride = kernel_size if self.stride is None else _pair(self.stride)
        output_h = (x.shape[-2] - kernel_size[0]) // stride[0] + 1
        output_w = (x.shape[-1] - kernel_size[1]) // stride[1] + 1
        return kernel_size, stride, output_h, output_w

    def _curve_assignment(self, x, pool_id, kernel_size, output_h, output_w):
        channels = x.shape[1]
        n_inputs = kernel_size[0] * kernel_size[1]
        key = (pool_id, channels, x.shape[-2], x.shape[-1], kernel_size,
               output_h, output_w)
        assignment = self._curve_assignments.get(key)
        if assignment is None or self.training:
            assignment = torch.randint(
                self.v_grid.shape[0],
                (channels, output_h * output_w,
                 1 if self.curve_sharing == "per_window" else n_inputs),
                generator=self._generator).to(x.device)
            self._curve_assignments[key] = assignment
        elif assignment.device != x.device:
            assignment = assignment.to(x.device)
            self._curve_assignments[key] = assignment
        return assignment

    def _sample_gaussian_curves(self, x, pool_id, kernel_size,
                                output_h, output_w):
        channels = x.shape[1]
        n_inputs = kernel_size[0] * kernel_size[1]
        key = (pool_id, channels, x.shape[-2], x.shape[-1], kernel_size,
               output_h, output_w)
        curves = self._gaussian_curve_samples.get(key)
        if curves is None or self.training:
            n_groups = channels * output_h * output_w
            if self.curve_sharing == "per_input":
                n_groups *= n_inputs
            n_points = self.gaussian_v_grid.numel()
            eps = torch.randn(
                (n_groups, n_points), generator=self._generator,
                dtype=self.gaussian_mean.dtype)
            eps = eps.to(device=self.gaussian_mean.device)
            sampled = (
                self.gaussian_mean + eps @ self.gaussian_factor.t())
            sampled = sampled * self.gaussian_value_scale
            positive_floor = (
                torch.finfo(sampled.dtype).eps * self.gaussian_value_scale)
            sampled.clamp_(min=positive_floor)
            if self.gaussian_quantity == "conductance":
                curves = sampled.reciprocal()
            elif self.gaussian_quantity == "resistance":
                curves = sampled
            else:
                raise ValueError(
                    "Measured-pooling Gaussian quantity is invalid.")
            self._gaussian_curve_samples[key] = curves
        elif curves.device != x.device:
            curves = curves.to(x.device)
            self._gaussian_curve_samples[key] = curves
        return curves

    def reset_measured_pooling(self):
        """Clear cached curves before one full-dataset inference run."""
        self._curve_assignments.clear()
        self._gaussian_curve_samples.clear()

    def _interpolate_resistance(self, values, curve_indices):
        grids = self.v_grid.index_select(0, curve_indices)
        lengths = self.lengths.index_select(0, curve_indices)
        last_grid = grids.gather(1, (lengths - 1).unsqueeze(1))
        query = torch.maximum(values, grids[:, :1])
        query = torch.minimum(query, last_grid)
        interval = torch.searchsorted(
            grids.contiguous(), query.contiguous(), right=True) - 1
        interval.clamp_(min=0)
        interval = torch.minimum(interval, (lengths - 2).unsqueeze(1))

        left = self.R_left.index_select(0, curve_indices).gather(1, interval)
        slope = self.R_slope.index_select(0, curve_indices).gather(1, interval)
        v_left = grids.gather(1, interval)
        return left + slope * (query - v_left)

    def _interpolate_gaussian_resistance(self, values, curves):
        grid = self.gaussian_v_grid
        query = torch.maximum(values, grid[:1])
        query = torch.minimum(query, grid[-1:])
        interval = torch.searchsorted(
            grid.contiguous(), query.contiguous(), right=True) - 1
        interval.clamp_(min=0, max=grid.numel() - 2)
        right_interval = interval + 1
        left = curves.gather(1, interval)
        right = curves.gather(1, right_interval)
        v_left = grid[interval]
        v_right = grid[right_interval]
        return left + (right - left) * (query - v_left) / (v_right - v_left)

    def forward(self, x, pool_id=None):
        physical_x = x * self.input_scale
        kernel_size, stride, output_h, output_w = self._pool_geometry(
            physical_x)
        if not self.enable_nonideality:
            pooled = F.avg_pool2d(
                physical_x, kernel_size=kernel_size, stride=stride)
            return pooled / self.input_scale

        batch_size, channels = physical_x.shape[:2]
        n_inputs = kernel_size[0] * kernel_size[1]
        patches = F.unfold(
            physical_x, kernel_size=kernel_size, stride=stride)
        patches = patches.view(
            batch_size, channels, n_inputs, output_h * output_w)
        patches = patches.permute(0, 1, 3, 2).contiguous()

        values = patches.reshape(batch_size, -1).t().contiguous()
        interpolation_values = values
        if self.curve_sharing == "per_window":
            n_windows = output_h * output_w
            interpolation_values = values.reshape(
                channels * n_windows, n_inputs * batch_size)

        if self.curve_gaussian:
            curves = self._sample_gaussian_curves(
                physical_x, pool_id, kernel_size, output_h, output_w)
            R_eff = self._interpolate_gaussian_resistance(
                interpolation_values, curves)
        else:
            curve_indices = self._curve_assignment(
                physical_x, pool_id, kernel_size,
                output_h, output_w).reshape(-1)
            R_eff = self._interpolate_resistance(
                interpolation_values, curve_indices)

        if self.curve_sharing == "per_window":
            R_eff = R_eff.reshape(
                channels * n_windows, n_inputs, batch_size)
            R_eff = R_eff.reshape(-1, batch_size)
        alpha = self.nominal_R / R_eff
        pooled = (values * alpha).t().reshape_as(patches).mean(dim=-1)
        pooled = pooled.view(batch_size, channels, output_h, output_w)
        return pooled / self.input_scale


def configure_measured_pooling(model, wrappers, enable_nonideality=False,
                               curve_path=None, quantity="conductance",
                               curve_indices=None, nominal_R=1.0, seed=None,
                               training_curve_mode=None, corner_range="all",
                               curve_gaussian=None):
    """Install physical-domain average pooling without changing ideal scaling."""
    if not wrappers:
        raise ValueError("Measured pooling requires wrapped ODE layers.")
    intermediate_scales = [float(wrapper.out_scale) for wrapper in wrappers[:-1]]
    if any(scale != 1.0 for scale in intermediate_scales):
        raise ValueError(
            "Measured pooling currently requires unit intermediate out_scale.")
    final_out_scale = float(wrappers[-1].out_scale)
    curve_path = curve_path if enable_nonideality else None
    curve_gaussian = curve_gaussian if enable_nonideality else None
    common = {
        "curve_path": curve_path,
        "curve_gaussian": curve_gaussian,
        "quantity": quantity,
        "curve_indices": curve_indices,
        "nominal_R": nominal_R,
        "training_curve_mode": training_curve_mode,
        "corner_range": corner_range,
    }
    device = next(model.parameters()).device

    model.global_avg_pool2d = MeasuredAvgPool2d(
        input_scale=final_out_scale, seed=seed, **common).to(device)
    model.global_avg_pool2d.train(model.training)
    if isinstance(model.max_pool2d, nn.AvgPool2d):
        model.max_pool2d = MeasuredAvgPool2d(
            kernel_size=model.max_pool2d.kernel_size,
            stride=model.max_pool2d.stride,
            seed=None if seed is None else int(seed) + 1,
            **common).to(device)
        model.max_pool2d.train(model.training)
    return model


def configure_feedforward_measured_pooling(
        model, *, enable_nonideality=False, curve_path=None,
        quantity="conductance", nominal_R=1.0, seed=None,
        training_curve_mode=None, corner_range="all", curve_indices=None,
        curve_gaussian=None):
    """Install measured pooling at feedforward global, main and shortcut pools."""
    from physical_feedforward import iter_physical_blocks

    common = dict(
        curve_path=curve_path if enable_nonideality else None,
        curve_gaussian=curve_gaussian if enable_nonideality else None,
        quantity=quantity, nominal_R=nominal_R,
        curve_indices=curve_indices,
        training_curve_mode=training_curve_mode,
        corner_range=corner_range)
    if not isinstance(getattr(model, "global_pool", None), nn.AdaptiveAvgPool2d):
        raise TypeError("Expected model.global_pool to be AdaptiveAvgPool2d.")
    final_scale = float(getattr(model, "_physical_state_scale", 1.0))
    model.global_pool = MeasuredAvgPool2d(
        input_scale=final_scale, seed=seed, **common)
    pool_index = 1
    for block in iter_physical_blocks(model):
        if isinstance(block.main_downsample, nn.AvgPool2d):
            pool = block.main_downsample
            block.main_downsample = MeasuredAvgPool2d(
                kernel_size=pool.kernel_size, stride=pool.stride,
                seed=None if seed is None else int(seed) + pool_index,
                **common)
            pool_index += 1
        shortcut_pool = getattr(block.shortcut, "pool", None)
        if isinstance(shortcut_pool, nn.AvgPool2d):
            block.shortcut.pool = MeasuredAvgPool2d(
                kernel_size=shortcut_pool.kernel_size,
                stride=shortcut_pool.stride,
                seed=None if seed is None else int(seed) + pool_index,
                **common)
            pool_index += 1
    return model
