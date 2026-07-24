import csv
import re

import torch
import torch.nn as nn


def _evaluate_piecewise_cubic(x, coefficients, span_left, span_width, controls, transform):
    flat = x.reshape(-1)
    coefficients = coefficients.to(device=x.device, dtype=x.dtype)
    span_left = span_left.to(device=x.device, dtype=x.dtype)
    span_width = span_width.to(device=x.device, dtype=x.dtype)
    position = (flat - span_left[0]) / span_width[0]
    span_offset = torch.floor(position).to(torch.long).clamp(
        min=0, max=span_left.numel() - 1)
    local_u = ((flat - span_left[span_offset]) / span_width[span_offset]).clamp(0, 1)

    local_coefficients = coefficients[controls.to(device=x.device)]
    polynomial = torch.matmul(
        transform.to(device=x.device, dtype=x.dtype),
        local_coefficients.unsqueeze(-1)).squeeze(-1)
    p0 = polynomial[:, 0][span_offset]
    p1 = polynomial[:, 1][span_offset]
    p2 = polynomial[:, 2][span_offset]
    p3 = polynomial[:, 3][span_offset]
    return (((p3 * local_u + p2) * local_u + p1) * local_u + p0).reshape_as(x)


_COMPILED_PIECEWISE_CUBIC = None


def _compiled_piecewise_cubic():
    global _COMPILED_PIECEWISE_CUBIC
    if _COMPILED_PIECEWISE_CUBIC is None:
        _COMPILED_PIECEWISE_CUBIC = torch.compile(
            _evaluate_piecewise_cubic, dynamic=True, fullgraph=True)
    return _COMPILED_PIECEWISE_CUBIC


class _SplineCoefficientBuffers(nn.Module):
    """Dictionary-like storage for fixed, device-aware spline coefficients."""

    def __init__(self, coefficients):
        super().__init__()
        for name, value in coefficients.items():
            self.register_buffer(name, value, persistent=False)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self._buffers


class CubicBSplineActivation(nn.Module):
    """Cubic B-spline activation fitted once from characterized voltage curves.

    The CSV axes are in volts. Runtime scaling uses s = v_dd / max(abs(Vin)),
    evaluates the spline at clamp(x / s), and returns s * Vout. There is no
    offset, so the voltage origin is preserved exactly. Fitted coefficients are
    fixed non-persistent buffers, not trainable parameters; gradients still
    propagate through the activation input during training. The buffers are
    rebuilt from the fixed CSV instead of being stored in model checkpoints.
    """

    degree = 3

    def __init__(self, curve_path, v_dd, corner="TT", num_parameters=10,
                 normalize_positive_endpoint=False, compile_evaluator=False):
        super().__init__()
        if num_parameters <= self.degree:
            raise ValueError("A cubic B-spline needs at least four control coefficients.")

        vin, curves, column_names = self._load_csv(curve_path)
        self.curve_path = str(curve_path)
        self.num_parameters = int(num_parameters)
        self.normalize_positive_endpoint = bool(normalize_positive_endpoint)
        self.compile_evaluator = bool(compile_evaluator)
        self.corner_columns = column_names
        self.corner_names = tuple(curves)

        self.register_buffer("vin_min", vin.min(), persistent=False)
        self.register_buffer("vin_max", vin.max(), persistent=False)
        self.register_buffer("v_char", vin.abs().max(), persistent=False)
        self.register_buffer("v_dd", torch.as_tensor(float(v_dd), dtype=vin.dtype), persistent=False)
        self.register_buffer("knots", self._make_open_uniform_knots(
            self.vin_min, self.vin_max, self.num_parameters, self.degree), persistent=False)
        span_transform, span_left, span_width, span_controls = self._make_span_polynomial_data()
        self.register_buffer("span_polynomial_transform", span_transform, persistent=False)
        self.register_buffer("span_left", span_left, persistent=False)
        self.register_buffer("span_width", span_width, persistent=False)
        self.register_buffer("span_control_indices", span_controls, persistent=False)

        design = self._basis(vin)
        coefficients = {}
        endpoint_scales = {}
        for name, values in curves.items():
            fit = torch.linalg.lstsq(design.to(torch.float64), values.to(torch.float64)).solution
            fit = fit.to(vin.dtype)
            endpoint_scale = fit.new_tensor(1.0)
            if self.normalize_positive_endpoint:
                fitted_endpoint = design[-1].matmul(fit)
                if fitted_endpoint <= 0:
                    raise ValueError(
                        "Cannot normalize corner {} with nonpositive fitted endpoint {}.".format(
                            name, fitted_endpoint.item()))
                endpoint_scale = self.v_char / fitted_endpoint
                fit = fit * endpoint_scale
            coefficients[name] = fit
            endpoint_scales[name] = float(endpoint_scale)
        self.coefficients = _SplineCoefficientBuffers(coefficients)
        self.endpoint_scales = endpoint_scales
        self.active_corner = self._resolve_corner(corner)
        # Ordinary transient attribute: it is intentionally absent from state_dict.
        self._coordinate_pullback_scale = None

    @staticmethod
    def _corner_name(column):
        name = column
        if name.lower().startswith("vout_"):
            name = name[5:]
        name = re.sub(r"\([^)]*\)", "", name).strip()
        return name.upper()

    @classmethod
    def _load_csv(cls, path):
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "Vin" not in rows[0]:
            raise ValueError("Measured activation CSV must contain a Vin column.")

        output_columns = [name for name in rows[0] if name.lower().startswith("vout")]
        if not output_columns:
            raise ValueError("Measured activation CSV must contain at least one Vout column.")

        vin = torch.tensor([float(row["Vin"]) for row in rows], dtype=torch.float32)
        order = torch.argsort(vin)
        vin = vin[order]
        curves = {}
        column_names = {}
        for column in output_columns:
            corner = cls._corner_name(column)
            if corner in curves:
                raise ValueError("Duplicate activation corner name: {}".format(corner))
            values = torch.tensor([float(row[column]) for row in rows], dtype=torch.float32)[order]
            curves[corner] = values
            column_names[corner] = column
        return vin, curves, column_names

    @staticmethod
    def _make_open_uniform_knots(x_min, x_max, n_coefficients, degree):
        n_interior = n_coefficients - degree - 1
        if n_interior > 0:
            interior = torch.linspace(x_min, x_max, n_interior + 2, dtype=x_min.dtype)[1:-1]
        else:
            interior = x_min.new_empty(0)
        return torch.cat((x_min.repeat(degree + 1), interior, x_max.repeat(degree + 1)))

    def _basis(self, x):
        flat = x.reshape(-1)
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        basis = ((flat[:, None] >= knots[:-1]) & (flat[:, None] < knots[1:])).to(x.dtype)
        for order in range(1, self.degree + 1):
            width = basis.shape[1] - 1
            left_den = knots[order:order + width] - knots[:width]
            right_den = knots[order + 1:order + 1 + width] - knots[1:1 + width]
            left_num = flat[:, None] - knots[:width]
            right_num = knots[order + 1:order + 1 + width] - flat[:, None]
            left = torch.where(left_den != 0, left_num / left_den.clamp_min(torch.finfo(x.dtype).tiny),
                               torch.zeros_like(left_num))
            right = torch.where(right_den != 0, right_num / right_den.clamp_min(torch.finfo(x.dtype).tiny),
                                torch.zeros_like(right_num))
            basis = left * basis[:, :width] + right * basis[:, 1:width + 1]

        at_right_endpoint = flat == knots[-1]
        if at_right_endpoint.any():
            endpoint_basis = torch.zeros_like(basis)
            endpoint_basis[:, -1] = 1
            basis = torch.where(at_right_endpoint[:, None], endpoint_basis, basis)
        return basis

    def _make_span_polynomial_data(self):
        """Precompute each knot span as a polynomial of its four controls."""
        last_control = self.num_parameters - 1
        spans = torch.arange(self.degree, last_control + 1)
        sample_u = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=self.knots.dtype)
        vandermonde = torch.stack(
            [torch.ones_like(sample_u), sample_u, sample_u.square(), sample_u.pow(3)], dim=1)
        transforms = []
        controls = []
        left_edges = []
        widths = []
        for span in spans.tolist():
            left = self.knots[span]
            width = self.knots[span + 1] - left
            control_idx = torch.arange(span - self.degree, span + 1)
            samples = left + width * sample_u
            local_basis = self._basis(samples)[:, control_idx]
            transform = torch.linalg.solve(
                vandermonde.to(torch.float64), local_basis.to(torch.float64)).to(self.knots.dtype)
            transforms.append(transform)
            controls.append(control_idx)
            left_edges.append(left)
            widths.append(width)
        return (torch.stack(transforms), torch.stack(left_edges), torch.stack(widths),
                torch.stack(controls))

    def _evaluate_local(self, x, coefficients):
        """Evaluate the exact cubic pieces without materializing a full basis."""
        evaluator = (
            _compiled_piecewise_cubic()
            if self.compile_evaluator and x.is_cuda
            else _evaluate_piecewise_cubic
        )
        return evaluator(
            x, coefficients, self.span_left, self.span_width,
            self.span_control_indices, self.span_polynomial_transform)

    def _resolve_corner(self, corner):
        normalized = self._corner_name(str(corner))
        aliases = {"REFERENCE": "TT", "REFERENCE_CORNER": "TT", "TYPICAL": "TT"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.coefficients:
            raise ValueError("Unknown activation corner {!r}; available corners are {}.".format(
                corner, ", ".join(self.corner_names)))
        return normalized

    def select_corner(self, corner):
        self.active_corner = self._resolve_corner(corner)

    def set_v_dd(self, v_dd):
        self.v_dd.fill_(float(v_dd))

    def set_coordinate_pullback_scale(self, scale):
        """Temporarily evaluate Phi(scale*x)/scale without refitting the spline."""
        if scale is None:
            self._coordinate_pullback_scale = None
            return
        if not torch.is_tensor(scale):
            scale = self.v_dd.new_tensor(float(scale))
        scale = scale.detach()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError("Measured-activation pullback scale must be one finite positive scalar.")
        self._coordinate_pullback_scale = scale

    def forward(self, x, corner=None):
        selected = self.active_corner if corner is None else self._resolve_corner(corner)
        pullback_scale = self._coordinate_pullback_scale
        if pullback_scale is not None:
            pullback_scale = pullback_scale.to(device=x.device, dtype=x.dtype)
            x = pullback_scale * x
        scale = self.v_dd.to(device=x.device, dtype=x.dtype) / self.v_char.to(device=x.device, dtype=x.dtype)
        x_char = (x / scale).clamp(
            min=self.vin_min.to(device=x.device, dtype=x.dtype),
            max=self.vin_max.to(device=x.device, dtype=x.dtype))
        coeff = self.coefficients[selected]
        y_char = self._evaluate_local(x_char, coeff)
        output = scale * y_char
        return output if pullback_scale is None else output / pullback_scale


class MeasuredReLU6Activation(CubicBSplineActivation):
    """Measured activation expressed in the unitless ReLU6 coordinate system.

    Both characterized axes are scaled by max_value / V_char. With the default
    max_value=6 and V_char=0.3 V, the axis scale is 20. Inputs below or above
    the scaled characterization range use the corresponding endpoint value
    through the base class's endpoint clamping.

    When normalize_positive_endpoint is false, the measured endpoint gain is
    preserved (about 4.2 at x=6 for the TT curve). When true, the entire fitted
    output curve is additionally scaled so its positive endpoint is 6.
    """

    def __init__(self, curve_path, corner="TT", num_parameters=10,
                 normalize_positive_endpoint=False, max_value=6.0,
                 compile_evaluator=False):
        max_value = float(max_value)
        if max_value <= 0:
            raise ValueError("Measured ReLU6 max_value must be positive.")
        super().__init__(
            curve_path=curve_path,
            v_dd=max_value,
            corner=corner,
            num_parameters=num_parameters,
            normalize_positive_endpoint=normalize_positive_endpoint,
            compile_evaluator=compile_evaluator)
        self.max_value = max_value
