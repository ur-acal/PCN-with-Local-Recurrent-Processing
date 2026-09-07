import csv
import os
import re

import numpy as np
import torch
import torch.nn as nn


_FIT_CONSTRAINTS = {"none", "nonnegative", "auto"}
_CORNER_MODES = {"fixed", "random_per_forward"}
_CURVE_SHARING_MODES = {"per_model", "per_layer", "per_spin"}
_RAW_CURVE_COLUMN_RE = re.compile(
    r":top_(?P<process>ff|fs|sf|ss|tt),.*?"
    r"VDD_VALUE=(?P<vdd>[-+0-9.eE]+),"
    r"temperature=(?P<temperature>[-+0-9.eE]+)\)\s*"
    r"(?P<axis>[XY])\s*\Z",
    re.IGNORECASE)


_ABSOLUTE_VOUT_REFERENCE = 0.6
_ABSOLUTE_VOUT_SOURCE = "0906_RELU_Voltage"


def _absolute_vout_reference(path):
    """Return the fixed reference used only by the 0906 absolute-Vout bank."""
    source = os.path.basename(os.path.dirname(os.path.abspath(os.fspath(path))))
    return (
        _ABSOLUTE_VOUT_REFERENCE
        if source == _ABSOLUTE_VOUT_SOURCE else 0.0)


def _metadata_token(value):
    value = float(value)
    text = str(int(value)) if value.is_integer() else format(value, "g")
    return text.replace("-", "M").replace(".", "P")


def _raw_curve_column_info(column):
    match = _RAW_CURVE_COLUMN_RE.search(column)
    if match is None:
        return None
    process = match.group("process").upper()
    vdd = _metadata_token(match.group("vdd"))
    temperature = _metadata_token(match.group("temperature"))
    return {
        "axis": match.group("axis").upper(),
        "corner": "{}_VDD{}_T{}".format(process, vdd, temperature),
    }


def _nonnegative_lstsq(design, values):
    """Lawson-Hanson active-set NNLS for the small fixed spline fit."""
    design = design.to(torch.float64)
    values = values.to(torch.float64)
    n_coefficients = design.shape[1]
    solution = design.new_zeros(n_coefficients)
    passive = torch.zeros(n_coefficients, dtype=torch.bool, device=design.device)
    max_iterations = 30 * n_coefficients

    for _ in range(max_iterations):
        dual = design.mT.matmul(values - design.matmul(solution))
        inactive_dual = dual.masked_fill(passive, -torch.inf)
        if inactive_dual.max() <= 0:
            return solution
        passive[inactive_dual.argmax()] = True

        for _ in range(max_iterations):
            candidate = torch.zeros_like(solution)
            candidate[passive] = torch.linalg.lstsq(
                design[:, passive], values).solution
            if torch.all(candidate[passive] > 0):
                solution = candidate
                break

            nonpositive = passive & (candidate <= 0)
            step = (solution[nonpositive] /
                    (solution[nonpositive] - candidate[nonpositive])).min()
            solution = solution + step * (candidate - solution)
            deactivate = passive & (solution <= 0)
            solution[deactivate] = 0
            passive[deactivate] = False
        else:
            raise RuntimeError("Nonnegative spline fit did not converge.")

    raise RuntimeError("Nonnegative spline fit did not converge.")


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


class _CurveSharingMixin:
    """Cache one empirical measured-curve assignment for a hardware trial."""

    def _configure_curve_sharing(self, curve_sharing, curve_seed):
        curve_sharing = str(curve_sharing).lower()
        if curve_sharing not in _CURVE_SHARING_MODES:
            raise ValueError(
                "Measured-activation curve sharing must be one of {}.".format(
                    ", ".join(sorted(_CURVE_SHARING_MODES))))
        self.curve_sharing = curve_sharing
        self.curve_seed = None if curve_seed is None else int(curve_seed)
        self._curve_generator = None
        if self.curve_seed is not None:
            self._curve_generator = torch.Generator(device="cpu")
            self._curve_generator.manual_seed(self.curve_seed)
        self.register_buffer("_sampled_curve_indices", None, persistent=False)

    def _curve_indices_for(self, x):
        if self.curve_sharing == "per_model":
            return None
        expected_shape = (
            torch.Size([]) if self.curve_sharing == "per_layer"
            else torch.Size((1,) + tuple(x.shape[1:])))
        if self._sampled_curve_indices is None:
            sampled = torch.randint(
                len(self.corner_names), expected_shape,
                generator=self._curve_generator, device="cpu")
            self._sampled_curve_indices = sampled.to(device=x.device)
        elif self._sampled_curve_indices.shape != expected_shape:
            raise RuntimeError(
                "Cached measured-activation curve assignment has shape {}, "
                "but this layer requested {}.".format(
                    tuple(self._sampled_curve_indices.shape),
                    tuple(expected_shape)))
        return self._sampled_curve_indices.to(device=x.device)

    def _expanded_curve_indices(self, x):
        indices = self._curve_indices_for(x)
        return None if indices is None else indices.expand_as(x).reshape(-1)


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


class CubicBSplineActivation(_CurveSharingMixin, nn.Module):
    """Cubic B-spline activation fitted once from characterized voltage curves.

    The CSV axes are in volts. Runtime scaling uses s = v_dd / max(abs(Vin)),
    evaluates the spline at clamp(x / s), and returns s * Vout. Curves in the
    0906_RELU_Voltage bank are first referred to the fixed 0.6 V output
    reference; this is deliberately not adapted per PVT corner. Other tables
    preserve their existing zero reference. Fitted coefficients are fixed
    non-persistent buffers, not trainable parameters; gradients still propagate
    through the activation input during training. The buffers are rebuilt from
    the fixed CSV instead of being stored in model checkpoints.
    """

    degree = 3

    def __init__(self, curve_path, v_dd, corner="TT", num_parameters=10,
                 normalize_positive_endpoint=False, compile_evaluator=False,
                 fit_constraint="auto", curve_sharing="per_model",
                 curve_seed=None):
        super().__init__()
        if num_parameters <= self.degree:
            raise ValueError("A cubic B-spline needs at least four control coefficients.")
        fit_constraint = str(fit_constraint).lower()
        if fit_constraint not in _FIT_CONSTRAINTS:
            raise ValueError(
                "fit_constraint must be one of {}.".format(
                    ", ".join(sorted(_FIT_CONSTRAINTS))))

        vin, curves, column_names = self._load_csv(curve_path)
        self.curve_path = str(curve_path)
        self.num_parameters = int(num_parameters)
        self.normalize_positive_endpoint = bool(normalize_positive_endpoint)
        self.compile_evaluator = bool(compile_evaluator)
        self.fit_constraint = fit_constraint
        self.corner_columns = column_names
        self.corner_names = tuple(curves)
        self._configure_curve_sharing(curve_sharing, curve_seed)

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
        corner_fit_constraints = {}
        table_is_nonnegative = all(torch.all(values >= 0) for values in curves.values())
        for name, values in curves.items():
            corner_constraint = self.fit_constraint
            if corner_constraint == "auto":
                corner_constraint = "nonnegative" if table_is_nonnegative else "none"
            if corner_constraint == "nonnegative":
                fit = _nonnegative_lstsq(design, values)
            else:
                fit = torch.linalg.lstsq(
                    design.to(torch.float64), values.to(torch.float64)).solution
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
            corner_fit_constraints[name] = corner_constraint
        self.coefficients = _SplineCoefficientBuffers(coefficients)
        self.register_buffer(
            "coefficient_bank",
            torch.stack([coefficients[name] for name in self.corner_names]),
            persistent=False)
        self.register_buffer(
            "nonnegative_curve_mask",
            torch.tensor([
                corner_fit_constraints[name] == "nonnegative"
                for name in self.corner_names], dtype=torch.bool),
            persistent=False)
        self.endpoint_scales = endpoint_scales
        self.corner_fit_constraints = corner_fit_constraints
        self.active_corner = self._resolve_corner(corner)
        self.default_corner = self.active_corner
        # Ordinary transient attribute: it is intentionally absent from state_dict.
        self._coordinate_pullback_scale = None

    @staticmethod
    def _corner_name(column):
        raw_info = _raw_curve_column_info(column)
        if raw_info is not None:
            return raw_info["corner"]
        name = column
        if name.lower().startswith("vout_"):
            name = name[5:]
        name = re.sub(r"\([^)]*\)", "", name).strip()
        return name.upper()

    @classmethod
    def _load_csv(cls, path):
        path = os.fspath(path)
        if os.path.isdir(path):
            csv_paths = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(".csv"))
            if not csv_paths:
                raise ValueError(
                    "Measured-activation directory contains no CSV files: {}"
                    .format(path))

            shared_vin = None
            curves = {}
            column_names = {}
            for csv_path in csv_paths:
                vin, file_curves, _ = cls._load_csv(csv_path)
                if shared_vin is None:
                    shared_vin = vin
                elif (vin.shape != shared_vin.shape or
                      not torch.allclose(vin, shared_vin, rtol=1e-6, atol=1e-12)):
                    raise ValueError(
                        "Measured-activation directory curves must share one "
                        "Vin grid.")
                file_prefix = os.path.splitext(
                    os.path.basename(csv_path))[0].upper()
                for name, values in file_curves.items():
                    qualified_name = "{}_{}".format(file_prefix, name)
                    curves[qualified_name] = values
                    column_names[qualified_name] = qualified_name
            return shared_vin, curves, column_names

        with open(path) as handle:
            header = handle.readline()
        if "mcparamset=" in header:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            if data.ndim != 2 or data.shape[1] % 2 != 0:
                raise ValueError(
                    "Measured-activation MC CSV must contain paired X/Y columns.")
            vin = torch.tensor(data[:, 0], dtype=torch.float32)
            curves = {}
            column_names = {}
            for index in range(data.shape[1] // 2):
                curve_vin = data[:, 2 * index]
                if not np.allclose(
                        curve_vin, data[:, 0], rtol=1e-6, atol=1e-12):
                    raise ValueError(
                        "Measured-activation MC curves must share one Vin grid.")
                name = "MC{}".format(index + 1)
                curves[name] = torch.tensor(
                    data[:, 2 * index + 1] -
                    _absolute_vout_reference(path), dtype=torch.float32)
                column_names[name] = name
            order = torch.argsort(vin)
            return (
                vin[order],
                {name: values[order] for name, values in curves.items()},
                column_names)

        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            raise ValueError("Measured activation CSV must contain data rows.")

        if "Vin" in fieldnames:
            vin_column = "Vin"
            output_columns = [
                name for name in fieldnames
                if name.lower().startswith("vout")]
        else:
            raw_columns = [
                (name, _raw_curve_column_info(name))
                for name in fieldnames]
            x_columns = [
                name for name, info in raw_columns
                if info is not None and info["axis"] == "X"]
            output_columns = [
                name for name, info in raw_columns
                if info is not None and info["axis"] == "Y"]
            if len(x_columns) != 1:
                raise ValueError(
                    "Raw measured activation CSV must contain exactly one "
                    "characterized X column.")
            vin_column = x_columns[0]
        if not output_columns:
            raise ValueError(
                "Measured activation CSV must contain at least one Vout/Y column.")

        vin = torch.tensor(
            [float(row[vin_column]) for row in rows], dtype=torch.float32)
        order = torch.argsort(vin)
        vin = vin[order]
        curves = {}
        column_names = {}
        for column in output_columns:
            corner = cls._corner_name(column)
            if corner in curves:
                raise ValueError("Duplicate activation corner name: {}".format(corner))
            values = torch.tensor(
                [float(row[column]) for row in rows],
                dtype=torch.float32)[order]
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

    def _evaluate_local_banked(self, x, curve_indices):
        flat = x.reshape(-1)
        span_left = self.span_left.to(device=x.device, dtype=x.dtype)
        span_width = self.span_width.to(device=x.device, dtype=x.dtype)
        position = (flat - span_left[0]) / span_width[0]
        span_offset = torch.floor(position).to(torch.long).clamp(
            min=0, max=span_left.numel() - 1)
        local_u = ((flat - span_left[span_offset]) /
                   span_width[span_offset]).clamp(0, 1)
        controls = self.span_control_indices.to(device=x.device)[span_offset]
        bank = self.coefficient_bank.to(device=x.device, dtype=x.dtype)
        local_coefficients = bank[curve_indices[:, None], controls]
        transform = self.span_polynomial_transform.to(
            device=x.device, dtype=x.dtype)[span_offset]
        polynomial = torch.bmm(
            transform, local_coefficients.unsqueeze(-1)).squeeze(-1)
        p0, p1, p2, p3 = polynomial.unbind(dim=1)
        return (((p3 * local_u + p2) * local_u + p1) *
                local_u + p0).reshape_as(x)

    def _resolve_corner(self, corner):
        requested = str(corner)
        for name, column in self.corner_columns.items():
            if requested == column:
                return name
        normalized = self._corner_name(requested)
        aliases = {"REFERENCE": "TT", "REFERENCE_CORNER": "TT", "TYPICAL": "TT"}
        normalized = aliases.get(normalized, normalized)
        if normalized == "TT" and normalized not in self.coefficients:
            normalized = "TT_VDD1_T25"
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
        curve_indices = None if corner is not None else self._expanded_curve_indices(x_char)
        if curve_indices is None:
            coeff = self.coefficients[selected]
            y_char = self._evaluate_local(x_char, coeff)
            if self.corner_fit_constraints[selected] == "nonnegative":
                y_char = y_char.clamp_min(0)
        else:
            y_char = self._evaluate_local_banked(x_char, curve_indices)
            nonnegative = self.nonnegative_curve_mask.to(
                device=x.device)[curve_indices].reshape_as(x_char)
            y_char = torch.where(nonnegative, y_char.clamp_min(0), y_char)
        output = scale * y_char
        output = output if pullback_scale is None else output / pullback_scale
        v_dd = self.v_dd.to(device=output.device, dtype=output.dtype)
        return output.clamp(min=-v_dd, max=v_dd)


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
                 compile_evaluator=False, fit_constraint="auto"):
        max_value = float(max_value)
        if max_value <= 0:
            raise ValueError("Measured ReLU6 max_value must be positive.")
        super().__init__(
            curve_path=curve_path,
            v_dd=max_value,
            corner=corner,
            num_parameters=num_parameters,
            normalize_positive_endpoint=normalize_positive_endpoint,
            compile_evaluator=compile_evaluator,
            fit_constraint=fit_constraint)
        self.max_value = max_value


class _CurveValueBuffers(nn.Module):
    """Dictionary-like storage for fixed, device-aware measured curves."""

    def __init__(self, curves):
        super().__init__()
        for name, value in curves.items():
            self.register_buffer(name, value, persistent=False)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self._buffers


class PiecewiseLinearActivation(_CurveSharingMixin, nn.Module):
    """Piecewise-linear activation from fixed characterized voltage curves.

    The voltage-coordinate transformation, endpoint normalization, corner
    selection, boundary clamping, and optional coordinate pullback match
    :class:`CubicBSplineActivation`. The only difference is interpolation:
    adjacent measured samples are joined by straight line segments. All table
    values are fixed non-persistent buffers. PyTorch autograd therefore gives
    the local segment slope with respect to the input without an STE or a
    custom backward implementation.
    """

    def __init__(self, curve_path, v_dd, corner="TT",
                 normalize_positive_endpoint=False,
                 curve_sharing="per_model", curve_seed=None):
        super().__init__()
        vin, curves, column_names = CubicBSplineActivation._load_csv(curve_path)
        if vin.numel() < 2:
            raise ValueError(
                "Piecewise-linear activation needs at least two Vin samples.")
        spacing = vin[1:] - vin[:-1]
        if torch.any(spacing <= 0):
            raise ValueError(
                "Measured activation Vin samples must be unique and increasing.")

        self.curve_path = str(curve_path)
        self.normalize_positive_endpoint = bool(normalize_positive_endpoint)
        self.corner_columns = column_names
        self.corner_names = tuple(curves)
        self._configure_curve_sharing(curve_sharing, curve_seed)

        endpoint_scales = {}
        scaled_curves = {}
        for name, values in curves.items():
            endpoint_scale = values.new_tensor(1.0)
            if self.normalize_positive_endpoint:
                measured_endpoint = values[-1]
                if measured_endpoint <= 0:
                    raise ValueError(
                        "Cannot normalize corner {} with nonpositive measured "
                        "endpoint {}.".format(name, measured_endpoint.item()))
                endpoint_scale = vin.abs().max() / measured_endpoint
                values = values * endpoint_scale
            scaled_curves[name] = values
            endpoint_scales[name] = float(endpoint_scale)

        self.register_buffer("vin", vin, persistent=False)
        self.register_buffer("vin_min", vin[0], persistent=False)
        self.register_buffer("vin_max", vin[-1], persistent=False)
        self.register_buffer("v_char", vin.abs().max(), persistent=False)
        self.register_buffer(
            "v_dd", torch.as_tensor(float(v_dd), dtype=vin.dtype),
            persistent=False)
        self.curves = _CurveValueBuffers(scaled_curves)
        self.register_buffer(
            "curve_bank",
            torch.stack([scaled_curves[name] for name in self.corner_names]),
            persistent=False)
        self.endpoint_scales = endpoint_scales
        self.active_corner = self._resolve_corner(corner)
        self.default_corner = self.active_corner
        self._coordinate_pullback_scale = None

        first_spacing = spacing[0]
        uniform_tolerance = (
            32 * torch.finfo(vin.dtype).eps * float(first_spacing.abs()))
        self.uniform_grid = bool(torch.allclose(
            spacing, first_spacing.expand_as(spacing),
            rtol=1e-4, atol=uniform_tolerance))
        self.register_buffer("grid_spacing", first_spacing, persistent=False)

    def _resolve_corner(self, corner):
        requested = str(corner)
        for name, column in self.corner_columns.items():
            if requested == column:
                return name
        normalized = CubicBSplineActivation._corner_name(requested)
        aliases = {
            "REFERENCE": "TT",
            "REFERENCE_CORNER": "TT",
            "TYPICAL": "TT",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized == "TT" and normalized not in self.curves:
            normalized = "TT_VDD1_T25"
        if normalized not in self.curves:
            raise ValueError(
                "Unknown activation corner {!r}; available corners are {}."
                .format(corner, ", ".join(self.corner_names)))
        return normalized

    def select_corner(self, corner):
        self.active_corner = self._resolve_corner(corner)

    def set_v_dd(self, v_dd):
        self.v_dd.fill_(float(v_dd))

    def set_coordinate_pullback_scale(self, scale):
        """Temporarily evaluate Phi(scale*x)/scale without changing the table."""
        if scale is None:
            self._coordinate_pullback_scale = None
            return
        if not torch.is_tensor(scale):
            scale = self.v_dd.new_tensor(float(scale))
        scale = scale.detach()
        if (scale.numel() != 1 or not torch.isfinite(scale).all() or
                scale.item() <= 0):
            raise ValueError(
                "Measured-activation pullback scale must be one finite "
                "positive scalar.")
        self._coordinate_pullback_scale = scale

    def _interpolate(self, x, values):
        vin = self.vin.to(device=x.device, dtype=x.dtype)
        values = values.to(device=x.device, dtype=x.dtype)
        flat = x.reshape(-1)
        if self.uniform_grid:
            position = (
                (flat - vin[0]) /
                self.grid_spacing.to(device=x.device, dtype=x.dtype))
            left_idx = torch.floor(position).to(torch.long)
        else:
            left_idx = torch.searchsorted(
                vin, flat.contiguous(), right=True) - 1
        left_idx = left_idx.clamp(min=0, max=vin.numel() - 2)
        x_left = vin[left_idx]
        x_right = vin[left_idx + 1]
        fraction = (flat - x_left) / (x_right - x_left)
        output = values[left_idx] + fraction * (
            values[left_idx + 1] - values[left_idx])
        return output.reshape_as(x)

    def _interpolate_banked(self, x, curve_indices):
        vin = self.vin.to(device=x.device, dtype=x.dtype)
        values = self.curve_bank.to(device=x.device, dtype=x.dtype)
        flat = x.reshape(-1)
        if self.uniform_grid:
            position = (
                (flat - vin[0]) /
                self.grid_spacing.to(device=x.device, dtype=x.dtype))
            left_idx = torch.floor(position).to(torch.long)
        else:
            left_idx = torch.searchsorted(
                vin, flat.contiguous(), right=True) - 1
        left_idx = left_idx.clamp(min=0, max=vin.numel() - 2)
        x_left = vin[left_idx]
        x_right = vin[left_idx + 1]
        fraction = (flat - x_left) / (x_right - x_left)
        left = values[curve_indices, left_idx]
        right = values[curve_indices, left_idx + 1]
        return (left + fraction * (right - left)).reshape_as(x)

    def forward(self, x, corner=None):
        selected = (
            self.active_corner if corner is None else
            self._resolve_corner(corner))
        pullback_scale = self._coordinate_pullback_scale
        if pullback_scale is not None:
            pullback_scale = pullback_scale.to(device=x.device, dtype=x.dtype)
            x = pullback_scale * x

        scale = (
            self.v_dd.to(device=x.device, dtype=x.dtype) /
            self.v_char.to(device=x.device, dtype=x.dtype))
        x_char = (x / scale).clamp(
            min=self.vin_min.to(device=x.device, dtype=x.dtype),
            max=self.vin_max.to(device=x.device, dtype=x.dtype))
        curve_indices = None if corner is not None else self._expanded_curve_indices(x_char)
        y_char = (
            self._interpolate(x_char, self.curves[selected])
            if curve_indices is None else
            self._interpolate_banked(x_char, curve_indices))
        output = scale * y_char
        output = output if pullback_scale is None else output / pullback_scale
        v_dd = self.v_dd.to(device=output.device, dtype=output.dtype)
        return output.clamp(min=-v_dd, max=v_dd)


class MeasuredPiecewiseLinearReLU6Activation(PiecewiseLinearActivation):
    """Piecewise-linear measured activation in unitless ReLU6 coordinates."""

    def __init__(self, curve_path, corner="TT",
                 normalize_positive_endpoint=False, max_value=6.0):
        max_value = float(max_value)
        if max_value <= 0:
            raise ValueError("Measured ReLU6 max_value must be positive.")
        super().__init__(
            curve_path=curve_path,
            v_dd=max_value,
            corner=corner,
            normalize_positive_endpoint=normalize_positive_endpoint)
        self.max_value = max_value


MEASURED_ACTIVATION_TYPES = (
    CubicBSplineActivation,
    PiecewiseLinearActivation,
)


def configure_measured_activation_corner_mode(
        model, mode="fixed", sharing="per_layer"):
    """Configure fixed or once-per-top-level-forward corner selection.

    Training and training-time evaluation both resample per top-level forward
    using the same per-model, per-layer, or per-spin granularity.
    """
    mode = str(mode).lower()
    sharing = str(sharing).lower()
    if mode not in _CORNER_MODES:
        raise ValueError(
            "Measured activation corner mode must be one of {}.".format(
                ", ".join(sorted(_CORNER_MODES))))
    if sharing not in _CURVE_SHARING_MODES:
        raise ValueError(
            "Random measured-activation curve sharing must be one of {}."
            .format(", ".join(sorted(_CURVE_SHARING_MODES))))

    previous_hook = getattr(
        model, "_measured_activation_corner_hook", None)
    if previous_hook is not None:
        previous_hook.remove()
        model._measured_activation_corner_hook = None

    activations = tuple(
        module for module in model.modules()
        if isinstance(module, MEASURED_ACTIVATION_TYPES))
    if not activations:
        if mode == "fixed":
            return 0
        raise ValueError(
            "random_per_forward requires at least one measured activation.")

    for activation in activations:
        activation.select_corner(activation.default_corner)
    model._measured_activation_corner_mode = mode
    model._measured_activation_random_curve_sharing = sharing
    model._last_measured_activation_corner = None
    if mode == "fixed":
        for activation in activations:
            activation.curve_sharing = "per_model"
            activation._sampled_curve_indices = None
        return len(activations)

    corner_names = activations[0].corner_names
    if any(activation.corner_names != corner_names
           for activation in activations[1:]):
        raise ValueError(
            "All randomly sampled measured activations must load the same corners.")

    def _sample_curve_assignment(module):
        if sharing == "per_spin":
            for activation in activations:
                activation.curve_sharing = "per_spin"
                activation._sampled_curve_indices = None
            module._last_measured_activation_corner = "per_spin"
            return

        for activation in activations:
            activation.curve_sharing = "per_model"
            activation._sampled_curve_indices = None
        sample_count = 1 if sharing == "per_model" else len(activations)
        corner_indices = torch.randint(len(corner_names), (sample_count,))
        selected = tuple(
            corner_names[int(index)] for index in corner_indices)
        if sharing == "per_model":
            selected = selected * len(activations)
        for activation, corner in zip(activations, selected):
            activation.select_corner(corner)
        module._last_measured_activation_corner = (
            selected[0] if sharing == "per_model" else selected)

    def _sample_corner_before_forward(module, inputs):
        _sample_curve_assignment(module)

    model._measured_activation_corner_hook = model.register_forward_pre_hook(
        _sample_corner_before_forward)
    return len(activations)


def feedforward_measured_activation_factory(
        curve_path, v_dd, corner="TT", curve_sharing="per_model",
        curve_seed=None, normalize_positive_endpoint=False,
        interpolation="piecewise_linear", spline_parameters=10,
        fit_constraint="auto", compile_evaluator=False):
    """Build measured activations for the stages of a feedforward CNN."""
    sharing = str(curve_sharing).lower()

    def factory(layer_idx, stage):
        seed = curve_seed
        if seed is not None and sharing != "per_model":
            stage_offset = 0 if stage == "conv1" else 104729
            seed = int(seed) + 1009 * int(layer_idx) + stage_offset
        common = dict(
            curve_path=curve_path, v_dd=v_dd, corner=corner,
            normalize_positive_endpoint=normalize_positive_endpoint,
            curve_sharing=sharing, curve_seed=seed)
        if interpolation == "cubic_bspline":
            return CubicBSplineActivation(
                num_parameters=spline_parameters,
                fit_constraint=fit_constraint,
                compile_evaluator=compile_evaluator, **common)
        if interpolation != "piecewise_linear":
            raise ValueError("Unknown measured-activation interpolation.")
        return PiecewiseLinearActivation(**common)

    return factory


def configure_feedforward_measured_activation(model, factory):
    """Replace residual-block activations while keeping the final ReLU ideal."""
    from physical_feedforward import iter_physical_blocks

    blocks = list(iter_physical_blocks(model))
    for block in blocks:
        if block.conv2 is None:
            continue
        block.act1 = factory(block.layer_idx, "conv1")
        block.act2 = factory(block.layer_idx, "conv2")
    return model


def configure_feedforward_activation_pullback(model, mode="none", q=None):
    """Apply the unitless feedforward pullback ``Phi(q*x)/q``."""
    mode = str(mode).lower()
    if mode not in ("none", "direct"):
        raise ValueError(
            "Feedforward unitless measured pullback must be none or direct.")
    activations = [
        module for module in model.modules()
        if isinstance(module, MEASURED_ACTIVATION_TYPES)
    ]
    if mode == "direct":
        if q is None:
            raise ValueError(
                "Direct feedforward pullback requires positive q.")
        q = torch.as_tensor(q)
        if q.numel() != 1 or not torch.isfinite(q) or q <= 0:
            raise ValueError(
                "Direct feedforward pullback requires positive q.")
        for activation in activations:
            activation.set_coordinate_pullback_scale(float(q))
    else:
        for activation in activations:
            activation.set_coordinate_pullback_scale(None)
    return len(activations)
