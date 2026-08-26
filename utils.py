'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import glob
import os
import re
import sys
import time
import math
from functools import lru_cache
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.init as init


@lru_cache(maxsize=256)
def _load_mc_res_vs_vin(path, curve_index, quantity, nominal_R):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError(
            "MC nonlinear-R data must contain paired X/Y columns.")
    curve_index = int(curve_index)
    n_curves = data.shape[1] // 2
    if curve_index < 0 or curve_index >= n_curves:
        raise IndexError(
            "MC nonlinear-R curve index {} is outside [0, {}).".format(
                curve_index, n_curves))

    v_grid = data[:, 2 * curve_index]
    values = data[:, 2 * curve_index + 1]
    valid = np.isfinite(v_grid) & np.isfinite(values)
    v_grid, values = v_grid[valid], values[valid]
    if v_grid.size < 2:
        raise ValueError("Selected MC nonlinear-R curve has fewer than two points.")

    quantity = str(quantity).lower()
    if quantity == "conductance":
        if np.any(values <= 0):
            raise ValueError(
                "Conductance MC curves must be strictly positive.")
        values = 1.0 / values
    elif quantity != "resistance":
        raise ValueError(
            "nonlinear_R_mc_quantity must be conductance or resistance.")

    return v_grid, np.asarray([float(nominal_R)]), values[:, None]


@lru_cache(maxsize=32)
def _load_mc_res_curve_bank(path, quantity):
    """Load all paired MC curves into a padded piecewise-linear curve bank."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError(
            "MC nonlinear-R data must contain paired X/Y columns.")

    quantity = str(quantity).lower()
    if quantity not in {"conductance", "resistance"}:
        raise ValueError(
            "nonlinear_R_mc_quantity must be conductance or resistance.")

    grids, values = [], []
    for curve_index in range(data.shape[1] // 2):
        grid = data[:, 2 * curve_index]
        curve = data[:, 2 * curve_index + 1]
        valid = np.isfinite(grid) & np.isfinite(curve)
        grid, curve = grid[valid], curve[valid]
        if grid.size < 2:
            raise ValueError(
                "MC nonlinear-R curve {} has fewer than two points.".format(
                    curve_index))
        order = np.argsort(grid)
        grid, curve = grid[order], curve[order]
        if np.any(np.diff(grid) <= 0):
            raise ValueError(
                "MC nonlinear-R curve {} must have a strictly increasing "
                "input grid.".format(curve_index))
        if quantity == "conductance":
            if np.any(curve <= 0):
                raise ValueError(
                    "Conductance MC curves must be strictly positive.")
            curve = 1.0 / curve
        grids.append(grid)
        values.append(curve)

    max_points = max(grid.size for grid in grids)
    n_curves = len(grids)
    padded_grid = np.empty((n_curves, max_points), dtype=np.float64)
    padded_left = np.empty((n_curves, max_points - 1), dtype=np.float64)
    padded_slope = np.zeros((n_curves, max_points - 1), dtype=np.float64)
    lengths = np.empty(n_curves, dtype=np.int64)
    for index, (grid, curve) in enumerate(zip(grids, values)):
        length = grid.size
        lengths[index] = length
        padded_grid[index, :length] = grid
        padded_grid[index, length:] = grid[-1]
        padded_left[index, :length - 1] = curve[:-1]
        padded_left[index, length - 1:] = curve[-1]
        padded_slope[index, :length - 1] = (
            np.diff(curve) / np.diff(grid))

    return padded_grid, padded_left, padded_slope, lengths


def load_mc_res_curve_bank(path, quantity="conductance", curve_indices=None,
                           dtype=torch.float32, device="cpu"):
    """Return a curve bank independent of how curves are later assigned."""
    grid, left, slope, lengths = _load_mc_res_curve_bank(
        os.path.abspath(os.fspath(path)), quantity)
    if curve_indices is not None:
        curve_indices = np.asarray(curve_indices, dtype=np.int64).reshape(-1)
        if curve_indices.size == 0:
            raise ValueError("curve_indices must not be empty.")
        if curve_indices.min() < 0 or curve_indices.max() >= grid.shape[0]:
            raise IndexError("Curve-bank index is out of range.")
        grid = grid[curve_indices]
        left = left[curve_indices]
        slope = slope[curve_indices]
        lengths = lengths[curve_indices]
    return {
        "v_grid": torch.tensor(grid, dtype=dtype, device=device),
        "R_left": torch.tensor(left, dtype=dtype, device=device),
        "R_slope": torch.tensor(slope, dtype=dtype, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.int64, device=device),
    }


_MC_PROCESS_ORDER = ("tt", "ff", "ss", "fs", "sf")
_MC_PROCESS_ALIASES = {
    "tt": "tt", "ttg": "tt",
    "ff": "ff", "ffg": "ff", "ffag": "ff",
    "ss": "ss", "ssg": "ss", "ssag": "ss",
    "fs": "fs", "fsg": "fs",
    "sf": "sf", "sfg": "sf",
}


def _mc_corner_ids(corner_range):
    if corner_range is None:
        return None
    if isinstance(corner_range, str):
        tokens = [
            token.strip().upper()
            for token in corner_range.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().upper() for token in corner_range]
    if not tokens or tokens == ["ALL"] or tokens == ["NONE"]:
        return None
    return tuple(tokens)


def _mc_coupler_paths(source, corner_ids):
    source = os.path.abspath(os.fspath(source))
    if os.path.isfile(source):
        if corner_ids is not None:
            raise ValueError(
                "Corner filtering requires a coupler_monte directory.")
        return (source,)
    if not os.path.isdir(source):
        raise FileNotFoundError(
            "Nonlinear-R Monte Carlo source not found: {}".format(source))

    paths = sorted(glob.glob(os.path.join(source, "*.csv")))
    pattern = re.compile(
        r"([a-z]+)_(-?[0-9.]+)_([0-2])\.csv\Z", re.IGNORECASE)
    parsed = []
    for path in paths:
        match = pattern.match(os.path.basename(path))
        if match is None:
            continue
        process_token = match.group(1).lower()
        if process_token not in _MC_PROCESS_ALIASES:
            continue
        parsed.append((
            path, _MC_PROCESS_ALIASES[process_token],
            float(match.group(2)), int(match.group(3))))
    if not parsed:
        raise ValueError(
            "Coupler Monte Carlo directory contains no recognized corner CSVs: "
            "{}".format(source))

    temperatures = sorted({entry[2] for entry in parsed})
    by_corner = {}
    for path, process, temperature, voltage_level in parsed:
        corner_id = "{}_V{}_T{}".format(
            process.upper(), voltage_level, temperatures.index(temperature))
        if corner_id in by_corner:
            raise ValueError(
                "Multiple coupler files map to corner {}.".format(corner_id))
        by_corner[corner_id] = path

    requested = set(by_corner) if corner_ids is None else set(corner_ids)
    unknown = sorted(requested - set(by_corner))
    if unknown:
        raise ValueError(
            "Unknown nonlinear-R training corners: {}.".format(unknown))
    ordered = [
        "{}_V{}_T{}".format(process.upper(), voltage, temperature)
        for process in _MC_PROCESS_ORDER
        for voltage in range(3)
        for temperature in range(3)
    ]
    return tuple(by_corner[corner] for corner in ordered if corner in requested)


@lru_cache(maxsize=16)
def _load_mc_res_training_curve_bank(source, mode, corner_ids, quantity):
    """Load raw curves or one raw-domain mean curve from each selected corner."""
    paths = _mc_coupler_paths(source, corner_ids)
    grids, values = [], []
    for path in paths:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] % 2 != 0:
            raise ValueError(
                "MC nonlinear-R data must contain paired X/Y columns.")
        file_grids, file_values = [], []
        for curve_index in range(data.shape[1] // 2):
            grid = data[:, 2 * curve_index]
            curve = data[:, 2 * curve_index + 1]
            valid = np.isfinite(grid) & np.isfinite(curve)
            grid, curve = grid[valid], curve[valid]
            order = np.argsort(grid)
            grid, curve = grid[order], curve[order]
            if grid.size < 2 or np.any(np.diff(grid) <= 0):
                raise ValueError(
                    "Every MC nonlinear-R curve requires a strictly increasing grid.")
            if np.any(curve <= 0):
                raise ValueError("MC nonlinear-R curves must be strictly positive.")
            file_grids.append(grid)
            file_values.append(curve)

        if mode == "mean":
            reference = file_grids[0]
            lower = max(grid[0] for grid in file_grids)
            upper = min(grid[-1] for grid in file_grids)
            reference = reference[
                (reference >= lower) & (reference <= upper)]
            if reference.size < 2:
                raise ValueError(
                    "MC curves have no common voltage range for their mean.")
            aligned = [
                np.interp(reference, grid, curve)
                for grid, curve in zip(file_grids, file_values)
            ]
            file_grids = [reference]
            file_values = [np.stack(aligned, axis=0).mean(axis=0)]

        for grid, curve in zip(file_grids, file_values):
            resistance = 1.0 / curve if quantity == "conductance" else curve
            grids.append(grid)
            values.append(resistance)

    max_points = max(grid.size for grid in grids)
    padded_grid = np.empty((len(grids), max_points), dtype=np.float64)
    padded_left = np.empty((len(grids), max_points - 1), dtype=np.float64)
    padded_slope = np.zeros((len(grids), max_points - 1), dtype=np.float64)
    lengths = np.empty(len(grids), dtype=np.int64)
    for index, (grid, curve) in enumerate(zip(grids, values)):
        length = grid.size
        lengths[index] = length
        padded_grid[index, :length] = grid
        padded_grid[index, length:] = grid[-1]
        padded_left[index, :length - 1] = curve[:-1]
        padded_left[index, length - 1:] = curve[-1]
        padded_slope[index, :length - 1] = np.diff(curve) / np.diff(grid)
    return padded_grid, padded_left, padded_slope, lengths


def load_mc_res_training_curve_bank(
        source, mode, corner_range="all", quantity="conductance",
        dtype=torch.float32, device="cpu"):
    """Return the Level-2 shared-curve training bank for selected MC45 corners."""
    mode = str(mode).lower()
    if mode not in {"exact_curve", "mean"}:
        raise ValueError(
            "nonlinear_R_train_mode must be exact_curve or mean.")
    quantity = str(quantity).lower()
    if quantity not in {"conductance", "resistance"}:
        raise ValueError(
            "nonlinear_R_mc_quantity must be conductance or resistance.")
    corner_ids = _mc_corner_ids(corner_range)
    grid, left, slope, lengths = _load_mc_res_training_curve_bank(
        os.path.abspath(os.fspath(source)), mode, corner_ids, quantity)
    return {
        "v_grid": torch.tensor(grid, dtype=dtype, device=device),
        "R_left": torch.tensor(left, dtype=dtype, device=device),
        "R_slope": torch.tensor(slope, dtype=dtype, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.int64, device=device),
    }

@lru_cache(maxsize=32)
def _load_mc_curve_gaussian(path, quantity, curve_indices):
    """Fit one full-covariance Gaussian to a selected bank of MC curves."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError(
            "MC nonlinear-R data must contain paired X/Y columns.")

    quantity = str(quantity).lower()
    if quantity not in {"conductance", "resistance"}:
        raise ValueError(
            "nonlinear_R_mc_quantity must be conductance or resistance.")

    n_curves = data.shape[1] // 2
    if curve_indices:
        selected = np.asarray(curve_indices, dtype=np.int64)
    else:
        selected = np.arange(n_curves, dtype=np.int64)
    if selected.size < 2:
        raise ValueError(
            "Full-covariance curve fitting requires at least two MC curves.")
    if selected.min() < 0 or selected.max() >= n_curves:
        raise IndexError("Curve-bank index is out of range.")

    grids, curves = [], []
    for curve_index in selected:
        grid = data[:, 2 * curve_index]
        curve = data[:, 2 * curve_index + 1]
        valid = np.isfinite(grid) & np.isfinite(curve)
        grid, curve = grid[valid], curve[valid]
        order = np.argsort(grid)
        grid, curve = grid[order], curve[order]
        if grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError(
                "Each MC nonlinear-R curve must have a strictly increasing "
                "input grid with at least two points.")
        if np.any(curve <= 0):
            raise ValueError(
                "{} MC curves must be strictly positive.".format(
                    quantity.capitalize()))
        grids.append(grid)
        curves.append(curve)

    # A covariance matrix requires every sample to describe the same voltage
    # coordinates. Preserve an already-common grid exactly; otherwise align
    # the curves to the first grid over their common characterized range.
    reference_grid = grids[0]
    if not all(np.array_equal(grid, reference_grid) for grid in grids[1:]):
        lower = max(grid[0] for grid in grids)
        upper = min(grid[-1] for grid in grids)
        reference_grid = reference_grid[
            (reference_grid >= lower) & (reference_grid <= upper)]
        if reference_grid.size < 2:
            raise ValueError(
                "MC nonlinear-R curves have no common voltage grid range.")
        curves = [
            np.interp(reference_grid, grid, curve)
            for grid, curve in zip(grids, curves)
        ]

    samples = np.stack(curves, axis=0)
    value_scale = float(np.mean(np.abs(samples)))
    if not np.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("MC curve scale must be finite and positive.")
    normalized = samples / value_scale
    mean = normalized.mean(axis=0)
    covariance = np.cov(normalized, rowvar=False, ddof=1)

    # Add only the minimum numerical diagonal regularization needed by
    # Cholesky; this is not a PCA or low-rank approximation.
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        diagonal_scale = max(
            float(np.max(np.diag(covariance))), np.finfo(np.float64).eps)
        jitter = diagonal_scale * 1e-12
        for _ in range(8):
            try:
                factor = np.linalg.cholesky(
                    covariance + jitter * np.eye(covariance.shape[0]))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise ValueError(
                "MC curve covariance is not numerically positive semidefinite.")

    return reference_grid, mean, factor, value_scale


def load_mc_res_curve_gaussian(path, quantity="conductance",
                               curve_indices=None, dtype=torch.float32,
                               device="cpu"):
    """Return a direct full-covariance Gaussian fit of measured MC curves."""
    index_key = (
        None if curve_indices is None
        else tuple(int(index) for index in curve_indices)
    )
    grid, mean, factor, value_scale = _load_mc_curve_gaussian(
        os.path.abspath(os.fspath(path)), str(quantity).lower(), index_key)
    return {
        "v_grid": torch.tensor(grid, dtype=dtype, device=device),
        "mean": torch.tensor(mean, dtype=dtype, device=device),
        "factor": torch.tensor(factor, dtype=dtype, device=device),
        "value_scale": float(value_scale),
        "quantity": str(quantity).lower(),
    }


# BEGIN TEMPORARY PATCH: Aug-4 single-curve nonlinear-R means.
# Delete this helper, its two wrapper arguments, and the MC45 launch flag once
# every corner has enough updated Monte Carlo curves to fit both moments from
# one data source.
def patch_mc_res_curve_gaussian_mean(curve_gaussian, mean_curve_path,
                                     mean_curve_index):
    """Replace only a fitted Gaussian's mean with one characterized curve."""
    if str(curve_gaussian["quantity"]).lower() != "conductance":
        raise ValueError(
            "patched_nonlinearity_data currently contains conductance curves.")

    data = np.loadtxt(mean_curve_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError(
            "Patched nonlinear-R data must contain paired X/Y columns.")
    curve_index = int(mean_curve_index)
    n_curves = data.shape[1] // 2
    if curve_index < 0 or curve_index >= n_curves:
        raise IndexError("Patched nonlinear-R curve index is out of range.")

    grid = data[:, 2 * curve_index]
    curve = data[:, 2 * curve_index + 1]
    valid = np.isfinite(grid) & np.isfinite(curve)
    grid, curve = grid[valid], curve[valid]
    order = np.argsort(grid)
    grid, curve = grid[order], curve[order]
    if grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError(
            "The patched nonlinear-R mean must use a strictly increasing grid.")
    if np.any(curve <= 0):
        raise ValueError(
            "The patched nonlinear-R conductance mean must be positive.")

    reference_grid = curve_gaussian["v_grid"].detach().cpu().numpy()
    endpoint_tolerance = (
        10 * np.finfo(reference_grid.dtype).eps *
        max(1.0, abs(grid[0]), abs(grid[-1])))
    if (reference_grid[0] < grid[0] - endpoint_tolerance or
            reference_grid[-1] > grid[-1] + endpoint_tolerance):
        raise ValueError(
            "The patched nonlinear-R mean does not cover the covariance grid.")
    interpolation_grid = np.clip(reference_grid, grid[0], grid[-1])
    patched_mean = np.interp(interpolation_grid, grid, curve)
    value_scale = float(curve_gaussian["value_scale"])

    output = dict(curve_gaussian)
    output["mean"] = torch.tensor(
        patched_mean / value_scale,
        dtype=curve_gaussian["mean"].dtype,
        device=curve_gaussian["mean"].device)
    return output
# END TEMPORARY PATCH: Aug-4 single-curve nonlinear-R means.


def load_res_vs_vin(dir_path=os.path.dirname(os.path.abspath(__file__)), R=50e3, R_max=180e3,
                    dtype=torch.float32, device="cpu", nonlinear_R_table=None,
                    nonlinear_R_mc_curve_index=None,
                    nonlinear_R_mc_quantity="conductance"):
    if nonlinear_R_table is None:
        if R_max is None:
            raise ValueError(
                "R_max is required to infer the nonlinear-R table filename when "
                "nonlinear_R_table is not provided.")
        data_dir = os.path.join(dir_path, "hardware_data", "res_vs_vin_{}k_{}k.csv".format(
            str(int(R/1e3)), str(int(R_max/1e3))))
    else:
        table_name = os.fspath(nonlinear_R_table)
        candidates = [
            table_name,
            os.path.join(dir_path, table_name),
            os.path.join(dir_path, "hardware_data", table_name),
        ]
        data_dir = next((path for path in candidates if os.path.exists(path)), None)
        if data_dir is None:
            raise FileNotFoundError(
                "Nonlinear-R table not found: {}".format(nonlinear_R_table))
    if not os.path.exists(data_dir):
        return None, None, None
    if nonlinear_R_mc_curve_index is not None:
        v_grid, R_codes, R_table = _load_mc_res_vs_vin(
            data_dir, nonlinear_R_mc_curve_index,
            nonlinear_R_mc_quantity, R)
    else:
        df = pd.read_csv(data_dir)
        v_grid = df[df.columns[0]].values
        R_codes = [float(_) for _ in list(df.columns)[1:]]
        R_table = df[list(df.columns)[1:]].values

    return (
        torch.tensor(v_grid, dtype=dtype, device=device),
        torch.tensor(R_codes, dtype=dtype, device=device),
        torch.tensor(R_table, dtype=dtype, device=device)
    )


def interpolate_R_eff(v, v_grid, R_codes, R_left, R_slope, code_idx, proj_fn=None):
    """Piecewise-linear R(v) interpolation shared by ordinary and pulse MVMs."""
    if proj_fn is not None:
        v = proj_fn(v)

    interval_idx = torch.bucketize(v, v_grid) - 1
    interval_idx = interval_idx.clamp(min=0, max=v_grid.numel() - 2)
    n_codes = R_codes.numel()
    v_flat = v.reshape(-1)
    interval_flat = interval_idx.reshape(-1)

    if torch.is_tensor(code_idx):
        if code_idx.numel() == 1:
            column = min(max(int(code_idx.item()), 0), n_codes - 1)
            position = interval_flat * n_codes + column
        else:
            columns = code_idx.to(torch.long).reshape(-1).clamp(0, n_codes - 1)
            position = interval_flat * n_codes + columns
    else:
        column = min(max(int(code_idx), 0), n_codes - 1)
        position = interval_flat * n_codes + column

    left = R_left.reshape(-1)[position]
    slope = R_slope.reshape(-1)[position]
    v_left = v_grid[interval_flat]
    return (left + slope * (v_flat - v_left)).reshape_as(v)


def expand_weights_to_matrix(input_shape, weight_tensor, stride=1, padding=0, flip_weight=False):
    if flip_weight:
        weight_tensor = weight_tensor.flip([2, 3])

    C_in, H_in, W_in = input_shape
    C_out, _, K, _ = weight_tensor.shape

    # Compute output dimensions
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    # List to store sparse indices and values
    indices = []
    values = []

    for c_out in range(C_out):
        for h in range(H_out):
            for w in range(W_out):
                start_h = h * stride
                start_w = w * stride
                filter_idx = c_out * H_out * W_out + h * W_out + w
                for c_in in range(C_in):
                    for i in range(K):
                        for j in range(K):
                            # c_in * (H_in + 2 * padding) * (W_in + 2 * padding)
                            # + (start_h + i) * (W_in + 2 * padding) + (start_w + j)
                            input_idx = (c_in * (H_in + 2 * padding) + (start_h + i)) * (W_in + 2 * padding) + (start_w + j)
                            value = weight_tensor[c_out, c_in, i, j].item()
                            if value != 0:
                                indices.append([filter_idx, input_idx])
                                values.append(value)
    # Convert to sparse tensor
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float32)
    size = (C_out * H_out * W_out, C_in * (H_in + 2 * padding) * (W_in + 2 * padding))
    expanded_weights = torch.sparse_coo_tensor(indices, values, size=size)

    return expanded_weights


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 77

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


