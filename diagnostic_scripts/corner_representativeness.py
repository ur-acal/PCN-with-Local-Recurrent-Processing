"""All-curve mean-distance and reciprocal-normalized MC45 corner scores.

This is an offline, model-independent curve-bank diagnostic. It never changes
banks or checkpoints and does not estimate actual corner sampling frequencies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from measured_activation import PiecewiseLinearActivation
from measured_pooling import MeasuredAvgPool2d
from utils import _mc_coupler_corner_paths, mc45_corner_ids


@torch.no_grad()
def coupler_vectors(path, grid):
    """Use production resistance interpolation, including endpoint clamping."""
    bank = MeasuredAvgPool2d(curve_path=str(path), quantity='conductance')
    query = torch.as_tensor(grid, dtype=bank.v_grid.dtype)
    count = bank.v_grid.shape[0]
    resistance = bank._interpolate_resistance(
        query.expand(count, -1), torch.arange(count))
    if not torch.isfinite(resistance).all() or (resistance <= 0).any():
        raise ValueError(f'Nonpositive/nonfinite resistance: {path}')
    return np.log(resistance.double().numpy())


@torch.no_grad()
def relu_vectors(path, grid, v_dd):
    """Use the production offset, voltage scaling, interpolation and rail clamp."""
    activation = PiecewiseLinearActivation(
        str(path), v_dd=v_dd, corner='MC1',
        normalize_positive_endpoint=False, adapt_relu_offset=True,
        fuse_measured_activation=False)
    query = torch.as_tensor(grid, dtype=activation.vin.dtype)
    return np.stack([
        activation(query, corner=name).double().numpy()
        for name in activation.corner_names])


def corner_distances(vectors):
    """Mean RMS distance across every pair, including within-corner pairs.

    vectors has shape (corners, curves, voltage points). Chunking one source
    corner at a time avoids storing the full 4500 x 4500 matrix.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if (vectors.ndim != 3 or min(vectors.shape) == 0
            or not np.isfinite(vectors).all()):
        raise ValueError('Expected finite (corners, curves, voltage points) data')
    corners, curves, points = vectors.shape
    flat = vectors.reshape(corners * curves, points)
    result = np.empty((corners, corners))
    for index, source in enumerate(vectors):
        distances = cdist(source, flat, metric='euclidean') / np.sqrt(points)
        result[index] = distances.reshape(curves, corners, curves).mean(axis=(0, 2))
    return result


def normalized_scores(distances):
    """Return A_c = mean_j D_cj and 100 (1/A_c) / sum_r (1/A_r)."""
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or distances.shape[0] == 0 or not np.isfinite(distances).all()
            or (distances < 0).any()):
        raise ValueError('Expected a finite nonnegative square distance matrix')
    average = distances.mean(axis=1)
    if np.all(average == 0):
        return average, np.full(len(average), 100.0 / len(average))
    if (average <= 0).any():
        raise ValueError('Inconsistent zero average distance')
    # Equivalent to 1/A, with a common factor to avoid reciprocal overflow.
    weights = average.min() / average
    return average, 100.0 * weights / weights.sum()


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(coupler_bank, relu_bank, output, v_dd=0.6, points=301):
    if v_dd <= 0 or points < 3 or points % 2 == 0:
        raise ValueError('Use positive VDD and an odd grid size >= 3')
    corners = mc45_corner_ids()
    paths = {
        'coupler': dict(_mc_coupler_corner_paths(coupler_bank, None)),
        'relu': dict(_mc_coupler_corner_paths(relu_bank, None)),
    }
    for component, mapping in paths.items():
        if set(mapping) != set(corners):
            raise ValueError(f'{component} requires all 45 corners')
    hashes = {path: fingerprint(path) for mapping in paths.values() for path in mapping.values()}
    grid = np.linspace(-v_dd, v_dd, points)
    results = {}
    for component, loader in (
            ('coupler', lambda path: coupler_vectors(path, grid)),
            ('relu', lambda path: relu_vectors(path, grid, v_dd))):
        vectors = np.stack([loader(paths[component][corner]) for corner in corners])
        if vectors.shape != (45, 100, points):
            raise ValueError(f'Unexpected {component} curve-bank shape: {vectors.shape}')
        distances = corner_distances(vectors)
        average, scores = normalized_scores(distances)
        results[component] = (distances, average, scores)
        print(f'{component}: 4500 curves, pairwise means complete', flush=True)
    if any(fingerprint(path) != digest for path, digest in hashes.items()):
        raise RuntimeError('Input bank changed during analysis; refusing output')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [dict(corner=corner,
                 coupler_mean_rms_logR=float(results['coupler'][1][i]),
                 coupler_score_percent=float(results['coupler'][2][i]),
                 relu_mean_rms_volts=float(results['relu'][1][i]),
                 relu_score_percent=float(results['relu'][2][i]))
            for i, corner in enumerate(corners)]
    with (output / 'corner_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Corner | Coupler score (%) | ReLU score (%) |',
             '|---|---:|---:|']
    lines.extend(f'| {r["corner"]} | {r["coupler_score_percent"]:.4f} | '
                 f'{r["relu_score_percent"]:.4f} |' for r in rows)
    (output / 'corner_scores.md').write_text('\n'.join(lines) + '\n')
    for component, (distances, _, _) in results.items():
        with (output / f'{component}_corner_distances.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['corner', *corners])
            writer.writerows([corner, *row] for corner, row in zip(corners, distances))
    report = dict(
        coupler_bank=str(Path(coupler_bank).resolve()),
        relu_bank=str(Path(relu_bank).resolve()), v_dd=v_dd,
        grid_volts=grid.tolist(), curves_per_corner=100,
        pairs_per_corner_pair=10000, comparisons_per_corner=450000,
        definition='D_cj = mean_ab RMS_V(f_ca - f_jb); A_c = mean_j D_cj; score_c = 100 (1/A_c) / sum_r (1/A_r)',
        coupler_transform='natural log of resistance in ohms; no per-curve R(0) normalization',
        relu_transform='production processed physical output volts; no unitless q pullback or endpoint normalization',
        boundaries='Production interpolation clamps to each stored domain; ReLU also clamps output to +/-VDD',
        weighting='Uniform corners, curves and grid points; includes own corner and zero-distance identical-curve pairs',
        interpretation='Reciprocal-normalized average-distance centrality, NOT actual sampling probability or a local-density estimate',
        source_sha256=hashes, source_files_unchanged=True, rows=rows)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coupler-bank', type=Path, required=True)
    parser.add_argument('--relu-bank', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--v-dd', type=float, default=0.6)
    parser.add_argument('--points', type=int, default=301)
    args = parser.parse_args()
    analyze(args.coupler_bank, args.relu_bank, args.output_dir, args.v_dd, args.points)
