#!/usr/bin/env python3
"""Compare shared-Vin native conductance banks without interpolation.

Nonlinearity: max_V |F(V)/F(0)-1| per curve, F=G or R=1/G.
Within-corner variation: population CV across 100 curves at each V.
Between-corner variation: population CV across 45 corner means at each V.
Shape-only variation applies these CVs to G(V)/G(0), removing zero-voltage gain.
All statistics weight curves/corners/voltage samples equally, not by current.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def distribution(a):
    return {name: float(value) for name, value in zip(
        ('min', 'median', 'p95', 'max'), np.percentile(a, [0, 50, 95, 100]))}


def cv(a, axis):
    return np.std(a, axis=axis) / np.mean(a, axis=axis)


def analyze(folder):
    files = sorted(folder.glob('*.csv'))
    if len(files) != 45:
        raise ValueError(f'Expected 45 corners: {folder}')
    arrays, hashes, grid = [], {}, None
    for path in files:
        with path.open(newline='') as stream:
            header = next(csv.reader(stream))
        a = np.loadtxt(path, delimiter=',', skiprows=1)
        if len(header) != 101 or a.shape[1] != 101:
            raise ValueError(f'Expected Vin plus 100 curves: {path}')
        if not np.isfinite(a).all() or (a[:, 1:] <= 0).any():
            raise ValueError(f'Nonfinite/nonpositive source: {path}')
        if grid is None:
            grid = a[:, 0]
        if not np.array_equal(grid, a[:, 0]) or (np.diff(grid) <= 0).any():
            raise ValueError(f'Inconsistent/nonmonotonic grid: {path}')
        arrays.append(a[:, 1:])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    g = np.stack(arrays)  # corner, voltage, curve
    r = 1 / g
    zero = np.flatnonzero(grid == 0)
    if zero.size != 1:
        raise ValueError('Expected one exact native zero sample')
    z = int(zero[0])
    shape = g / g[:, z:z+1, :]
    r0 = r[:, z, :]
    return dict(
        source=str(folder.resolve()), files=[p.name for p in files],
        source_sha256=hashes, corners=45, curves_per_corner=100,
        samples_per_curve=len(grid), voltage_domain=[float(grid[0]), float(grid[-1])],
        conductance_range_S=[float(g.min()), float(g.max())],
        resistance_at_zero_ohms=distribution(r0),
        conductance_nonlinearity_percent=distribution(100*np.max(np.abs(shape-1), axis=1)),
        resistance_nonlinearity_percent=distribution(
            100*np.max(np.abs(r/r[:, z:z+1, :]-1), axis=1)),
        within_corner_R0_cv_percent=distribution(100*cv(r0, axis=1)),
        between_corner_mean_R0_cv_percent=float(100*cv(r0.mean(axis=1), axis=0)),
        within_corner_shape_G_cv_percent=distribution(100*cv(shape, axis=2)),
        between_corner_mean_shape_G_cv_percent=distribution(100*cv(shape.mean(axis=2), axis=0)),
        within_corner_R_cv_percent=distribution(100*cv(r, axis=2)),
        between_corner_mean_R_cv_percent=distribution(100*cv(r.mean(axis=2), axis=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = {sub: {name: analyze(root/sub) for name, root in (
        ('old', args.old), ('new', args.new))} for sub in ('Gm_full', 'Gm')}
    for sub, pair in results.items():
        if pair['old']['files'] != pair['new']['files']:
            raise ValueError(f'Corner filenames differ: {sub}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    lines = ['# Native coupler-bank comparison', '', __doc__, '',
             'CV = population standard deviation / mean. Lower means less variation.',
             'R(0) is the sampled zero-voltage resistance, not the configured nominal R.', '',
             'The full-range bank is the drop-in source; Gm is a separate narrow-range export.', '']
    for sub, pair in results.items():
        lines += [f'## {sub}', '', '| Metric | Old 0p6 | New 1024 |', '|---|---:|---:|']
        for label, key in (
            ('R(0), median [kOhm]', 'resistance_at_zero_ohms'),
            ('G nonlinearity, median / p95 [%]', 'conductance_nonlinearity_percent'),
            ('R nonlinearity, median / p95 [%]', 'resistance_nonlinearity_percent'),
            ('Within-corner R(0) CV, median [%]', 'within_corner_R0_cv_percent'),
            ('Between-corner mean R(0) CV [%]', 'between_corner_mean_R0_cv_percent'),
            ('Within-corner normalized G-shape CV, median [%]', 'within_corner_shape_G_cv_percent'),
            ('Between-corner mean normalized G-shape CV, median [%]', 'between_corner_mean_shape_G_cv_percent'),
            ('Within-corner R(V) CV, median [%]', 'within_corner_R_cv_percent'),
            ('Between-corner mean R(V) CV, median [%]', 'between_corner_mean_R_cv_percent')):
            values = []
            for name in ('old', 'new'):
                value = pair[name][key]
                if key == 'resistance_at_zero_ohms':
                    values.append(f'{value["median"]/1000:.6g}')
                elif 'nonlinearity' in key:
                    values.append(f'{value["median"]:.6g} / {value["p95"]:.6g}')
                else:
                    values.append(f'{value["median"] if isinstance(value, dict) else value:.6g}')
            lines.append('| '+ ' | '.join([label, *values])+' |')
        lines += ['', f'Domain: {pair["new"]["voltage_domain"]} V; '
                  f'{pair["new"]["samples_per_curve"]} samples per curve; 45 x 100 curves.', '']
    (args.output/'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
