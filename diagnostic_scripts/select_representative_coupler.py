#!/usr/bin/env python3
"""Select a TT realization by its mean RMS log-R distance to the full bank.

Distance is evaluated on 50 uniformly spaced voltages in the common stored
domain. The output repeats the selected native X/Y columns twice, as required
by the existing distinct-index FF/FB training sampler. No samples are changed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import _mc_coupler_corner_paths


def select_representative(bank, output, lower=None, upper=None):
    bank, output = Path(bank).resolve(), Path(output).resolve()
    metadata = output.with_suffix('.json')
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {metadata}')
    entries = _mc_coupler_corner_paths(bank, None)
    if len(entries) != 45:
        raise ValueError('Selection requires the full 45-corner directory')
    curves, tt_indices, tt_path = [], [], None
    for corner, path in entries:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        if data.ndim != 2 or data.shape[1] != 200:
            raise ValueError(f'Expected 100 paired conductance curves: {path}')
        for index in range(100):
            x, g = data[:, 2*index], data[:, 2*index+1]
            if not np.isfinite(data[:, 2*index:2*index+2]).all() or (g <= 0).any() or (np.diff(x) <= 0).any():
                raise ValueError(f'Invalid curve {index} in {path}')
            if corner == 'TT_V1_T1':
                tt_indices.append(len(curves))
                tt_path = Path(path)
            curves.append((x, 1/g))
    low, high = max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves)
    low = low if lower is None else float(lower)
    high = high if upper is None else float(upper)
    if low < max(x[0] for x, _ in curves) or high > min(x[-1] for x, _ in curves) or low >= high:
        raise ValueError('Requested distance domain is outside the common stored domain')
    grid = np.linspace(low, high, 50)
    log_r = np.log(np.stack([np.interp(grid, x, r) for x, r in curves]))
    distances = np.array([
        np.sqrt(np.mean((log_r-log_r[i])**2, axis=1)).mean()
        for i in tt_indices])
    chosen = int(distances.argmin())
    x, r = curves[tt_indices[chosen]]
    report = dict(bank=str(bank), corner='TT_V1_T1', curve_index=chosen,
                  mcparamset=chosen+1, metric='mean full-bank RMS natural-log resistance distance',
                  distance=float(distances[chosen]), distance_grid=grid.tolist(),
                  R_at_zero_ohms=float(np.interp(0, x, r)),
                  source_file=str(tt_path), output=str(output),
                  exported_samples='all native rows, verbatim X/Y strings; curve repeated twice')
    with tt_path.open(newline='') as handle:
        source_rows = list(csv.reader(handle))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([f'Gm_full (mcparamset={chosen+1}) {axis}' for _ in range(2) for axis in ('X', 'Y')])
        for row in source_rows[1:]:
            pair = row[2*chosen:2*chosen+2]
            writer.writerow(pair+pair)
    metadata.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lower', type=float)
    parser.add_argument('--upper', type=float)
    args = parser.parse_args()
    print(json.dumps(select_representative(args.bank, args.output, args.lower, args.upper), indent=2))
