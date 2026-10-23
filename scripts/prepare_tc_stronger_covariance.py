"""Prepare experimental TC resistance CSVs; no training code is changed.

Keep the existing mean-table and paired-X/Y covariance-source schemas. Convert
each toggle conductance curve to resistance before interpolation/covariance.
The two source banks are alternatives, never combined. Their absolute pooled
covariance is shared by all 15 active TC codes, as in the production loader.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_tc_covariance_comparison import align, export
from tc_nonidealities import prepare_tc_resistance_curves
from utils import load_mc_res_curve_gaussian, load_mc_res_training_curve_bank, _mc_coupler_paths


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'hardware_data/tc_stronger_covariance')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Do not overwrite a previous experimental input silently.
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    old_mean = ROOT / 'hardware_data/res_vs_vin_10k_150k.csv'
    reference = ROOT / 'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'
    with old_mean.open(newline='') as handle:
        header = next(csv.reader(handle))
    mean_data = np.loadtxt(old_mean, delimiter=',', skiprows=1)
    scaled = mean_data.copy()
    scaled[:, 1:] *= 5
    mean_path = out / 'res_vs_vin_50k_750k.csv'
    np.savetxt(mean_path, scaled, delimiter=',', comments='',
               header=','.join([header[0]] + [format(float(v) * 5, '.17g') for v in header[1:]]))
    reference_fit = load_mc_res_curve_gaussian(reference, quantity='resistance', dtype=torch.float64)
    reference_grid = reference_fit['v_grid'].numpy()
    levels = torch.arange(16, dtype=torch.float64) / 15
    report = dict(
        mean_table=str(mean_path.relative_to(ROOT)) if mean_path.is_relative_to(ROOT) else str(mean_path),
        mean_source=str(old_mean.relative_to(ROOT)), mean_source_sha256=digest(old_mean),
        mean_table_note='Synthetic rescaling: multiply both resistance column labels and R(V) by 5; Vin unchanged.',
        R_ohms=50000, R_max_ohms=750000, active_codes=15,
        covariance_note='One absolute resistance covariance pooled over 45 corners x 100 samples, shared by all codes. No extra covariance scaling.',
        cnn_mapping='TC CNN supports positive nominal R with the existing derived R_max=15R; these inputs use 50k/750k.',
        pooling_note='The TC path also uses the covariance source for measured pooling; set pooling nominal resistance to 50000 for these means.',
        sources={},
    )
    for name in ('coupler_monte_v2', 'coupler_full_range'):
        source = ROOT / 'hardware_data/mc_45_corners' / name
        paths = _mc_coupler_paths(source, None)
        bank = load_mc_res_training_curve_bank(source, mode='exact_curve',
                    corner_range='all', quantity='conductance', dtype=torch.float64)
        lower = float(bank['v_grid'][:, 0].max())
        upper = float(bank['v_grid'][torch.arange(bank['lengths'].numel()), bank['lengths'] - 1].min())
        grid = reference_grid[(reference_grid >= lower) & (reference_grid <= upper)]
        samples = align(bank, grid)
        assert len(paths) == 45 and samples.shape == (4500, len(grid))
        assert np.isfinite(samples).all() and (samples > 0).all()
        covariance_path = out / f'{name}_all4500_resistance.csv'
        export(covariance_path, samples, grid)
        covariance = np.cov(samples, rowvar=False, ddof=1)
        package = prepare_tc_resistance_curves(mean_path, covariance_path,
                    levels=levels, R=50000, R_max=750000, dtype=torch.float64)
        np.testing.assert_array_equal(package.v_grid.numpy(), grid)
        np.testing.assert_allclose(package.covariance.numpy(), covariance, rtol=1e-9, atol=.001)
        original = prepare_tc_resistance_curves(old_mean, covariance_path,
                    levels=levels, R=10000, R_max=150000, dtype=torch.float64)
        torch.testing.assert_close(package.means, 5 * original.means, rtol=1e-13, atol=1e-8)
        torch.testing.assert_close(package.mean_column_resistances,
                                   5 * original.mean_column_resistances, rtol=0, atol=0)
        fp32 = prepare_tc_resistance_curves(mean_path, covariance_path,
                    levels=levels, R=50000, R_max=750000, dtype=torch.float32)
        torch.testing.assert_close(fp32.covariance.double(), package.covariance, rtol=1e-5, atol=100)
        draws = package.sample(torch.arange(1, 16).repeat(10000, 1),
                               generator=torch.Generator().manual_seed(4096))
        std = np.sqrt(np.diag(covariance))
        near = int(np.argmin(abs(grid)))
        min_z = float(np.min(package.means.numpy() / std))
        report['sources'][name] = dict(
            covariance_file=str(covariance_path), sha256=digest(covariance_path),
            source_files=[dict(path=str(Path(p).relative_to(ROOT)), sha256=digest(p)) for p in paths],
            curve_count=4500, corner_count=45, columns=9000,
            grid_points=len(grid), grid_min_V=float(grid[0]), grid_max_V=float(grid[-1]),
            omitted_reference_voltages_V=reference_grid[~np.isin(reference_grid, grid)].tolist(),
            extrapolation=False, near_zero_V=float(grid[near]),
            near_zero_std_ohms=float(std[near]), source_near_zero_mean_ohms=float(samples[:, near].mean()),
            code15_near_zero_mean_ohms=float(package.means[-1, near]),
            code15_near_zero_std_over_mean=float(std[near] / package.means[-1, near]),
            largest_pointwise_negative_gaussian_probability=.5 * math.erfc(min_z / math.sqrt(2)),
            floor_hits_10000_draws_per_code=int((draws == package.floor_ohms).sum()),
            guard_note='Gaussian support remains unbounded: finite sampling cannot guarantee no negative draw in expanded models.',
            validation='PASS: production float64/float32 loaders, direct covariance, all 15 means, positive finite input curves.',
        )
        print(name, json.dumps({k: v for k, v in report['sources'][name].items() if k != 'source_files'}, indent=2))
    report['mean_table_sha256'] = digest(mean_path)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Created', out)


if __name__ == '__main__':
    main()
