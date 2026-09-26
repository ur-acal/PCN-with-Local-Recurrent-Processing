"""Save a finite TC Gaussian curve bank for empirical-with-replacement inference."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ode_pc import ODEBlockPC
from tc_nonidealities import prepare_tc_resistance_curves, load_tc_empirical_bank


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mean-table', default='hardware_data/res_vs_vin_10k_150k.csv')
    parser.add_argument('--covariance-table', default='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
    parser.add_argument('--draws', type=int, default=100)
    parser.add_argument('--seed', type=int, default=4096)
    parser.add_argument('--weight-scale', type=float, default=1.)
    args = parser.parse_args()
    if args.draws < 1: parser.error('--draws must be positive')
    levels = ODEBlockPC._get_quant_magnitude_levels(
        SimpleNamespace(q_hi=15, weight_scale=args.weight_scale),
        torch.empty(0, dtype=torch.float64))
    package = prepare_tc_resistance_curves(args.mean_table, args.covariance_table, levels=levels)
    codes = torch.arange(1, levels.numel()).repeat_interleave(args.draws)
    curves = package.sample(codes, generator=torch.Generator().manual_seed(args.seed))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez(handle, v_grid=package.v_grid.numpy(),
                 programmed_resistances=package.programmed_resistances.numpy(),
                 curves=curves.reshape(-1, args.draws, curves.shape[-1]).numpy(), seed=args.seed)
    load_tc_empirical_bank(package, path)
    print(f'Saved {levels.numel()-1} codes x {args.draws} curves to {path}')


if __name__ == '__main__': main()
