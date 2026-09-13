"""Export all-corner resistance curves on the existing TC covariance grid.

Diagnostic data preparation only. The v2 source contains conductance; use
the existing loader to invert each curve before fitting resistance covariance.
Keep the existing TC code-specific mean table unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from utils import load_mc_res_training_curve_bank,load_mc_res_curve_gaussian,load_mc_res_curve_bank,_mc_coupler_paths
from tc_nonidealities import prepare_tc_resistance_curves


def align(bank,grid):
    samples=[]
    for g,left,slope,length in zip(bank['v_grid'],bank['R_left'],bank['R_slope'],bank['lengths']):
        length=int(length);g=g[:length].numpy()
        if grid[0]<g[0] or grid[-1]>g[-1]:raise ValueError('Reference grid requires extrapolation.')
        values=np.r_[left[:length-1].numpy(),float(left[length-2]+slope[length-2]*(g[-1]-g[-2]))]
        samples.append(np.interp(grid,g,values))
    return np.stack(samples)


def export(path,samples,grid):
    paired=np.empty((len(grid),2*len(samples)))
    paired[:,0::2]=grid[:,None];paired[:,1::2]=samples.T
    np.savetxt(path,paired,delimiter=',',comments='',
        header=','.join(f'curve{i}_{axis}' for i in range(len(samples)) for axis in ('V','R_ohms')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='hardware_data/mc_45_corners/coupler_monte_v2')
    parser.add_argument('--reference',default='hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv')
    parser.add_argument('--output-dir',default='results/tc_covariance_v2_comparison')
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    fit=load_mc_res_curve_gaussian(args.reference,quantity='resistance',dtype=torch.float64)
    grid=fit['v_grid'].numpy()
    paths=_mc_coupler_paths(args.source,None)
    bank=load_mc_res_training_curve_bank(args.source,mode='exact_curve',corner_range='all',
        quantity='conductance',dtype=torch.float64)
    lower=float(bank['v_grid'][:,0].max())
    upper=float(bank['v_grid'][torch.arange(bank['lengths'].numel()),bank['lengths']-1].min())
    common=(grid>=lower)&(grid<=upper);grid=grid[common]
    samples=align(bank,grid)
    if len(paths)!=45 or len(samples)!=4500:
        raise ValueError(f'Expected all 45 corners / 4500 curves, got {len(paths)} / {len(samples)}')
    target=output/'coupler_monte_v2_all4500_resistance.csv'
    export(target,samples,grid)
    baseline=align(load_mc_res_curve_bank(args.reference,quantity='resistance',dtype=torch.float64),grid)
    export(output/'current_all4500_common_grid.csv',baseline,grid)
    reread=load_mc_res_curve_gaussian(target,quantity='resistance',dtype=torch.float64)
    direct=np.cov(samples,rowvar=False,ddof=1)
    loaded=(reread['factor']@reread['factor'].T).numpy()*reread['value_scale']**2
    np.testing.assert_allclose(direct,loaded,rtol=1e-9,atol=.001)
    old=np.cov(baseline,rowvar=False,ddof=1)
    def stats(cov,mean):
        std=np.sqrt(np.diag(cov));index=np.argmin(abs(grid))
        return dict(rms_std_ohms=float(np.sqrt(np.diag(cov).mean())),
            near_zero_voltage=float(grid[index]),near_zero_std_ohms=float(std[index]),
            rms_relative_std=float(np.sqrt(np.mean((std/mean)**2))),
            near_zero_mean_ohms=float(mean[index]),mean_ohms=float(mean.mean()))
    result=dict(source=str(Path(args.source).resolve()),source_quantity='conductance',
        fit_quantity='resistance',curve_count=len(samples),corner_count=len(paths),
        source_paths=list(paths),grid_points=len(grid),grid_min=float(grid[0]),grid_max=float(grid[-1]),
        output=str(target.resolve()),reference=args.reference,
        current=stats(old,baseline.mean(0)),
        coupler_monte_v2=stats(direct,samples.mean(0)),
        covariance_trace_ratio=float(np.trace(direct)/np.trace(old)))
    (output/'covariance_stats.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_paths'},indent=2))
    # Independent diagnostic of the existing Gaussian positive-resistance guard;
    # do not modify the floor or the inference model to make the test succeed.
    package=prepare_tc_resistance_curves('hardware_data/res_vs_vin_10k_150k.csv',target,
        levels=torch.arange(16,dtype=torch.float64)/15,R=1e4,R_max=150e3)
    sampled=package.sample(torch.arange(1,16).repeat(10000,1),
        generator=torch.Generator().manual_seed(4096))
    near=int(package.v_grid.abs().argmin());clipped=sampled==package.floor_ohms
    guard=dict(draws_per_code=10000,seed=4096,floor_ohms=package.floor_ohms,
        reference_voltage=float(package.v_grid[near]),rows=[])
    for code in range(15):
        guard['rows'].append(dict(code=code+1,nominal_R=float(package.programmed_resistances[code]),
            near_zero_mean_R=float(package.means[code,near]),
            near_zero_floor_fraction=float(clipped[:,code,near].double().mean()),
            any_voltage_floor_fraction=float(clipped[:,code].any(1).double().mean()),
            max_relative_conductance=float(package.programmed_resistances[code]/sampled[:,code].min())))
    (output/'guard_diagnostic.json').write_text(json.dumps(guard,indent=2))
    print('Largest-code guard diagnostic:',guard['rows'][-1])

if __name__=='__main__':main()
