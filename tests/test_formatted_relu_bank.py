"""1022 data is a numeric-preserving, midpoint-corrected 0906 replacement."""
import csv
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from data_utils import MC45CornerData
from ft_corner_sampling import _activation_corner_curve_indices, _validated_mc45_mapping
from measured_activation import CubicBSplineActivation, PiecewiseLinearActivation

ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage'
RAW=BANK.with_name(BANK.name+'_raw')


@pytest.mark.parametrize('tc', [False, True])
@pytest.mark.parametrize('source', ['0906_RELU_Voltage', '1022_RELU_Voltage'])
@pytest.mark.parametrize('random', [False, True])
@pytest.mark.parametrize('override', [False, True])
def test_pcn_slurm_exports_selected_voltage_bank(tc, source, random, override):
    env = dict(PATH=os.environ['PATH'], TC_DRY_RUN='true',
               TC_NONIDEALITIES=str(tc).lower(),
               TOGGLE_MODE='none' if tc else 'odexinit',
               TASK='cifar100', IMG_TYPE='CiFAIR',
               MC_RELU_MONTE_CARLO_SOURCE=source,
               ACTIVATION_CORNER_MODE='random_per_forward' if random else 'fixed')
    if override:
        env.update(ACTIVATION_CURVE_PATH='explicit/bank', ACTIVATION_CORNER='EXPLICIT_MC1')
    out = subprocess.check_output(['bash', 'launch_scripts/slurm_search_config.sh'],
                                  cwd=ROOT, env=env, text=True, timeout=30)
    commands = [shlex.split(line) for line in out.splitlines() if line.startswith('sbatch ')]
    assert commands
    for command in commands:
        exports = next(value.removeprefix('--export=') for value in command
                       if value.startswith('--export='))
        fields = dict(item.split('=', 1) for item in exports.split(',') if '=' in item)
        assert fields['ACTIVATION_CURVE_PATH'] == (
            'explicit/bank' if override else
            f'./hardware_data/mc_45_corners/{source}'+('' if random else '/tt_25_1.csv'))
        assert fields['ACTIVATION_CORNER'] == (
            'EXPLICIT_MC1' if override else 'TT_25_1_MC18' if random else 'MC18')


def test_formatted_bank_matches_recorded_output_hashes():
    manifest = json.loads((BANK/'format_manifest.json').read_text())
    assert len(manifest['files']) == 45
    for record in manifest['files']:
        assert hashlib.sha256((BANK/record['name']).read_bytes()).hexdigest() == record['output_sha256']


def test_all_exported_strings_match_untouched_raw_files():
    if not RAW.is_dir():
        pytest.skip('Optional local raw ReLU bank is not installed')
    manifest=json.loads((BANK/'format_manifest.json').read_text())
    assert len(manifest['files']) == 45
    for record in manifest['files']:
        source=RAW/record['name']; target=BANK/record['name']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==record['output_sha256']
        with source.open(newline='') as stream: original=list(csv.reader(stream))[1:]
        with target.open(newline='') as stream: output=list(csv.reader(stream))
        assert len(output[0])==200 and len(output)==202
        for actual,row in zip(output[1:],original):
            assert actual==[v for y in row[1:] for v in (row[0],y)]


def test_catalog_and_coupled_sampler_load_all_4500_curves():
    catalog=MC45CornerData(ROOT/'hardware_data/mc_45_corners',
                          relu_monte_carlo_source=BANK,
                          coupler_nonlinear_variation_source='coupler_full_range_0p6_150pts')
    assert len(catalog.corners)==45
    for corner in catalog.corners:
        assert len(corner['relu']['curve_indices'])==100
        assert corner['relu']['voltage']==pytest.approx((1.176,1.2,1.224)[corner['voltage_level']])
    vin,curves,_=CubicBSplineActivation._load_csv(BANK)
    assert len(vin)==201 and len(curves)==4500
    assert 'TT_25_1_MC18' in curves
    mapping=_validated_mc45_mapping(_activation_corner_curve_indices(tuple(curves)), '1022 ReLU')
    assert len(mapping)==45
    for corner in catalog.corners:
        file_curves=set(CubicBSplineActivation._load_csv(corner['relu']['path'])[1])
        assert file_curves=={f'MC{i}' for i in range(1,101)}


@pytest.mark.parametrize('level,reference', [(0,.588),(1,.6),(2,.612)])
def test_runtime_offset_subtracted_exactly_once(level,reference):
    path=BANK/f'tt_25_{level}.csv'
    data=np.loadtxt(path,delimiter=',',skiprows=1)
    vin,curves,_=CubicBSplineActivation._load_csv(path)
    torch.testing.assert_close(curves['MC18'],torch.tensor(data[:,35]-reference,dtype=torch.float32))
    _,legacy,_=CubicBSplineActivation._load_csv(path,adapt_relu_offset=False)
    torch.testing.assert_close(legacy['MC18'],torch.tensor(data[:,35]-.6,dtype=torch.float32))
    activation=PiecewiseLinearActivation(path,v_dd=.5,corner='MC18',fuse_measured_activation=False)
    output=activation(torch.tensor([0.,.2],requires_grad=True))
    assert torch.isfinite(output).all()
    output.sum().backward()


@pytest.mark.parametrize('source',['0906_RELU_Voltage','1022_RELU_Voltage'])
@pytest.mark.parametrize('builder',['tc_nonideality_args.sh','tc_feedforward_args.sh'])
@pytest.mark.parametrize('mode',['fixed','random_per_forward','eval'])
def test_tc_builder_preserves_old_defaults_and_selects_new_bank(source,builder,mode):
    stage='eval' if mode=='eval' else 'ft'
    env=dict(PATH=os.environ['PATH'],TC_NONIDEALITIES='true',TC_FEEDFORWARD='true',
             MC_RELU_MONTE_CARLO_SOURCE=source,
             ACTIVATION_CORNER_MODE='random_per_forward' if mode=='random_per_forward' else 'fixed')
    array='TC_ARGS' if builder=='tc_nonideality_args.sh' else 'TC_FEEDFORWARD_ARGS'
    command=f'set -e; source launch_scripts/{builder} {stage}; '
    if builder=='tc_nonideality_args.sh':
        command+=f'printf "%s\\n" "${{{array}[@]}}"'
    else:
        command+='printf "%s\\n" "$ACTIVATION_CURVE_PATH" "$ACTIVATION_CORNER"'
    result=subprocess.run(['bash','-c',command],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    if builder=='tc_nonideality_args.sh':
        path=lines[lines.index('--activation_curve_path')+1]
        corner=lines[lines.index('--activation_corner')+1]
    else:path,corner=lines[:2]
    assert path==f'./hardware_data/mc_45_corners/{source}'+('/tt_25_1.csv' if mode=='fixed' else '')
    assert corner==('MC18' if mode=='fixed' else 'TT_25_1_MC18')


@pytest.mark.parametrize('mode',['fixed','random_per_forward'])
@pytest.mark.parametrize('source',['0906_RELU_Voltage','1022_RELU_Voltage'])
def test_toggle_launch_selects_bank_with_same_curve_ids(tmp_path,mode,source):
    python=tmp_path/'python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    env=dict(PATH=str(tmp_path)+':'+os.environ['PATH'],LOGDIR=str(tmp_path/'logs'),
             MODEL_DIR=str(tmp_path/'source'),ACTIVATION_CORNER_MODE=mode,
             MC_RELU_MONTE_CARLO_SOURCE=source)
    result=subprocess.run(['bash','launch_scripts/run_toggle_ode_mixed_ft.sh'],cwd=ROOT,
                           env=env,capture_output=True,text=True,check=True)
    args=result.stdout.splitlines()
    path=args[args.index('--activation_curve_path')+1]
    corner=args[args.index('--activation_corner')+1]
    assert path==f'./hardware_data/mc_45_corners/{source}'+('/tt_25_1.csv' if mode=='fixed' else '')
    assert corner==('MC18' if mode=='fixed' else 'TT_25_1_MC18')
