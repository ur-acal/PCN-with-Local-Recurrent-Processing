"""Hardware plots accept both CSV layouts without changing measured samples."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'diagnostic_scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('plot_corner_curves',
        ROOT / 'diagnostic_scripts/plot_corner_curves.py')
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)


@pytest.mark.parametrize('bank, points', [
    ('0906_RELU_Voltage', 201), ('1022_RELU_Voltage', 201),
    ('1022_RELU_Voltage_raw', 201),
    ('coupler_full_range', 150), ('coupler_full_range_0p6/Gm', 400),
    ('coupler_full_range_0p6/Gm_full', 2400)])
def test_native_samples_and_100_curves_preserved(bank, points):
    source = ROOT / 'hardware_data/mc_45_corners' / bank
    if bank in {'1022_RELU_Voltage_raw', 'coupler_full_range_0p6/Gm',
                'coupler_full_range_0p6/Gm_full'} and not source.is_dir():
        pytest.skip('Optional local raw bank is not installed')
    path = source / 'tt_25_1.csv'
    curves = plotting.load_paired_curves(path)
    raw = np.loadtxt(path, delimiter=',', skiprows=1)
    assert len(curves) == 100
    for i, (x, y) in enumerate(curves):
        assert len(x) == len(y) == points
        np.testing.assert_array_equal(x, raw[:, 0 if raw.shape[1] == 101 else 2*i])
        np.testing.assert_array_equal(y, raw[:, 1+i if raw.shape[1] == 101 else 2*i+1])


def test_per_corner_reference_and_legacy_defaults():
    root = ROOT / 'hardware_data/mc_45_corners'
    for level, offset in enumerate((.588, .600, .612)):
        old = root / '0906_RELU_Voltage' / f'tt_25_{level}.csv'
        new = root / '1022_RELU_Voltage' / old.name
        assert plotting.relu_output_reference(old) == pytest.approx(offset)
        assert plotting.relu_output_reference(new, 'vdd_half',
            (1.176, 1.2, 1.224)) == pytest.approx(offset)
        assert plotting.relu_output_reference(new, 'none') == 0
        assert plotting.relu_output_reference(new) == pytest.approx(offset)
    with pytest.raises(ValueError, match='conflicts'):
        plotting.relu_output_reference(old, 'vdd_half', (1, 1, 1))
    assert plotting.relu_output_reference(root/'relu_monteCarlo/relu_ttg0.csv') == 0


def test_missing_supply_metadata_requires_explicit_reference(tmp_path):
    path = tmp_path / 'tt_25_1.csv'
    path.write_text('Vin,Vout\n0,0.6\n1,1.2\n')
    with pytest.raises(ValueError, match='Missing supply metadata'):
        plotting.relu_output_reference(path, 'vdd_half')


def test_all_4500_relu_curves_use_own_reference(tmp_path):
    source = ROOT / 'hardware_data/mc_45_corners/1022_RELU_Voltage'
    with patch.object(plotting, 'draw_curves') as draw:
        plotting.plot_all('relu', tmp_path, relu_source=source, v_dd=.5,
                          offset_mode='vdd_half', supply_voltages=(1.176,1.2,1.224))
    curves = draw.call_args.args[0]
    assert len(curves) == 4500
    for corner_index, corner in enumerate(plotting.all_corner_ids()):
        raw = np.loadtxt(plotting.relu_path(corner, source), delimiter=',', skiprows=1)
        level = int(corner.split('_')[1][1:])
        x, y = curves[100*corner_index]
        np.testing.assert_allclose(x, raw[:,0]*.5/.6)
        np.testing.assert_allclose(y, (raw[:,1]-(.588,.6,.612)[level])*.5/.6)


@pytest.mark.parametrize('bank,points', [
    ('coupler_full_range_0p6_150pts', 150),
    ('coupler_full_range_0p6/Gm_full', 2400)])
def test_coupler_source_is_dynamic_and_preserves_full_grid(tmp_path, bank, points):
    source = ROOT / 'hardware_data/mc_45_corners' / bank
    if bank == 'coupler_full_range_0p6/Gm_full' and not source.is_dir():
        pytest.skip('Optional local raw bank is not installed')
    with patch.object(plotting, 'draw_curves') as draw:
        plotting.plot_all('nonlinear_R', tmp_path, coupler_source=source)
    curves = draw.call_args.args[0]
    assert len(curves) == 4500
    assert all(len(x) == points for x,y in curves)
    assert plotting.nonlinear_r_path('TT_V1_T1').parent.name == 'coupler_monte_v2'
    assert plotting.nonlinear_r_path('TT_V1_T1', 'coupler_monte').name == 'ttg_25_1.csv'


def test_pdf_and_png_written(tmp_path):
    destination = tmp_path / 'curves.pdf'
    plotting.draw_curves([(np.array([0.,1.]),np.array([0.,1.]))],
                         'x','y','test',destination)
    assert destination.is_file() and destination.with_suffix('.png').is_file()
