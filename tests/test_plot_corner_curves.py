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


@pytest.mark.parametrize('curve_type,bank', [
    ('relu', '0906_RELU_Voltage'), ('relu', '1022_RELU_Voltage'),
    ('nonlinear_R', 'coupler_full_range'),
    ('nonlinear_R', 'coupler_full_range_0p6_1024/Gm_full'),
])
def test_v0_t0_colors_preserve_all_curves_and_use_new_filename(tmp_path, curve_type, bank):
    source = ROOT / 'hardware_data/mc_45_corners' / bank
    if not source.is_dir():
        pytest.skip('Optional local bank is not installed')
    kwargs = {'relu_source': source} if curve_type == 'relu' else {'coupler_source': source}
    with patch.object(plotting, 'draw_curves') as draw:
        plotting.plot_all(curve_type, tmp_path, **kwargs)
        original = draw.call_args
        plotting.plot_all(curve_type, tmp_path, highlight_v0_t0=True, **kwargs)
        highlighted = draw.call_args
    colors = highlighted.kwargs['curve_colors']
    assert len(colors) == 4500
    assert colors.count('blue') == 500 and colors.count('red') == 4000
    assert 'curve_colors' not in original.kwargs
    assert highlighted.args[4] != original.args[4]
    assert highlighted.args[4].stem.endswith('_v0_t0_blue_others_red')
    for i, corner in enumerate(plotting.all_corner_ids()):
        expected = 'blue' if corner.endswith('_V0_T0') else 'red'
        assert colors[100*i:100*(i+1)] == [expected]*100
    for old_curve, new_curve in zip(original.args[0], highlighted.args[0]):
        for old_axis, new_axis in zip(old_curve, new_curve):
            np.testing.assert_array_equal(old_axis, new_axis)


def test_highlight_legend_and_color_rendering(tmp_path):
    destination = tmp_path/'highlight.pdf'
    curves = [(np.array([0., 1.]), np.array([0., 1.]))]*2
    with patch.object(plotting.plt, 'close'):
        plotting.draw_curves(curves, 'x', 'y', 'test', destination,
                             curve_colors=['blue', 'red'])
        fig = plotting.plt.gcf()
        ax = fig.axes[0]
        assert [line.get_color() for line in ax.lines] == ['blue', 'red']
        assert ax.lines[0].get_zorder() > ax.lines[1].get_zorder()
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            'V0_T0 (all processes)', 'All other corners']
    plotting.plt.close(fig)
    assert destination.is_file() and destination.with_suffix('.png').is_file()
