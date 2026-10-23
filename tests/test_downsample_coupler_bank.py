import csv
import pytest

from diagnostic_scripts.downsample_coupler_bank import export_bank, select_rows
from utils import load_mc_res_curve_bank


def test_selection_preserves_native_endpoints_and_zero():
    grid = [(i-1199)/2000 for i in range(2400)]
    selected = select_rows(grid)
    assert len(selected) == len(set(selected)) == 150
    assert selected == sorted(selected)
    assert selected[0] == 0 and selected[-1] == 2399 and 1199 in selected
    with pytest.raises(ValueError):
        select_rows([1.,2.,3.])


def test_export_copies_strings_and_loads_as_paired_bank(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    source_file = source/'tt_25_1.csv'
    with source_file.open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Vin',*[f'Gm_full_{i}' for i in range(1,101)]])
        for i in range(2400):
            writer.writerow([f'{(i-1199)/2000:.8f}',*['0.000020000000000000']*100])
    before = source_file.read_bytes()
    output = tmp_path/'output'
    manifest = export_bank(source,output)
    assert source_file.read_bytes() == before
    with (output/source_file.name).open(newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 151 and all(len(row)==200 for row in rows)
    assert rows[1][1] == '0.000020000000000000'
    original = list(csv.reader(source_file.open()))[1:]
    for row,index in zip(rows[1:],manifest['selected_row_indices_zero_based']):
        assert row == [v for y in original[index][1:] for v in (original[index][0],y)]
    bank = load_mc_res_curve_bank(str(output/source_file.name), quantity='conductance')
    assert bank['v_grid'].shape[0] == 100
    with pytest.raises(FileExistsError):
        export_bank(source,output)
