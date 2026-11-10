import csv
import hashlib
from pathlib import Path

import pytest

from diagnostic_scripts.select_representative_coupler import select_representative
from utils import mc45_corner_ids, load_mc_res_training_curve_bank


def test_typical_selection_and_verbatim_singleton(tmp_path):
    bank = tmp_path/'bank'
    bank.mkdir()
    for corner in mc45_corner_ids():
        process, voltage, temperature = corner.split('_')
        path = bank/f'{process.lower()}_{(-20,25,85)[int(temperature[1:])]}_{voltage[1:]}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([f'Gm (mcparamset={i}) {axis}' for i in range(1,101) for axis in ('X','Y')])
            for x in ['-0.1', '0', '0.1']:
                writer.writerow([value for i in range(100) for value in
                    (x, '2.000e-5' if corner != 'TT_V1_T1' or i == 33 else '1e-5')])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bank.glob('*.csv')}
    output = tmp_path/'singleton.csv'
    report = select_representative(bank, output)
    assert report['mcparamset'] == 34
    assert report['R_at_zero_ohms'] == pytest.approx(50000)
    with output.open(newline='') as handle:
        rows = list(csv.reader(handle))[1:]
    assert rows == [[x,'2.000e-5',x,'2.000e-5'] for x in ['-0.1','0','0.1']]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bank.glob('*.csv')}
    assert load_mc_res_training_curve_bank(output, 'exact_curve')['v_grid'].shape == (2,3)
    with pytest.raises(FileExistsError):
        select_representative(bank, output)
