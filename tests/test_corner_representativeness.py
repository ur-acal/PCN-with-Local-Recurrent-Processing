import csv

import numpy as np
import pytest

from diagnostic_scripts.corner_representativeness import (
    corner_distances, normalized_scores, coupler_vectors, relu_vectors)


def paired_csv(path, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([f'curve (mcparamset={i+1}) {axis}'
                         for i in range(len(outputs)) for axis in ('X', 'Y')])
        for index, x in enumerate([-.6, 0, .6]):
            writer.writerow([v for output in outputs for v in (x, output[index])])


def test_exact_all_pair_averages_and_self_corner_spread():
    vectors = np.arange(18, dtype=float).reshape(3, 2, 3)
    actual = corner_distances(vectors)
    expected = np.array([[np.mean([
        np.sqrt(np.mean((a-b)**2)) for a in left for b in right])
        for right in vectors] for left in vectors])
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 0] > 0  # Same-corner diversity is not silently discarded.
    average, scores = normalized_scores(actual)
    np.testing.assert_allclose(average, expected.mean(axis=1))
    np.testing.assert_allclose(scores, (1/average)/(1/average).sum()*100)
    assert scores[1] > scores[0]  # Middle bank is closer to the whole bank.
    np.testing.assert_allclose(normalized_scores(actual*1000)[1], scores)


def test_identical_and_invalid_inputs():
    distances = corner_distances(np.ones((45, 2, 3)))
    average, scores = normalized_scores(distances)
    assert np.all(average == 0)
    np.testing.assert_allclose(scores, 100/45)
    with pytest.raises(ValueError):
        corner_distances(np.full((2, 2, 3), np.nan))


def test_production_resistance_and_voltage_processing(tmp_path):
    coupler = tmp_path/'coupler'/'tt_25_1.csv'
    paired_csv(coupler, [[2e-5]*3, [1e-5]*3])
    grid = np.array([-.6, 0, .6])
    log_r = coupler_vectors(coupler, grid)
    np.testing.assert_allclose(log_r, np.log([[50000]*3, [100000]*3]), rtol=1e-7)
    assert np.sqrt(np.mean((log_r[0]-log_r[1])**2)) == pytest.approx(np.log(2))
    relu = tmp_path/'1022_RELU_Voltage'/'tt_-20_0.csv'
    paired_csv(relu, [[.588, .588, 1.188], [.588, .588, .888]])
    output = relu_vectors(relu, grid, .6)
    np.testing.assert_allclose(output, [[0, 0, .6], [0, 0, .3]], atol=1e-7)
