import pytest
import numpy as np
import similarity


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 0),
    (np.array([0, 0]), np.array([1, 1]), np.sqrt(2)),
    (np.array([0, 0]), np.array([3, 4]), 5),
    (np.array([1]), np.array([2]), 1),
], ids=['same_point', 'root', 'integer', '1-dimentional_vector'])
def test_calc_euclidean_distance(v1, v2, expected):
    assert similarity.calc_euclidean_distance(v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([0, 0]), np.array([1, 1]), 1/(1+np.sqrt(2))),
    (np.array([0, 0]), np.array([3, 4]), 1/6),
    (np.array([1]), np.array([2]), 1/2),
], ids=['same_point', 'root', 'integer', '1-dimentional_vector'])
def test_calc_euclidean_similarity(v1, v2, expected):
    assert similarity.calc_euclidean_similarity(v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([1, 2]), np.array([1, 0]), 1),
    (np.array([0, 0]), np.array([3, 4]), 0),
    (np.array([1]), np.array([2]), 1/2),
], ids=['same_point', 'one_missing_value', 'all_missing_value', '1-dimentional_vector'])
def test_calc_similarity_with_missing_value(v1, v2, expected):
    assert similarity.calc_similarity_with_missing_value(
        v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([1, 2]), np.array([-1, -2]), -1),
    (np.array([1, 2]), np.array([3, 3]), 0),
    (np.array([1, 2, 3]), np.array([1, 3, 2]), 0.5),
    (np.array([1]), np.array([2]), 0),
], ids=['1', '-1', 'constant', 'float', '1-dimentional_vector'])
def test_calc_pearson_correlation_coefficient(v1, v2, expected):
    assert similarity.calc_pearson_correlation_coefficient(
        v1, v2) == expected


@pytest.mark.parametrize('v1,v2', [
    (np.array([1, 2, 3]), np.array([1, 3, 2])),
], ids=['float'])
def test_calc_similarity(v1, v2):
    assert similarity.calc_similarity(
        v1, v2, metric='euclidean') == similarity.calc_euclidean_similarity(v1, v2)
    assert similarity.calc_similarity(
        v1, v2, metric='pearson') == similarity.calc_pearson_correlation_coefficient(v1, v2)
