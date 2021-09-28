import pytest
import numpy as np
import recommendations


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 0),
    (np.array([0, 0]), np.array([1, 1]), np.sqrt(2)),
    (np.array([0, 0]), np.array([3, 4]), 5),
    (np.array([1]), np.array([2]), 1),
])
def test_calc_euclidean_distance(v1, v2, expected):
    assert recommendations.calc_euclidean_distance(v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([0, 0]), np.array([1, 1]), 1/(1+np.sqrt(2))),
    (np.array([0, 0]), np.array([3, 4]), 1/6),
    (np.array([1]), np.array([2]), 1/2),
])
def test_calc_euclidean_similarity(v1, v2, expected):
    assert recommendations.calc_euclidean_similarity(v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([1, 2]), np.array([1, 0]), 1),
    (np.array([0, 0]), np.array([3, 4]), 0),
    (np.array([1]), np.array([2]), 1/2),
])
def test_calc_similarity_with_missing_value(v1, v2, expected):
    assert recommendations.calc_similarity_with_missing_value(
        v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([1, 2]), np.array([-1, -2]), -1),
    (np.array([1, 2]), np.array([3, 3]), 0),
    (np.array([1, 2, 3]), np.array([1, 3, 2]), 0.5),
    (np.array([1]), np.array([2]), 0),
])
def test_calc_pearson_correlation_coefficient(v1, v2, expected):
    assert recommendations.calc_pearson_correlation_coefficient(
        v1, v2) == expected


@pytest.mark.parametrize('v1,v2', [
    (np.array([1, 2, 3]), np.array([1, 3, 2])),
])
def test_calc_similarity(v1, v2):
    assert recommendations.calc_similarity(
        v1, v2, metric='euclidean') == recommendations.calc_euclidean_similarity(v1, v2)
    assert recommendations.calc_similarity(
        v1, v2, metric='pearson') == recommendations.calc_pearson_correlation_coefficient(v1, v2)
