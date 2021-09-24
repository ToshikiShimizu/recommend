import pytest
import numpy as np
from recommendations import distance_euclidean, sim_distance_euclidean


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 0),
    (np.array([0, 0]), np.array([1, 1]), np.sqrt(2)),
    (np.array([0, 0]), np.array([3, 4]), 5),
    (np.array([1]), np.array([2]), 1),
])
def test_distance_euclidean(v1, v2, expected):
    assert distance_euclidean(v1, v2) == expected


@pytest.mark.parametrize('v1,v2,expected', [
    (np.array([1, 2]), np.array([1, 2]), 1),
    (np.array([0, 0]), np.array([1, 1]), 1/(1+np.sqrt(2))),
    (np.array([0, 0]), np.array([3, 4]), 1/6),
    (np.array([1]), np.array([2]), 1/2),
])
def test_sim_distance_euclidean(v1, v2, expected):
    assert sim_distance_euclidean(v1, v2) == expected
