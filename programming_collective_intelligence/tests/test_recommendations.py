import pytest
from src.recommendations import sim_distance

def test_sim_distance():
    v1 = np.array([1,2])
    v2 = np.array([1,2])
    assert sim_distance(v1, v2) == 0
