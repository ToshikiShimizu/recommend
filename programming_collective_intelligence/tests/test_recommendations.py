import pytest
from src.recommendations import sim_distance

def test_sim_distance():
    assert sim_distance() == 1
