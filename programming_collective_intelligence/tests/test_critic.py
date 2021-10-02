import numpy as np
from critic import Critic
import pytest


@pytest.fixture
def critic_fixture():
    critic = Critic()
    return critic


def test_load_critics(critic_fixture):
    assert critic_fixture.matrix.shape[0] == len(critic_fixture.user_dic)
    assert critic_fixture.matrix.shape[1] == len(critic_fixture.item_dic)
    assert np.all(0 <= critic_fixture.matrix)
    assert np.all(critic_fixture.matrix <= 5)
    assert np.max(critic_fixture.matrix, axis=0).min() != 0  # 全てが0の列は存在してはならない
    assert np.max(critic_fixture.matrix, axis=1).min() != 0  # 全てが0の行は存在してはならない


def test_get_critics_for_one_user(critic_fixture):
    assert isinstance(critic_fixture.get_critics_for_one_user(
        'Lisa Rose'), np.ndarray)


def test_get_critics_for_one_item(critic_fixture):
    assert isinstance(critic_fixture.get_critics_for_one_item(
        'Lady in the Water'), np.ndarray)


def test_get_user_list(critic_fixture):
    assert isinstance(critic_fixture.get_user_list(), list)


def test_get_item_list(critic_fixture):
    assert isinstance(critic_fixture.get_item_list(), list)
