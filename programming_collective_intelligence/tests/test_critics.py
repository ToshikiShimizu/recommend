import pytest
import numpy as np
from critics import Critic


def test_load_critics():
    critic = Critic()
    assert critic.critics_matrix.shape[0] == len(critic.user_dic)
    assert critic.critics_matrix.shape[1] == len(critic.item_dic)
    assert np.all(0 <= critic.critics_matrix)
    assert np.all(critic.critics_matrix <= 5)
    assert np.max(critic.critics_matrix, axis=0).min() != 0  # 全てが0の列は存在してはならない
    assert np.max(critic.critics_matrix, axis=1).min() != 0  # 全てが0の行は存在してはならない
