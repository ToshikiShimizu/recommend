import pytest
import numpy as np
import critics


def test_load_critics():
    critics_matrix, user_dic, item_dic = critics.load_critics()
    assert critics_matrix.shape[0] == len(user_dic)
    assert critics_matrix.shape[1] == len(item_dic)
    assert np.all(0 <= critics_matrix)
    assert np.all(critics_matrix <= 5)
    assert np.max(critics_matrix, axis=0).min() != 0  # 全てが0の列は存在してはならない
    assert np.max(critics_matrix, axis=1).min() != 0  # 全てが0の行は存在してはならない
