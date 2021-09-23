import numpy as np
from critics import critics
from typing import Tuple


def distance_euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    """ユークリッド距離を計算する

    Args:
        arg1 (np.ndarray): ベクトル
        arg2 (np.ndarray): ベクトル

    Returns:
        float: ユークリッド距離
    """
    # np.linalg.norm(v1-v2)としてもよい
    dist: float = np.sqrt(np.sum(np.square(v1 - v2)))
    return dist


def sim_distance_euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    """ユークリッド距離を利用した類似度を計算する

    Args:
        arg1 (np.ndarray): ベクトル
        arg2 (np.ndarray): ベクトル

    Returns:
        float: 類似度
    """
    # np.linalg.norm(v1-v2)としてもよい
    sim: float = 1 / (1 + distance_euclidean(v1, v2))
    return sim


def load_critics() -> Tuple[np.ndarray, dict, dict]:
    user_set = set()
    item_set = set()
    for user_k, user_v in critics.items():
        user_set.add(user_k)
        for item_k, item_v in user_v.items():
            item_set.add(item_k)
    user_list = sorted(list(user_set))
    item_list = sorted(list(item_set))
    user_dic: dict = dict(zip(user_list, range(len(user_list))))
    item_dic = dict(zip(item_list, range(len(item_list))))
    critics_matrix = np.zeros((len(user_dic), len(item_dic)))
    for user_k, user_v in critics.items():
        user_idx = user_dic[user_k]
        for item_k, item_v in user_v.items():
            item_idx = item_dic[item_k]
            critics_matrix[user_idx][item_idx] = item_v
    return critics_matrix, user_dic, item_dic

