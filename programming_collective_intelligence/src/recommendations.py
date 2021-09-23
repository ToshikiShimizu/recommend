import numpy as np
from critics import critics
from typing import Tuple


def distance_euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    """ユークリッド距離を計算する

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル

    Returns:
        float: ユークリッド距離
    """
    # np.linalg.norm(v1-v2)としてもよい
    dist: float = np.sqrt(np.sum(np.square(v1 - v2)))
    return dist


def sim_distance_euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    """ユークリッド距離を利用した類似度を計算する

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル

    Returns:
        float: 類似度
    """
    # np.linalg.norm(v1-v2)としてもよい
    sim: float = 1 / (1 + distance_euclidean(v1, v2))
    return sim


def load_critics() -> Tuple[np.ndarray, dict, dict]:
    """criticsをロードし、user-item行列に変換して返却する

    Returns:
        Tuple[np.ndarray, dict, dict]: \
            user-item matrix,user_master, item_master
    """
    user_set = set()
    item_set = set()
    # criticsはimportが必要な関数で、辞書を返す
    for user_k, user_v in critics.items():
        user_set.add(user_k)
        for item_k, item_v in user_v.items():
            item_set.add(item_k)
    # ソートしてリストとして保持
    user_list = sorted(list(user_set))
    item_list = sorted(list(item_set))
    # 辞書に変換。valueとして0-originのindexを付与する
    user_dic: dict = dict(zip(user_list, range(len(user_list))))
    item_dic = dict(zip(item_list, range(len(item_list))))
    # 行列は、行方向がユーザで列方向がアイテム
    critics_matrix = np.zeros((len(user_dic), len(item_dic)))
    # もう一度ループして、ratingを格納
    for user_k, user_v in critics.items():
        user_idx = user_dic[user_k]
        for item_k, item_v in user_v.items():
            item_idx = item_dic[item_k]
            critics_matrix[user_idx][item_idx] = item_v
    return critics_matrix, user_dic, item_dic
