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


def calc_similarity_with_missing_value(v1: np.ndarray, v2: np.ndarray, 
    metric: str = 'euclidean', missing_value: float = 0) -> float:
    """[summary]

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル
        metric (str, optional): スコア計算のメトリック
        missing_value (float, optional): 欠損値

    Returns:
        float: 類似度
    """
    idx = np.where((v1 != missing_value) & (v2 != missing_value))[0]
    if len(idx) == 0:
        return 0
    sim: float = calc_similarity(v1[idx], v2[idx], metric)
    return sim


def calc_similarity(v1: np.ndarray, v2: np.ndarray, \
    metric: str = 'euclidean') -> float:
    """[summary]

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル
        metric (str, optional): スコア計算のメトリック

    Returns:
        float: 類似度
    """

    sim: float
    if (metric == 'euclidean'):
        sim = sim_distance_euclidean(v1, v2)
    elif (metric == 'pearson'):
        sim = calc_pearson_correlation_coefficient(v1, v2)
    return sim


def calc_pearson_correlation_coefficient(v1: np.ndarray, v2: np.ndarray) -> float:
    """ピアソン相関係数を計算する

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル

    Returns:
        float: ピアソン相関係数
    """
    v1_diff = (v1 - v1.mean())
    v2_diff = (v2 - v2.mean())
    square_denominator = np.sum(v1_diff**2) * np.sum(v2_diff**2) 
    if square_denominator == 0:
        return 0
    coef: float = np.sum(v1_diff * v2_diff) / np.sqrt(square_denominator)
    return coef