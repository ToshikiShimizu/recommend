import numpy as np


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
