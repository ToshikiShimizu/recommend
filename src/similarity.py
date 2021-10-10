import numpy as np


def calc_euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
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


def calc_euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """ユークリッド距離を利用した類似度を計算する

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル

    Returns:
        float: 類似度
    """
    # np.linalg.norm(v1-v2)としてもよい
    sim: float = 1 / (1 + calc_euclidean_distance(v1, v2))
    return sim


def calc_similarity_with_missing_value(v1: np.ndarray, v2: np.ndarray,
                                       metric: str = 'euclidean',
                                       missing_value: float = 0) -> float:
    """類似度を計算する関数。ベクトルのどちらかに欠損値が含まれる場合、該当idxは無視する。

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル
        metric (str, optional): スコア計算のメトリック
        missing_value (float, optional): 欠損値

    Returns:
        float: 類似度
    """
    idx: np.array = np.where((v1 != missing_value) & (v2 != missing_value))[0]
    if len(idx) == 0:
        return 0
    sim: float = calc_similarity(v1[idx], v2[idx], metric)
    return sim


def calc_similarity(v1: np.ndarray, v2: np.ndarray,
                    metric: str = 'euclidean') -> float:
    """類似度を計算する関数。metricによる分岐はこの関数で処理する。

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル
        metric (str, optional): スコア計算のメトリック

    Returns:
        float: 類似度
    """

    sim: float
    if (metric == 'euclidean'):
        sim = calc_euclidean_similarity(v1, v2)
    elif (metric == 'pearson'):
        sim = calc_pearson_correlation_coefficient(v1, v2)
    return sim


def calc_pearson_correlation_coefficient(v1: np.ndarray, v2: np.ndarray) \
        -> float:
    """ピアソン相関係数を計算する

    Args:
        v1 (np.ndarray): ベクトル
        v2 (np.ndarray): ベクトル

    Returns:
        float: ピアソン相関係数
    """
    v1_diff = (v1 - v1.mean())
    v2_diff = (v2 - v2.mean())
    square_denominator: float = np.sum(v1_diff**2) * np.sum(v2_diff**2)
    if square_denominator == 0:
        return 0
    coef: float = np.sum(v1_diff * v2_diff) / np.sqrt(square_denominator)
    return coef
