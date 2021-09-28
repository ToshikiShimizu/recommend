import numpy as np
from typing import Tuple
from critics_data import critics


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
