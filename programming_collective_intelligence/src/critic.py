import numpy as np
from typing import Tuple
from critics_data import critics


class Critic:
    def __init__(self):
        self.load_critics()

    def load_critics(self):
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
        self.user_dic: dict = dict(zip(user_list, range(len(user_list))))
        self.item_dic = dict(zip(item_list, range(len(item_list))))
        # 行列は、行方向がユーザで列方向がアイテム
        self.matrix = np.zeros(
            (len(self.user_dic), len(self.item_dic)))
        # もう一度ループして、ratingを格納
        for user_k, user_v in critics.items():
            user_idx = self.user_dic[user_k]
            for item_k, item_v in user_v.items():
                item_idx = self.item_dic[item_k]
                self.matrix[user_idx][item_idx] = item_v

    def get_critics_for_one_user(self, user_name):
        return self.matrix[self.user_dic[user_name]]



    def get_critics_for_one_item(self, item_name):
        return self.matrix[:,self.item_dic[item_name]]
