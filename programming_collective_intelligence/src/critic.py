import numpy as np
from typing import Tuple, List, Dict
from critics_data import critics


class Critic:
    def __init__(self):
        self.load_critics()

    def load_critics(self):
        """criticsをロードし、user-item行列を作成する

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
        self.user_list: List[str] = sorted(list(user_set))
        self.item_list: List[str] = sorted(list(item_set))
        # 辞書に変換。valueとして0-originのindexを付与する
        self.user_dic: Dict[str, int] = dict(
            zip(self.user_list, range(len(self.user_list))))
        self.item_dic: Dict[str, int] = dict(
            zip(self.item_list, range(len(self.item_list))))
        # 行列は、行方向がユーザで列方向がアイテム
        self.matrix = np.zeros(
            (len(self.user_dic), len(self.item_dic)))
        # もう一度ループして、ratingを格納
        for user_k, user_v in critics.items():
            user_idx = self.user_dic[user_k]
            for item_k, item_v in user_v.items():
                item_idx = self.item_dic[item_k]
                self.matrix[user_idx][item_idx] = item_v

    def get_critics_for_one_user(self, user_name: str) -> np.ndarray:
        """指定されたユーザの評価値ベクトルを返却する

        Args:
            user_name (str): キーとなるユーザ

        Returns:
            np.ndarray: 評価値ベクトル
        """
        return self.matrix[self.user_dic[user_name]]

    def get_critics_for_one_item(self, item_name: str) -> np.ndarray:
        """指定されたアイテムの評価値ベクトルを返却する

        Args:
            item_name (str): キーとなるアイテム

        Returns:
            np.ndarray: 評価値ベクトル
        """
        return self.matrix[:, self.item_dic[item_name]]

    def get_user_list(self) -> List[int]:
        """ユーザリストを返却する

        Returns:
            List[int]: ユーザリスト
        """
        return self.user_list

    def get_item_list(self) -> List[int]:
        """アイテムリストを返却する

        Returns:
            List[int]: アイテムリスト
        """
        return self.item_list
