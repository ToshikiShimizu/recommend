import numpy as np
from typing import Union, List, Dict
from critics_data import critics
from recommendations import calc_similarity


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

    def get_critics_for_one_object(self, object_type: str, object_name: str) -> np.ndarray:
        """指定されたユーザまたはアイテムの評価値ベクトルを返却する

        Args:
            object_type (str): 'user' or 'item'
            object_name (str): キーとなるユーザまたはアイテム

        Returns:
            np.ndarray: 評価値ベクトル
        """
        if object_type == 'user':
            return self.get_critics_for_one_user(object_name)
        elif object_type == 'item':
            return self.get_critics_for_one_item(object_name)

    def get_user_list(self) -> List[str]:
        """ユーザリストを返却する

        Returns:
            List[str]: ユーザリスト
        """
        return self.user_list

    def get_item_list(self) -> List[str]:
        """アイテムリストを返却する

        Returns:
            List[str]: アイテムリスト
        """
        return self.item_list

    def get_similar_objects(self, object_type: str, object_name: str) -> List[List[Union[str, float]]]:
        """類似度が高いユーザとその類似度のリストを、類似度の降順に返却する

        Args:
            user (str): 類似ユーザのリストを取得したいユーザ

        Returns:
            List[List[Union[str, float]]]: 類似ユーザと類似度のリスト
        """
        v1 = self.get_critics_for_one_object(object_type, object_name)
        sim_list = []
        if object_type == 'user':
            object_list = self.get_user_list()
        elif object_type == 'item':
            object_list = self.get_item_list()
        for key in object_list:
            v2 = self.get_critics_for_one_object(object_type, key)
            sim = calc_similarity(v1, v2)
            sim_list.append(sim)
        idx = np.argsort(np.array(sim_list))[::-1]
        object_sim = []
        for i in idx:
            object_sim.append([object_list[i], sim_list[i]])
        return object_sim
