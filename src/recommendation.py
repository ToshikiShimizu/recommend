import numpy as np
from typing import Union, List, Dict
from ratings_data import ratings
from similarity import calc_similarity_with_missing_value
import dataclasses


@dataclasses.dataclass
class Recommendation:
    missing_value: float = 0

    def __post_init__(self):
        self._load_ratings()

    def _load_ratings(self):
        """ratingsをロードし、user-item行列を作成する

        Returns:
            Tuple[np.ndarray, dict, dict]: \
                user-item matrix,user_master, item_master
        """
        user_set = set()
        item_set = set()
        # ratingsはimportが必要な関数で、辞書を返す
        for user_k, user_v in ratings.items():
            user_set.add(user_k)
            for item_k, item_v in user_v.items():
                item_set.add(item_k)
        # ソートしてリストとして保持
        self._user_list: List[str] = sorted(list(user_set))
        self._item_list: List[str] = sorted(list(item_set))
        # 辞書に変換。valueとして0-originのindexを付与する
        self._user_dic: Dict[str, int] = dict(
            zip(self._user_list, range(len(self._user_list))))
        self._item_dic: Dict[str, int] = dict(
            zip(self._item_list, range(len(self._item_list))))
        # 行列は、行方向がユーザで列方向がアイテム
        self.matrix = np.zeros(
            (len(self._user_dic), len(self._item_dic)))
        # もう一度ループして、ratingを格納
        for user_k, user_v in ratings.items():
            user_idx = self._user_dic[user_k]
            for item_k, item_v in user_v.items():
                item_idx = self._item_dic[item_k]
                self.matrix[user_idx][item_idx] = item_v

    def _get_ratings_for_one_user(self, user_name: str) -> np.ndarray:
        """指定されたユーザの評価値ベクトルを返却する

        Args:
            user_name (str): キーとなるユーザ

        Returns:
            np.ndarray: 評価値ベクトル
        """
        return self.matrix[self._user_dic[user_name]]

    def _get_ratings_for_one_item(self, item_name: str) -> np.ndarray:
        """指定されたアイテムの評価値ベクトルを返却する

        Args:
            item_name (str): キーとなるアイテム

        Returns:
            np.ndarray: 評価値ベクトル
        """
        return self.matrix[:, self._item_dic[item_name]]

    def _get_ratings_for_one_object(self, object_type: str, object_name: str) -> np.ndarray:
        """指定されたユーザまたはアイテムの評価値ベクトルを返却する

        Args:
            object_type (str): 'user' or 'item'
            object_name (str): キーとなるユーザまたはアイテム

        Returns:
            np.ndarray: 評価値ベクトル
        """
        if object_type == 'user':
            return self._get_ratings_for_one_user(object_name)
        elif object_type == 'item':
            return self._get_ratings_for_one_item(object_name)

    def get_user_list(self) -> List[str]:
        """ユーザリストを返却する

        Returns:
            List[str]: ユーザリスト
        """
        return self._user_list

    def get_item_list(self) -> List[str]:
        """アイテムリストを返却する

        Returns:
            List[str]: アイテムリスト
        """
        return self._item_list

    def get_similar_objects(self, object_type: str, object_name: str) -> List[List[Union[str, float]]]:
        """類似度が高いユーザ(アイテム)とその類似度のリストを、類似度の降順に返却する

        Args:
            user (str): 類似リストを取得したいユーザ(アイテム)

        Returns:
            List[List[Union[str, float]]]: 類似ユーザ(アイテム)と類似度のリスト
        """
        v1 = self._get_ratings_for_one_object(object_type, object_name)
        sim_list = []
        if object_type == 'user':
            object_list = self.get_user_list()
        elif object_type == 'item':
            object_list = self.get_item_list()
        for key in object_list:
            v2 = self._get_ratings_for_one_object(object_type, key)
            sim = calc_similarity_with_missing_value(
                v1, v2, 'euclidean', self.missing_value)
            sim_list.append(sim)
        idx = np.argsort(np.array(sim_list))[::-1]
        object_sim = []
        for i in idx:
            object_sim.append([object_list[i], sim_list[i]])
        return object_sim

    def _get_item_list_not_rated_by(self, user: str) -> List[str]:
        """対象ユーザが未評価であるアイテムのリスト

        Args:
            user (str): 対象ユーザ

        Returns:
            List[str]: アイテムのリスト
        """
        ratings = self._get_ratings_for_one_user(user)
        idx = np.where(ratings == self.missing_value)[0]
        return np.array(self._item_list)[idx].tolist()

    def predict_ratings(self, user: str, based: str) -> Dict[str, float]:
        """対象ユーザの未評価アイテム集合の評価値を予測

        Args:
            user (str): 対象ユーザ

        Returns:
            Dict[str, float]: 未評価のアイテム集合とそれらの予測評価値
        """
        item_list = self._get_item_list_not_rated_by(user)
        user_bias = self._get_average_rating_for_one_user(user)
        for item in item_list:
            user_list = self._get_user_who_rated_item(item)
            print(item, user_list)
        ratings = {item: user_bias for item in item_list}  # TODO:値を正しく計算する
        print(ratings)
        return ratings

    def _get_average_rating_for_one_user(self, user_name: str) -> float:
        """対象ユーザの評価済みアイテムの平均評価値を計算する

        Args:
            user_name (str): 対象ユーザ

        Returns:
            float: 平均評価値
        """
        return self._get_ratings_for_one_user(user_name).mean()

    def _get_user_who_rated_item(self, item_name: str) -> List[str]:
        """対象アイテムを評価したユーザのリスト

        Args:
            item_name (str): 対象アイテム

        Returns:
            List[str]: ユーザのリスト
        """
        ratings = self._get_ratings_for_one_item(item_name)
        idx = np.where(ratings != self.missing_value)[0]
        return np.array(self._user_list)[idx].tolist()
