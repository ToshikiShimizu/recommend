import numpy as np
from recommendation import Recommendation
import pytest


@pytest.fixture
def recommendation_fixture():
    recommendation = Recommendation()
    return recommendation


def test__load_ratings(recommendation_fixture):
    assert recommendation_fixture.matrix.shape[0] == len(
        recommendation_fixture._user_dic)
    assert recommendation_fixture.matrix.shape[1] == len(
        recommendation_fixture._item_dic)
    assert np.all(0 <= recommendation_fixture.matrix)
    assert np.all(recommendation_fixture.matrix <= 5)
    assert np.max(recommendation_fixture.matrix,
                  axis=0).min() != 0  # 全てが0の列は存在してはならない
    assert np.max(recommendation_fixture.matrix,
                  axis=1).min() != 0  # 全てが0の行は存在してはならない


def test__get_ratings_for_one_user(recommendation_fixture):
    assert isinstance(recommendation_fixture._get_ratings_for_one_user(
        'Lisa Rose'), np.ndarray)


def test__get_ratings_for_one_item(recommendation_fixture):
    assert isinstance(recommendation_fixture._get_ratings_for_one_item(
        'Lady in the Water'), np.ndarray)


def test__get_ratings_for_one_object(recommendation_fixture):
    expected = recommendation_fixture._get_ratings_for_one_item(
        'Lady in the Water')
    results = recommendation_fixture._get_ratings_for_one_object(
        'item', 'Lady in the Water')
    assert (expected == results).all()
    expected = recommendation_fixture._get_ratings_for_one_user('Lisa Rose')
    results = recommendation_fixture._get_ratings_for_one_object(
        'user', 'Lisa Rose')
    assert (expected == results).all()


def test_get_user_list(recommendation_fixture):
    assert isinstance(recommendation_fixture.get_user_list(), list)


def test_get_item_list(recommendation_fixture):
    assert isinstance(recommendation_fixture.get_item_list(), list)


def test_get_similar_objects(recommendation_fixture):
    # user
    user = recommendation_fixture.get_user_list()[0]
    user_sim = recommendation_fixture.get_similar_objects('user', user)
    assert isinstance(user_sim[0], list)
    assert isinstance(user_sim[0][0], str)
    assert isinstance(user_sim[0][1], float)
    assert user_sim[0][1] >= user_sim[-1][1]
    # item
    item = recommendation_fixture.get_item_list()[0]
    item_sim = recommendation_fixture.get_similar_objects('item', item)
    assert isinstance(item_sim[0], list)
    assert isinstance(item_sim[0][0], str)
    assert isinstance(item_sim[0][1], float)
    assert item_sim[0][1] >= item_sim[-1][1]


def test__get_item_list_not_rated_by(recommendation_fixture):
    user = recommendation_fixture.get_user_list()[0]
    not_rated_item_list = recommendation_fixture._get_item_list_not_rated_by(
        user)
    assert isinstance(not_rated_item_list, list)  # list
    ratings = recommendation_fixture._get_ratings_for_one_user(user)
    not_rated_item_idx = [recommendation_fixture._item_dic[k]
                          for k in not_rated_item_list]
    assert np.array(ratings)[not_rated_item_idx].sum() == 0


@pytest.mark.skip(reason="ratingsの予測を先に実装する")
def test_get_recommendations(recommendation_fixture):
    user = recommendation_fixture.get_user_list()[0]
    recommendations = recommendation_fixture.get_recommendations(
        user, based='user')
    assert isinstance(recommendations, list)  # list


def test_predict_ratings(recommendation_fixture):
    user = 'Michael Phillips'
    recommendations = recommendation_fixture.predict_ratings(
        user, based='user')
    assert 'Just My Luck' in recommendations


def test__get_average_rating_for_one_user(recommendation_fixture):
    user = recommendation_fixture.get_user_list()[0]
    rating = recommendation_fixture._get_average_rating_for_one_user(user)
    assert 1.0 <= rating <= 5.0


def test__get_average_rating_for_one_item(recommendation_fixture):
    item = recommendation_fixture.get_item_list()[0]
    rating = recommendation_fixture._get_average_rating_for_one_item(item)
    assert 1.0 <= rating <= 5.0


def test__get_user_who_rated_item(recommendation_fixture):
    item_list = recommendation_fixture.get_item_list()
    for item in item_list:
        user_list = recommendation_fixture._get_user_who_rated_item(item)
        assert isinstance(user_list, list)
        assert len(user_list) > 0  # データの仕様上、一人以上は評価している
