import numpy as np
from recommendation import Recommendation
import pytest


@pytest.fixture
def recommendation_fixture():
    recommendation = Recommendation()
    return recommendation


@pytest.fixture
def recommendation_fixture_special_user():
    # 評価したアイテムが1つもないユーザのみのデータ
    recommendation = Recommendation(file_name='data/ratings_special_user.json')
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
    user_name = recommendation_fixture.get_user_list()[0]
    user_sim = recommendation_fixture.get_similar_objects('user', user_name)
    assert isinstance(user_sim[0], list)
    assert isinstance(user_sim[0][0], str)
    assert isinstance(user_sim[0][1], float)
    assert user_sim[0][1] >= user_sim[-1][1]
    # item
    item_name = recommendation_fixture.get_item_list()[0]
    item_sim = recommendation_fixture.get_similar_objects('item', item_name)
    assert isinstance(item_sim[0], list)
    assert isinstance(item_sim[0][0], str)
    assert isinstance(item_sim[0][1], float)
    assert item_sim[0][1] >= item_sim[-1][1]


def test__get_item_list_not_rated_by(recommendation_fixture):
    user_name = recommendation_fixture.get_user_list()[0]
    not_rated_item_list = recommendation_fixture._get_item_list_not_rated_by(
        user_name)
    assert isinstance(not_rated_item_list, list)  # list
    ratings = recommendation_fixture._get_ratings_for_one_user(user_name)
    not_rated_item_idx = [recommendation_fixture._item_dic[k]
                          for k in not_rated_item_list]

    assert np.all(np.array(ratings)[not_rated_item_idx]
                  == recommendation_fixture.missing_value)


def test__get_item_list_rated_by(recommendation_fixture):
    user_name = recommendation_fixture.get_user_list()[0]
    rated_item_list = recommendation_fixture._get_item_list_rated_by(
        user_name)
    assert isinstance(rated_item_list, list)  # list
    ratings = recommendation_fixture._get_ratings_for_one_user(user_name)
    rated_item_idx = [recommendation_fixture._item_dic[k]
                      for k in rated_item_list]
    assert np.all(np.array(ratings)[rated_item_idx]
                  != recommendation_fixture.missing_value)


def test_get_recommendations(recommendation_fixture):
    user_name = recommendation_fixture.get_user_list()[4]
    recommendations = recommendation_fixture.get_recommendations(
        user_name, based='user')
    assert isinstance(recommendations, list)  # list
    recommendations = recommendation_fixture.get_recommendations(
        user_name, based='user', top_n=1)
    assert len(recommendations) == 1


def test_predict_ratings(recommendation_fixture):
    user_name = 'Michael Phillips'
    recommendations = recommendation_fixture.predict_ratings(
        user_name, based='user')
    assert 'Just My Luck' in recommendations
    for rating in recommendations.values():
        assert 1.0 <= rating <= 5.0
    user_name = 'Michael Phillips'
    recommendations = recommendation_fixture.predict_ratings(
        user_name, based='item')
    assert 'Just My Luck' in recommendations
    for rating in recommendations.values():
        assert 1.0 <= rating <= 5.0


def test__get_average_rating_for_one_user(recommendation_fixture, recommendation_fixture_special_user):
    user_name = recommendation_fixture.get_user_list()[0]
    rating = recommendation_fixture._get_average_rating_for_one_user(user_name)
    assert 1.0 <= rating <= 5.0
    with pytest.raises(Exception):
        user_name = recommendation_fixture_special_user.get_user_list()[0]
        recommendation_fixture_special_user._get_average_rating_for_one_user(
            user_name)


def test__get_average_rating_for_one_item(recommendation_fixture):
    item_name = recommendation_fixture.get_item_list()[0]
    rating = recommendation_fixture._get_average_rating_for_one_item(item_name)
    assert 1.0 <= rating <= 5.0


def test__get_user_who_rated_item(recommendation_fixture):
    item_list = recommendation_fixture.get_item_list()
    for item_name in item_list:
        user_list = recommendation_fixture._get_user_who_rated_item(item_name)
        assert isinstance(user_list, list)
        assert len(user_list) > 0  # データの仕様上、一人以上は評価している


def test__get_rating(recommendation_fixture):
    item_list = recommendation_fixture.get_item_list()
    for item_name in item_list:
        user_list = recommendation_fixture._get_user_who_rated_item(item_name)
        for user_name in user_list:
            rating = recommendation_fixture._get_rating(user_name, item_name)
            assert 1.0 <= rating <= 5.0


def test_calc_similarity_with_missing_value_by_name(recommendation_fixture):
    user_list = recommendation_fixture.get_user_list()
    similarity = recommendation_fixture.calc_similarity_with_missing_value_by_name(
        'user', user_list[0], user_list[-1])
    assert -1.0 <= similarity <= 1.0


def test_predict_ratings_with_baseline_estimation(recommendation_fixture):
    user_name = 'Michael Phillips'
    recommendations = recommendation_fixture.predict_ratings_with_baseline_estimation(
        user_name, based='user')
    assert 'Just My Luck' in recommendations
    for rating in recommendations.values():
        assert 1.0 <= rating <= 5.0
    user_name = 'Michael Phillips'
    recommendations = recommendation_fixture.predict_ratings_with_baseline_estimation(
        user_name, based='item')
    assert 'Just My Luck' in recommendations
    for rating in recommendations.values():
        assert 1.0 <= rating <= 5.0