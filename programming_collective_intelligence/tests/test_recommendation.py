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


def test_get_ratings_for_one_object(recommendation_fixture):
    expected = recommendation_fixture._get_ratings_for_one_item(
        'Lady in the Water')
    results = recommendation_fixture.get_ratings_for_one_object(
        'item', 'Lady in the Water')
    assert (expected == results).all()
    expected = recommendation_fixture._get_ratings_for_one_user('Lisa Rose')
    results = recommendation_fixture.get_ratings_for_one_object(
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


def test_get_item_list_not_rated_by(recommendation_fixture):
    user = recommendation_fixture.get_user_list()[0]
    item_list = recommendation_fixture.get_item_list_not_rated_by(user)
    assert isinstance(item_list, list)  # list
    ratings = recommendation_fixture._get_ratings_for_one_user(user)
    expected_idx = np.where(ratings > 0)[0]
    result_idx = [recommendation_fixture._item_dic[k] for k in item_list]
    assert result_idx == expected_idx
