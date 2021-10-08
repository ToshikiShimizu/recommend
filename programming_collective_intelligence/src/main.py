from similarity import calc_similarity
from recommendation import Recommendation
import numpy as np


def main() -> None:
    recommendation = Recommendation(-1)

    # get_similar_users
    user_list = recommendation.get_user_list()
    user_name = user_list[0]
    print(recommendation.get_similar_objects('user', user_name))

    # get_similar_items
    item_list = recommendation.get_item_list()
    item_name = item_list[0]
    print(recommendation.get_similar_objects('item', item_name))


if __name__ == "__main__":
    main()
