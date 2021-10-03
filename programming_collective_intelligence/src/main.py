from recommendations import calc_similarity
from critic import Critic
import numpy as np


def main() -> None:
    critic = Critic()

    # get_similar_users
    user_list = critic.get_user_list()
    user_name = user_list[0]
    print(critic.get_similar_objects('user', user_name))

    # get_similar_items
    item_list = critic.get_item_list()
    item_name = item_list[0]
    print(critic.get_similar_objects('item', item_name))


if __name__ == "__main__":
    main()
