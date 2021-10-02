from recommendations import calc_similarity
from critic import Critic
import numpy as np


def main() -> None:
    critic = Critic()

    # top_matches
    user_list = critic.get_user_list()
    user_name = user_list[0]
    print(critic.top_matches(user_name))


if __name__ == "__main__":
    main()
