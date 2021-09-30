from recommendations import calc_similarity
from critic import Critic
import numpy as np


def main() -> None:
    critic = Critic()
    print(critic.matrix, critic.user_dic, critic.item_dic)
    sim = calc_similarity(critic.matrix[0], critic.matrix[1])
    print(sim)

    # top_matches
    user_list = list(critic.user_dic.keys())
    user_name = user_list[0]
    print(user_name)
    v1 = critic.get_critics_for_one_user(user_name)
    sims = []
    for key in critic.user_dic.keys():
        v2 = critic.get_critics_for_one_user(key)
        sim = calc_similarity(v1, v2)
        sims.append(sim)
    idx = np.argsort(sims)[::-1]
    for i in idx:
        print(user_list[i], sims[i])


if __name__ == "__main__":
    main()
