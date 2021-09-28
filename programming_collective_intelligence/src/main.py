from recommendations import calc_similarity
from critic import Critic


def main() -> None:
    critic = Critic()
    print(critic.critics_matrix, critic.user_dic, critic.item_dic)
    sim = calc_similarity(critic.critics_matrix[0], critic.critics_matrix[1])
    print(sim)


if __name__ == "__main__":
    main()
