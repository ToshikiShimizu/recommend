from recommendations import calc_similarity
from critic import Critic


def main() -> None:
    critic = Critic()
    print(critic.matrix, critic.user_dic, critic.item_dic)
    sim = calc_similarity(critic.matrix[0], critic.matrix[1])
    print(sim)


if __name__ == "__main__":
    main()
