from recommendations import load_critics, calc_similarity


def main() -> None:
    critics_matrix, user_dic, item_dic = load_critics()
    print(critics_matrix, user_dic, item_dic)
    sim = calc_similarity(critics_matrix[0], critics_matrix[1])
    print(sim)

if __name__ == "__main__":
    main()
