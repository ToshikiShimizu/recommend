from recommendations import load_critics


def main() -> None:
    critics_matrix, user_dic, item_dic = load_critics()
    print(critics_matrix, user_dic, item_dic)


if __name__ == "__main__":
    main()