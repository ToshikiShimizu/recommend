from recommendation import Recommendation


def main() -> None:
    recommendation = Recommendation(missing_value=0)
    user_list = recommendation._get_user_list()
    for user_name in user_list:
        print(user_name, recommendation.get_recommendations(
            user_name, based='user'))
        print(user_name, recommendation.get_recommendations(
            user_name, based='item'))
        print(user_name, recommendation.get_recommendations(
            user_name, based='item', top_n=2))


if __name__ == "__main__":
    main()
