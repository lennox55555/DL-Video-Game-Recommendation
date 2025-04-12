import pandas as pd
import numpy as np

class NaiveGameRecommender:
    def __init__(self, user_data_path):
        """
        load ratings csv.
        """
        self.user_data = pd.read_csv(user_data_path)
        self.avg_ratings = None
        self.compute_average_ratings()

    def compute_average_ratings(self):
        """
        compute each game's average rating.
        """
        self.avg_ratings = self.user_data.groupby("game_title")["rating"].mean().reset_index()

    def get_recommendations(self, target_game, top_n=3):
        """
        recommend top_n games by average rating.
        """
        # remove target game.
        recs = self.avg_ratings[self.avg_ratings["game_title"] != target_game]
        # sort by highest rating.
        recs = recs.sort_values(by="rating", ascending=False)
        return recs["game_title"].head(top_n).tolist()

    def evaluate_rmse(self, test_ratio=0.2, random_state=42):
        """
        evaluate rmse on held-out ratings.
        """
        # shuffle and split data into test and train sets.
        data_shuffled = self.user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_size = int(len(data_shuffled) * test_ratio)
        test_set = data_shuffled.iloc[:test_size].copy()
        train_set = data_shuffled.iloc[test_size:].copy()

        # compute average rating per game from training data.
        train_avg = train_set.groupby("game_title")["rating"].mean().to_dict()
        global_avg = train_set["rating"].mean()

        # predict test ratings using the game average (or global average).
        test_set["predicted"] = test_set["game_title"].apply(lambda g: train_avg.get(g, global_avg))

        # calculate rmse.
        rmse = np.sqrt(np.mean((test_set["rating"] - test_set["predicted"])**2))
        return rmse

if __name__ == '__main__':
    recommender = NaiveGameRecommender("../data/fake_user_data.csv")
    target_game = "minecraft"
    recommendations = recommender.get_recommendations(target_game=target_game, top_n=3)
    print("we predict you will like these games:")
    print(recommendations)
    
    rmse_value = recommender.evaluate_rmse(test_ratio=0.2)
    print("\nevaluation metrics:")
    print(f"rmse on held-out ratings: {rmse_value:.4f}")
