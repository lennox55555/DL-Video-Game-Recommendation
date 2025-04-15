import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import random

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

    def get_recommendations(self, target_game, top_n=5):
        """
        recommend top_n games by average rating.
        """
        # remove target game.
        recs = self.avg_ratings[self.avg_ratings["game_title"] != target_game]
        # sort by highest rating.
        recs = recs.sort_values(by="rating", ascending=False)
        return recs["game_title"].head(top_n).tolist()
    
    def mean_average_precision_at_k(self, actual, predicted, k):
        """
        Calculate Mean Average Precision at k
        """
        avg_precisions = []
        for a, p in zip(actual, predicted):
            if not a:
                continue
            hits = 0
            sum_precisions = 0.0
            for i, pred in enumerate(p[:k]):
                if pred in a:
                    hits += 1
                    sum_precisions += hits / (i + 1.0)
            avg_precisions.append(sum_precisions / min(len(a), k))
        return np.mean(avg_precisions)

    def ndcg_at_k(self, actual, predicted, k):
        """
        Calculate Normalized Discounted Cumulative Gain at k
        """
        def dcg(rel):
            return sum([(2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rel)])

        ndcgs = []
        for a, p in zip(actual, predicted):
            relevance = [1 if item in a else 0 for item in p[:k]]
            ideal_relevance = sorted(relevance, reverse=True)
            ndcgs.append(dcg(relevance) / (dcg(ideal_relevance) or 1))
        return np.mean(ndcgs)

    def evaluate_metrics(self, test_ratio=0.2, k=10, random_state=42):
        """
        Evaluate model with multiple metrics: MSE, RMSE, R^2, MAP@k, NDCG@k
        """
        # train and test
        data_shuffled = self.user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_size = int(len(data_shuffled) * test_ratio)
        test_set = data_shuffled.iloc[:test_size].copy()
        train_set = data_shuffled.iloc[test_size:].copy()

        # getompute avg rating per game from training data
        train_avg = train_set.groupby("game_title")["rating"].mean().to_dict()
        global_avg = train_set["rating"].mean()

        test_set["predicted"] = test_set["game_title"].apply(lambda g: train_avg.get(g, global_avg))

        # calc regression metrics
        mse = mean_squared_error(test_set["rating"], test_set["predicted"])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_set["rating"], test_set["predicted"])

        # calc ranking metrics
        user_game_dict = defaultdict(set)
        for _, row in train_set.iterrows():
            user_id = row['user_id']
            game = row['game_title']
            user_game_dict[user_id].add(game)
        
        all_games = list(set(self.user_data['game_title']))
        
        # generate recommendations and compare with test set
        actual_items = []
        predicted_items = []
        
        for user_id in user_game_dict.keys():
            user_test_games = test_set[test_set['user_id'] == user_id]['game_title'].tolist()
            if not user_test_games:
                continue
                
            user_train_games = list(user_game_dict[user_id])
            
            candidate_games = [g for g in all_games if g not in user_train_games]
            
            game_scores = [(g, train_avg.get(g, global_avg)) for g in candidate_games]
            
            # sort by ppred rating
            ranked_games = [g for g, _ in sorted(game_scores, key=lambda x: x[1], reverse=True)]
            
            actual_items.append(user_test_games)
            predicted_items.append(ranked_games[:k])
        
        # calc metrics
        mapk = self.mean_average_precision_at_k(actual_items, predicted_items, k)
        ndcg = self.ndcg_at_k(actual_items, predicted_items, k)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2,
            f'MAP@{k}': mapk,
            f'NDCG@{k}': ndcg
        }

if __name__ == '__main__':
    recommender = NaiveGameRecommender("../data/fake_user_data.csv")
    target_game = "minecraft"
    recommendations = recommender.get_recommendations(target_game=target_game, top_n=5)
    print("We predict you will like these games:")
    print(recommendations)
    
    print("\nCalculating all metrics...")
    metrics = recommender.evaluate_metrics(test_ratio=0.2, k=10)
    
    print("\nEvaluation Metrics:")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R^2: {metrics['R^2']:.4f}")
    print(f"MAP@10: {metrics['MAP@10']:.4f}")
    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
