# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import random
import json

from torch.utils.data import DataLoader

from deep_learning_training import RecommendationDataset, NCFRecommendationSystem
import pandas as pd

torch.manual_seed(42)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Using device: {device}')

class NaiveGameRecommender:
    def __init__(self, user_data_path):
        self.user_data = pd.read_csv(user_data_path)
        self.avg_ratings = None
        self.compute_average_ratings()

    def compute_average_ratings(self):
        self.avg_ratings = self.user_data.groupby("game_title")["rating"].mean().reset_index()

    def get_recommendations(self, target_game, top_n=5):
        recs = self.avg_ratings[self.avg_ratings["game_title"] != target_game]
        recs = recs.sort_values(by="rating", ascending=False)
        return recs["game_title"].head(top_n).tolist()

    def mean_average_precision_at_k(self, actual, predicted, k):
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
        def dcg(rel):
            return sum([(2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rel)])
        ndcgs = []
        for a, p in zip(actual, predicted):
            relevance = [1 if item in a else 0 for item in p[:k]]
            ideal_relevance = sorted(relevance, reverse=True)
            ndcgs.append(dcg(relevance) / (dcg(ideal_relevance) or 1))
        return np.mean(ndcgs)

    def evaluate_metrics(self, test_ratio=0.2, k=10, random_state=42):
        data_shuffled = self.user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_size = int(len(data_shuffled) * test_ratio)
        test_set = data_shuffled.iloc[:test_size].copy()
        train_set = data_shuffled.iloc[test_size:].copy()

        train_avg = train_set.groupby("game_title")["rating"].mean().to_dict()
        global_avg = train_set["rating"].mean()

        test_set["predicted"] = test_set["game_title"].apply(lambda g: train_avg.get(g, global_avg))

        mse = mean_squared_error(test_set["rating"], test_set["predicted"])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_set["rating"], test_set["predicted"])

        user_game_dict = defaultdict(set)
        for _, row in train_set.iterrows():
            user_id = row['user_id']
            game = row['game_title']
            user_game_dict[user_id].add(game)

        all_games = list(set(self.user_data['game_title']))

        actual_items = []
        predicted_items = []

        for user_id in user_game_dict.keys():
            user_test_games = test_set[test_set['user_id'] == user_id]['game_title'].tolist()
            if not user_test_games:
                continue
            user_train_games = list(user_game_dict[user_id])
            candidate_games = [g for g in all_games if g not in user_train_games]
            game_scores = [(g, train_avg.get(g, global_avg)) for g in candidate_games]
            ranked_games = [g for g, _ in sorted(game_scores, key=lambda x: x[1], reverse=True)]
            actual_items.append(user_test_games)
            predicted_items.append(ranked_games[:k])

        mapk = self.mean_average_precision_at_k(actual_items, predicted_items, k)
        ndcg = self.ndcg_at_k(actual_items, predicted_items, k)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2,
            f'MAP@{k}': mapk,
            f'NDCG@{k}': ndcg
        }

def evaluate_naive_model(user_data_path, test_ratio=0.2, k=10):
    recommender = NaiveGameRecommender(user_data_path)
    print("Evaluating Naive Model...")

    metrics = recommender.evaluate_metrics(test_ratio=test_ratio, k=k)

    print("\n📊 Naive Model Evaluation Metrics:")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R^2: {metrics['R^2']:.4f}")
    print(f"MAP@{k}: {metrics[f'MAP@{k}']:.4f}")
    print(f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")

    return metrics


class RecommendationDataset(torch.utils.data.Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df['user_idx'].astype(int).values
        self.games = ratings_df['game_idx'].astype(int).values
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'game_idx': torch.tensor(self.games[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }

def mean_average_precision_at_k(actual, predicted, k):
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

def ndcg_at_k(actual, predicted, k):
    def dcg(rel):
        return sum([(2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rel)])

    ndcgs = []
    for a, p in zip(actual, predicted):
        relevance = [1 if item in a else 0 for item in p[:k]]
        ideal_relevance = sorted(relevance, reverse=True)
        ndcgs.append(dcg(relevance) / (dcg(ideal_relevance) or 1))
    return np.mean(ndcgs)

def evaluate(model, ratings_df, all_game_indices, device='cpu', k=10, num_neg_samples=100):
    model.eval()
    dataset = RecommendationDataset(ratings_df)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []

    user_game_dict = defaultdict(set)
    for _, row in ratings_df.iterrows():
        user_game_dict[int(row['user_idx'])].add(int(row['game_idx']))

    with torch.no_grad():
        for batch in loader:
            user_idx = batch['user_idx'].to(device)
            game_idx = batch['game_idx'].to(device)
            ratings = batch['rating'].cpu().numpy()
            preds = model(user_idx, game_idx).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(ratings)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    print(f"Regression Metrics → MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Ranking Evaluation
    actual_items = []
    predicted_items = []

    user_ids = list(user_game_dict.keys())
    for user in user_ids:
        pos_items = list(user_game_dict[user])
        neg_items = list(set(all_game_indices) - user_game_dict[user])
        sampled_neg = random.sample(neg_items, min(num_neg_samples, len(neg_items)))
        candidate_items = pos_items + sampled_neg

        user_tensor = torch.tensor([user] * len(candidate_items), dtype=torch.long).to(device)
        game_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = model(user_tensor, game_tensor).cpu().numpy()
        ranked_items = [x for _, x in sorted(zip(scores, candidate_items), reverse=True)]

        actual_items.append(pos_items)
        predicted_items.append(ranked_items[:k])

    mapk = mean_average_precision_at_k(actual_items, predicted_items, k)
    ndcg = ndcg_at_k(actual_items, predicted_items, k)

    print(f"Ranking Metrics → MAP@{k}: {mapk:.4f} | NDCG@{k}: {ndcg:.4f}")
    return mse, rmse, r2, mapk, ndcg


def main():
    k = 10
    test_dataset = "data/inference_data/test_dataset.csv"
    evaluate_naive_model(test_dataset, test_ratio=0.2, k=k)

    cleaned_games_df = pd.read_csv('data/inference_data/cleaned_games_df.csv')
    with open('data/inference_data/user_mapping.json') as f:
        user_mapping = json.load(f)
    with open('data/inference_data/game_mapping.json') as f:
        game_mapping = json.load(f)

    # restore model
    num_users = len(user_mapping)
    num_games = len(game_mapping)

    test_df = pd.read_csv(test_dataset)

    print('\nEvaluating Deep Learning Model...')
    model = NCFRecommendationSystem(num_users, num_games)
    model.load_state_dict(torch.load('models/deep_learning_model_500_combined.pth'))
    model.to(device)

    evaluate(model, test_df, all_game_indices=list(game_mapping.values()), device=device, k=k)

if __name__ == '__main__':
    main()