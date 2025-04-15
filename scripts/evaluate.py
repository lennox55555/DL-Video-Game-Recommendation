# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import random

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
