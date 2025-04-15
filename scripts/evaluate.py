"""
Evaluation module for game recommendation systems.

This module provides functionality to evaluate different recommendation system approaches:
1. Traditional ML-based recommender
2. Naive rating-based recommender
3. Deep learning (NCF) recommender

The module compares performance metrics such as MSE, RMSE, R2, MAP@k, and NDCG@k.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import random
import json
import os
import joblib
import pandas as pd

from deep_learning_training import RecommendationDataset, NCFRecommendationSystem

# Set seed for reproducibility
torch.manual_seed(42)

# Set device for computation
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Using device: {device}')


class TraditionalRecommender:
    """
    Traditional machine learning-based recommender system.
    
    Uses pre-trained model to make recommendations based on game features.
    
    Attributes:
        model: The trained ML model for making predictions
        feature_names: Names of features used by the model
        features_df: DataFrame containing feature data for games
    """
    
    def __init__(self, model_path="models/traditional_model.pkl", feature_path="data/inference_data/traditional_feature_matrix.csv", title_path="data/inference_data/game_titles.csv"):
        """
        Initialize the traditional recommender system.
        
        Args:
            model_path (str): Path to the trained model file
            feature_path (str): Path to the feature matrix CSV file
            title_path (str): Path to the game titles CSV file
        """
        self.model = self.load_model(model_path)
        self.feature_names = self.model.feature_names_in_
        self.features_df = self.load_feature_data(feature_path, title_path)

    def load_model(self, model_path):
        """
        Load the trained machine learning model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            The loaded model object
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        return joblib.load(model_path)

    def load_feature_data(self, feature_matrix_path, titles_path):
        """
        Load feature data and game titles.
        
        Args:
            feature_matrix_path (str): Path to the feature matrix CSV file
            titles_path (str): Path to the game titles CSV file
            
        Returns:
            DataFrame containing features and titles
            
        Raises:
            ValueError: If the titles CSV doesn't have a 'Title' column
        """
        features_df = pd.read_csv(feature_matrix_path)
        titles_df = pd.read_csv(titles_path)

        if 'Title' not in titles_df.columns:
            raise ValueError("Game titles CSV must contain a 'Title' column.")

        features_df['Title'] = titles_df['Title']
        return features_df

    def recommend_games(self, liked_games=None, top_n=10):
        """
        Generate game recommendations based on liked games.
        
        Args:
            liked_games (list, optional): List of game titles the user likes
            top_n (int): Number of recommendations to generate
            
        Returns:
            list: Top recommended games with their predicted scores
        """
        df = self.features_df.copy()

        if liked_games:
            df = df[~df['Title'].isin(liked_games)]
        else:
            print("⚠️ No liked games provided — returning global top recommendations.")

        X = df[self.feature_names]
        df['predicted_score'] = self.model.predict(X)

        top_recs = df.sort_values(by='predicted_score', ascending=False).head(top_n)

        return [
            {"title": row["Title"], "description": f"Predicted Score: {row['predicted_score']:.2f}"}
            for _, row in top_recs.iterrows()
        ]


def evaluate_traditional_model(model_path, feature_path, title_path, liked_games=None, top_n=10):
    """
    Evaluate the traditional ML-based recommender system.
    
    Args:
        model_path (str): Path to the trained model file
        feature_path (str): Path to the feature matrix CSV file
        title_path (str): Path to the game titles CSV file
        liked_games (list, optional): Games the user likes
        top_n (int): Number of recommendations to generate
        
    Returns:
        list: The top recommendations from the model
    """
    print("\nEvaluating Traditional Model...")
    recommender = TraditionalRecommender(model_path, feature_path, title_path)
    recommendations = recommender.recommend_games(liked_games=liked_games, top_n=top_n)

    print(f"\n📊 Top {top_n} Recommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['title']} - {rec['description']}")

    return recommendations


class NaiveGameRecommender:
    """
    Naive rating-based game recommender system.
    
    Makes recommendations based on average game ratings.
    
    Attributes:
        user_data: DataFrame containing user ratings
        avg_ratings: DataFrame containing average ratings per game
    """
    
    def __init__(self, user_data_path):
        """
        Initialize the naive recommender system.
        
        Args:
            user_data_path (str): Path to the user ratings CSV file
        """
        self.user_data = pd.read_csv(user_data_path)
        self.avg_ratings = None
        self.compute_average_ratings()

    def compute_average_ratings(self):
        """
        Compute average ratings for each game.
        """
        self.avg_ratings = self.user_data.groupby("game_title")["rating"].mean().reset_index()

    def get_recommendations(self, target_game, top_n=5):
        """
        Get recommendations excluding the target game.
        
        Args:
            target_game (str): The game to exclude
            top_n (int): Number of recommendations to return
            
        Returns:
            list: Top game recommendations
        """
        recs = self.avg_ratings[self.avg_ratings["game_title"] != target_game]
        recs = recs.sort_values(by="rating", ascending=False)
        return recs["game_title"].head(top_n).tolist()

    def mean_average_precision_at_k(self, actual, predicted, k):
        """
        Calculate Mean Average Precision at k.
        
        Args:
            actual (list): List of lists with actual relevant items
            predicted (list): List of lists with predicted items
            k (int): Number of items to consider
            
        Returns:
            float: MAP@k score
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
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            actual (list): List of lists with actual relevant items
            predicted (list): List of lists with predicted items
            k (int): Number of items to consider
            
        Returns:
            float: NDCG@k score
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
        Evaluate the naive recommender using various metrics.
        
        Args:
            test_ratio (float): Ratio of data to use for testing
            k (int): Number of items to consider in ranking metrics
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
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
    """
    Evaluate the naive recommender model.
    
    Args:
        user_data_path (str): Path to the user ratings data
        test_ratio (float): Ratio of data to use for testing
        k (int): Number of items to consider in ranking metrics
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
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
    """
    PyTorch dataset for recommendation data.
    
    Attributes:
        users: Array of user indices
        games: Array of game indices
        ratings: Array of ratings
    """
    
    def __init__(self, ratings_df):
        """
        Initialize the dataset from a ratings DataFrame.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user-game-rating data
        """
        self.users = ratings_df['user_idx'].astype(int).values
        self.games = ratings_df['game_idx'].astype(int).values
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of entries in the dataset
        """
        return len(self.ratings)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: Dictionary containing user index, game index, and rating
        """
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'game_idx': torch.tensor(self.games[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


def mean_average_precision_at_k(actual, predicted, k):
    """
    Calculate Mean Average Precision at k.
    
    Args:
        actual (list): List of lists with actual relevant items
        predicted (list): List of lists with predicted items
        k (int): Number of items to consider
        
    Returns:
        float: MAP@k score
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


def ndcg_at_k(actual, predicted, k):
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        actual (list): List of lists with actual relevant items
        predicted (list): List of lists with predicted items
        k (int): Number of items to consider
        
    Returns:
        float: NDCG@k score
    """
    def dcg(rel):
        return sum([(2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rel)])

    ndcgs = []
    for a, p in zip(actual, predicted):
        relevance = [1 if item in a else 0 for item in p[:k]]
        ideal_relevance = sorted(relevance, reverse=True)
        ndcgs.append(dcg(relevance) / (dcg(ideal_relevance) or 1))
    return np.mean(ndcgs)


def evaluate(model, ratings_df, all_game_indices, device='cpu', k=10, num_neg_samples=100):
    """
    Evaluate the deep learning recommender model.
    
    Args:
        model: The trained neural network model
        ratings_df (DataFrame): DataFrame with user-game-rating data
        all_game_indices (list): List of all game indices
        device (str): Device to run the evaluation on ('cpu' or 'cuda')
        k (int): Number of items to consider in ranking metrics
        num_neg_samples (int): Number of negative samples to use
        
    Returns:
        tuple: Tuple containing (mse, rmse, r2, mapk, ndcg) metrics
    """
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
    """
    Main function to run the evaluation of all recommendation systems.
    """
    k = 10
    test_dataset = "data/inference_data/test_dataset.csv"
    evaluate_naive_model(test_dataset, test_ratio=0.2, k=k)

    cleaned_games_df = pd.read_csv('data/inference_data/cleaned_games_df.csv')
    with open('data/inference_data/user_mapping.json') as f:
        user_mapping = json.load(f)
    with open('data/inference_data/game_mapping.json') as f:
        game_mapping = json.load(f)

    # Evaluate traditional model
    model_path = "models/traditional_model.pkl"
    feature_path = "data/inference_data/traditional_feature_matrix.csv"
    title_path = "data/inference_data/game_titles.csv"

    # Simulate a user who likes these games:
    liked_games = ["Minecraft", "Far Cry 3", "Mass Effect 2"]
    evaluate_traditional_model(model_path, feature_path, title_path, liked_games=liked_games, top_n=5)
    
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