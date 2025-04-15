import os
import pandas as pd
import numpy as np
import joblib

class TraditionalRecommender:
    """
    Traditional recommender system using a trained Random Forest model on tabular game metadata.
    """

    def __init__(self, model_path="models/traditional_model.pkl", feature_path="data/inference_data/traditional_feature_matrix.csv", title_path="data/inference_data/game_titles.csv"):
        self.model = self.load_model(model_path)
        self.feature_names = self.model.feature_names_in_
        self.features_df = self.load_feature_data(feature_path, title_path)

    def load_model(self, model_path):
        """
        Load a trained scikit-learn model from a file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        return joblib.load(model_path)

    def load_feature_data(self, feature_matrix_path, titles_path):
        """
        Load the preprocessed feature matrix and game titles for inference.
        """
        features_df = pd.read_csv(feature_matrix_path)
        titles_df = pd.read_csv(titles_path)

        if 'Title' not in titles_df.columns:
            raise ValueError("Game titles CSV must contain a 'Title' column.")

        features_df['Title'] = titles_df['Title']
        return features_df

    def recommend_games(self, liked_games=None, top_n=10):
        """
        Generate top N game recommendations based on the user's liked games.

        Parameters:
            liked_games (list[str]): List of games the user already liked
            top_n (int): Number of recommendations to return

        Returns:
            list[dict]: List of recommended games with predicted scores
        """
        df = self.features_df.copy()

        # Filter out liked games
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