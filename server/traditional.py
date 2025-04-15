import os
import pandas as pd
import joblib

class TraditionalRecommender:
    """
    Traditional recommender class for UI application.
    Loads a combined feature matrix with titles and returns top N recommendations
    based on user-liked games, using a trained Random Forest model.
    """

    def __init__(self, model_path="models/traditional_model.pkl",
                 data_path="data/inference_data/traditional_combined.csv"):
        base_dir = os.path.dirname(__file__)
        self.model = self.load_model(os.path.abspath(os.path.join(base_dir, "../", model_path)))
        self.feature_names = self.model.feature_names_in_

        self.features_df = self.load_feature_data(
            os.path.abspath(os.path.join(base_dir, "../", data_path))
        )

    def load_model(self, model_path):
        """Load a trained scikit-learn model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        return joblib.load(model_path)

    def load_feature_data(self, combined_csv_path):
        """Load a single combined CSV with both features and game titles."""
        df = pd.read_csv(combined_csv_path)
        if 'Title' not in df.columns:
            raise ValueError("Combined CSV must contain a 'Title' column.")
        return df

    def get_recommendations(self, games, ratings=None, top_n=5):
        """
        Recommend games using a trained Random Forest model.

        Args:
            games (list): List of liked game titles
            ratings (dict, optional): Optional ratings per game (unused)
            top_n (int): Number of recommendations to return

        Returns:
            list: Recommendations with title and predicted score
        """
        df = self.features_df.copy()

        # Normalize input titles to avoid filtering issues
        games_normalized = set(g.strip().lower() for g in games)
        df = df[~df['Title'].str.strip().str.lower().isin(games_normalized)]

        # Fallback: if all liked games were removed and nothing is left
        if df.empty:
            print("⚠️ All games filtered out — falling back to full set.")
            df = self.features_df.copy()

        try:
            X = df[self.feature_names]
            df['predicted_score'] = self.model.predict(X)
        except Exception as e:
            print(f"[ERROR during prediction]: {e}")
            fallback_df = self.features_df.copy()
            X = fallback_df[self.feature_names]
            fallback_df['predicted_score'] = self.model.predict(X)
            df = fallback_df

        top_recs = df.sort_values(by='predicted_score', ascending=False).head(top_n)

        return [
            {
                "id": i + 1,
                "title": row["Title"],
                "description": f"Predicted Score: {row['predicted_score']:.2f}"
            }
            for i, row in top_recs.iterrows()
        ]
