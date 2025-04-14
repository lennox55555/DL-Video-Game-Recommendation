import os
import pandas as pd
import numpy as np
import joblib

class TraditionalRecommender:
    def __init__(self):
        # Load trained model
        self.model_path = os.path.join("models", "traditional_model.pkl")
        self.model = joblib.load(self.model_path)
        self.feature_names = self.model.feature_names_in_

        # Load preprocessed data for inference
        feature_matrix_path = os.path.join("data", "inference_data", "traditional_feature_matrix.csv")
        titles_path = os.path.join("data", "inference_data", "game_titles.csv")

        self.features_df = pd.read_csv(feature_matrix_path)
        self.titles_df = pd.read_csv(titles_path)
        self.features_df['Title'] = self.titles_df['Title']  # attach titles to feature rows

    def get_recommendations(self, liked_games=None, top_n=10):
        df = self.features_df.copy()

        # Filter out games the user already liked
        if liked_games:
            df = df[~df['Title'].isin(liked_games)]
        else:
            print("⚠️ No liked games provided — returning global top recommendations.")

        # Predict using trained model
        X = df[self.feature_names]
        df['predicted_score'] = self.model.predict(X)

        top_recs = df.sort_values(by='predicted_score', ascending=False).head(top_n)

        return [
            {"title": row["Title"], "description": f"Predicted Score: {row['predicted_score']:.2f}"}
            for _, row in top_recs.iterrows()
        ]


# ✅ For testing standalone
if __name__ == "__main__":
    recommender = TraditionalRecommender()

    print("\n🎮 Example: Recommendations excluding liked games ['Halo 3', 'DOOM']\n")
    recs = recommender.get_recommendations(liked_games=['Halo 3', 'DOOM'], top_n=5)
    for rec in recs:
        print(f"- {rec['title']} → {rec['description']}")

    print("\n🎮 Example: Cold-start (no liked games)\n")
    recs = recommender.get_recommendations(liked_games=None, top_n=5)
    for rec in recs:
        print(f"- {rec['title']} → {rec['description']}")
