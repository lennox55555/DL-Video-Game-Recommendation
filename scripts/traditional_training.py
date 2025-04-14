import pandas as pd
import numpy as np
import os, re
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import warnings
import joblib

warnings.filterwarnings("ignore")

def main():
    # Load data
    csv_path = os.path.join("data", "all_video_games.csv")
    df = pd.read_csv(csv_path)

    # Basic cleaning
    df = df[['Title', 'Release Date', 'Genres', 'Product Rating', 'Publisher', 'Developer', 'User Score', 'User Ratings Count']]
    df = df.dropna(subset=['User Score', 'Release Date', 'Genres', 'Product Rating', 'User Ratings Count'])
    df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')
    df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
    df.dropna(subset=['User Score', 'Release Year'], inplace=True)

    # Log transform count-based column
    df['Log Ratings Count'] = np.log1p(df['User Ratings Count'])

    # Compute game age from release year
    df['Game Age'] = 2025 - df['Release Year']

    # Genre multi-hot encoding
    df['Genres'] = df['Genres'].apply(lambda x: re.split(',\s*', x.lower()) if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    genres_df = pd.DataFrame(mlb.fit_transform(df['Genres']), columns=mlb.classes_, index=df.index)

    # Product Rating one-hot encoding
    ohe = OneHotEncoder(sparse_output=False)
    product_df = pd.DataFrame(
        ohe.fit_transform(df[['Product Rating']]),
        columns=ohe.get_feature_names_out(['Product Rating']),
        index=df.index
    )

    # Developer & Publisher encoding (top N only)
    for col in ['Publisher', 'Developer']:
        top_n = df[col].value_counts().nlargest(20).index
        df[col] = df[col].where(df[col].isin(top_n), other='Other')
    pub_dev_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    pub_dev_encoded = pub_dev_ohe.fit_transform(df[['Publisher', 'Developer']])
    pub_dev_df = pd.DataFrame(pub_dev_encoded, columns=pub_dev_ohe.get_feature_names_out(['Publisher', 'Developer']), index=df.index)

    # Combine features
    features = pd.concat([
        df[['Release Year', 'Log Ratings Count', 'Game Age']],
        genres_df,
        product_df,
        pub_dev_df
    ], axis=1)

    X = features
    y = df['User Score'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest with tuning
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 0.5]
    }

    print("Tuning Random Forest with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate
    preds = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"✅ Tuned Random Forest RMSE: {rmse:.4f}")

    # ✅ Save the model inside the function
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/traditional_model.pkl")


if __name__ == "__main__":
    main()
