import ast
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# select appropriate device based on availability
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Using device: {device}')

def clean_genre_list(raw_genres):
    '''
    Function to clean the genre column of the video games csv

    Inputs:
        - raw_genres: The raw genres list
    
    Returns:
        - the sorted and cleaned genres list
    '''

    if isinstance(raw_genres, str):
        # normalize spacing, casing, and duplicate entries
        cleaned = list({g.strip().lower().replace("’", "'").replace('–', '-') for g in raw_genres.split(',') if g.strip()})
        return sorted(cleaned)
    return []

def extract_platforms_and_metascores(platform_info):
    '''
    Function to extract platform names and metascores from the platform info

    Inputs:
        - platform_info: string representation of a list of platform dictionaries

    Returns:
        - platform_names: list of platform names
        - metascores: list of platform metascores
    '''
    if pd.isna(platform_info):
        return [], []
    try:
        # convert string to list of dicts
        platforms = ast.literal_eval(platform_info)

        # extract platform name
        platform_names = [entry.get('Platform', '') for entry in platforms]

        # extract metascores for the platforms
        metascores = [entry.get('Platform Metascore', None) for entry in platforms]

        return platform_names, metascores
    except:
        return [], []

def platform_score_dict(row):
    '''
    Function to generate a dictionary of platform metascores from a row

    Inputs:
        - row: dataframe row with 'Platform Names' and 'Platform Metascores' lists

    Returns:
        - dict of platform names mapped to integer metascores
    '''
    if isinstance(row['Platform Names'], list) and isinstance(row['Platform Metascores'], list):
        # convert metascores to int and default to 0 if not digit
        return {p: int(m) if str(m).isdigit() else 0 for p, m in zip(row['Platform Names'], row['Platform Metascores'])}
    return {}

def clean_games_df(games_df):
    '''
    Function to clean and preprocess the games dataframe

    Inputs:
        - games_df: raw dataframe containing game metadata

    Returns:
        - games_df: cleaned games dataframe
        - game_features: numpy array of normalized features for each game
    '''
    # fill missing values and apply cleaning
    games_df['Genres'] = games_df['Genres'].fillna('').apply(clean_genre_list)
    games_df['Developer'] = games_df['Developer'].fillna('unknown')
    games_df['Publisher'] = games_df['Publisher'].fillna('unknown')
    games_df['Product Rating'] = games_df['Product Rating'].fillna('unknown')
    games_df['Release Date'] = pd.to_datetime(games_df['Release Date'], errors='coerce')
    games_df['Release Year'] = games_df['Release Date'].dt.year.fillna(0)
    games_df['User Score'] = games_df['User Score'].fillna(games_df['User Score'].median())
    games_df['User Ratings Count'] = games_df['User Ratings Count'].fillna(0)

    # extract platform data into new columns
    games_df[['Platform Names', 'Platform Metascores']] = games_df['Platforms Info'].apply(
        lambda x: pd.Series(extract_platforms_and_metascores(x))
    )

    # build dataframe of platform metascores by game
    platform_score_df = games_df.apply(platform_score_dict, axis=1).apply(pd.Series).fillna(0).astype(int)

    # one-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(games_df['Genres']), columns=mlb.classes_, index=games_df.index)

    # drop the original genre and Platform Info columns
    games_df = pd.concat([games_df.drop(columns=['Genres', 'Platform Names', 'Platform Metascores', 'Platforms Info']), genre_df, platform_score_df], axis=1)

    return games_df

def clean_user_df(user_df, games_df):
    '''
    Function to map user and game ids to indices

    Inputs:
        - user_df: user interaction dataframe with raw ids

    Returns:
        - user_df: dataframe with user_idx and game_idx columns
        - user_mapping: dict mapping user_id to index
        - game_mapping: dict mapping game_title to index
    '''
    # create index mappings for users and games
    user_mapping = get_user_id_mappings(user_df)
    game_mapping = get_game_mapping(games_df)

    # apply mappings to new columns
    user_df['user_idx'] = user_df['user_id'].map(user_mapping)
    user_df['game_idx'] = user_df['game_title'].map(game_mapping)

    user_df = user_df.dropna()

    return user_df, user_mapping, game_mapping

def get_user_id_mappings(user_df):
    '''
    Function to create mapping from user_id to integer index

    Inputs:
        - user_df: dataframe with user_id column

    Returns:
        - dictionary mapping user_id to index
    '''
    return {user: i for i, user in enumerate(user_df['user_id'].unique())}

def get_game_mapping(games_df):
    '''
    Function to create mapping from game_title to integer index

    Inputs:
        - user_df: dataframe with game_title column

    Returns:
        - dictionary mapping game_title to index
    '''
    return {title: i for i, title in enumerate(games_df['Title'].tolist())}

class RecommendationDataset(Dataset):
    '''
    Torch dataset for loading user-game interactions
    '''
    def __init__(self, ratings_df):
        # store data and convert features to tensor
        self.users = ratings_df['user_idx'].astype(int).values
        self.games = ratings_df['game_idx'].astype(int).values
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # return a sample for training
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'game_idx': torch.tensor(self.games[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }

class NCFRecommendationSystem(nn.Module):
    def __init__(self, num_users, num_games, embedding_dim=50):
        super(NCFRecommendationSystem, self).__init__()

        # Separate embeddings for GMF and MLP paths
        self.user_emb_gmf = nn.Embedding(num_users, embedding_dim)
        self.game_emb_gmf = nn.Embedding(num_games, embedding_dim)

        self.user_emb_mlp = nn.Embedding(num_users, embedding_dim)
        self.game_emb_mlp = nn.Embedding(num_games, embedding_dim)

        # MLP layers for NCF path
        self.mlp_fc1 = nn.Linear(embedding_dim * 2, 128)
        self.mlp_fc2 = nn.Linear(128, 64)

        # Final combined layer (GMF + MLP)
        self.final_fc = nn.Linear(64 + embedding_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, user_idx, game_idx):
        # GMF path
        u_gmf = self.user_emb_gmf(user_idx)
        v_gmf = self.game_emb_gmf(game_idx)
        gmf_out = u_gmf * v_gmf

        # MLP path
        u_mlp = self.user_emb_mlp(user_idx)
        v_mlp = self.game_emb_mlp(game_idx)
        mlp_x = torch.cat([u_mlp, v_mlp], dim=1)
        mlp_x = self.relu(self.mlp_fc1(mlp_x))
        mlp_x = self.dropout(mlp_x)
        mlp_out = self.relu(self.mlp_fc2(mlp_x))

        # Combine all
        x = torch.cat([gmf_out, mlp_out], dim=1)
        output = self.final_fc(x)
        return output.squeeze()

def train_model(model, train_loader, optimizer, alpha=0.6, epochs=10):
    '''
    Function to train the recommendation model

    Inputs:
        - model: the recommender model
        - train_loader: DataLoader object for training data
        - optimizer: optimizer used for gradient descent
        - alpha: the hyperparameter to use to balance BPR loss and MSE
        - epochs: number of training epochs
    '''
    # set model to training mode
    model.train()

    bpr = BPRLoss()
    mse = nn.MSELoss()
    
    for epoch in range(epochs):
        train_loss = 0

        for batch in train_loader:
            # move batch to device
            user_idx = batch['user_idx'].to(device)
            game_idx = batch['game_idx'].to(device)
            ratings = batch['rating'].to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            pos_scores = model(user_idx, game_idx)
            neg_game_idx = torch.randint(0, model.game_emb_gmf.num_embeddings, game_idx.shape, device=device)
            neg_scores = model(user_idx, neg_game_idx)

            bpr_loss = bpr(pos_scores, neg_scores)
            mse_loss = mse(pos_scores, ratings)

            total_loss = alpha * mse_loss + (1 - alpha) * bpr_loss

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        if epoch % 1 == 0:
            print(f"Epoch {epoch} - Loss: {train_loss:.4f}")

def evaluate(model, test_loader, criterion):
    '''
    Function to evaluate model performance on a test dataset

    Inputs:
        - model: trained recommender model
        - test_loader: DataLoader object for test data
        - criterion: loss function

    Returns:
        - average test loss
    '''
    # set model to evaluation mode
    model.eval()

    # initialize val loss
    val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            # move batch to device
            user_idx = batch['user_idx'].to(device)
            game_idx = batch['game_idx'].to(device)
            ratings = batch['rating'].to(device)

            # get model outputs
            outputs = model(user_idx, game_idx)
            loss = criterion(outputs, ratings)

            # get loss
            val_loss += loss.item()

    return val_loss / len(test_loader)

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

def main():
    # load raw datasets
    games_df = pd.read_csv('./data/all_video_games.csv')
    user_df = pd.read_csv('./data/metacritic_user_data.csv')

    # clean datasets
    cleaned_games_df = clean_games_df(games_df)
    cleaned_users_df, user_mapping, game_mapping = clean_user_df(user_df, cleaned_games_df)

    # save cleaned data artifacts
    cleaned_games_df.to_csv('./data/inference_data/cleaned_games_df.csv')
    with open('./data/inference_data/user_mapping.json', 'w') as f:
        json.dump(user_mapping, f)
    with open('./data/inference_data/game_mapping.json', 'w') as f:
        json.dump(game_mapping, f)

    train_df, test_dataset = train_test_split(cleaned_users_df, test_size=0.65, random_state=42)

    train_user_ids = set(train_df['user_idx'])
    test_dataset = test_dataset[test_dataset['user_idx'].isin(train_user_ids)]

    test_dataset.to_csv('./data/inference_data/test_dataset.csv')
    train_df.to_csv('./data/inference_data/train_dataset.csv')

    # create training dataset and dataloader
    train_dataset = RecommendationDataset(train_df)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # instantiate the model
    num_users = len(user_mapping)
    num_games = len(game_mapping)

    model = NCFRecommendationSystem(num_users, num_games)
    model.to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('Starting Training')

    # train the model
    epochs = 900
    train_model(model, train_loader, optimizer, epochs=epochs)

    # save the trained model weights
    torch.save(model.state_dict(), './models/deep_learning_model.pth')

if __name__=='__main__':
    main()