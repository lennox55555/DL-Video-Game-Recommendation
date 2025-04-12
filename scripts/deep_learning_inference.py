import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from deep_learning_training import RecommendationDataset, HybridRecommender

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Using device: {device}')

def freeze_all_but_one(embedding, idx):
    '''
    function to freeze all rows of the embedding layers except one

    Inputs:
        - embedding: embedding layer
        - idx: index of row to keep trainable
    '''
    # create a mask that blocks gradients for all embeddings except one
    mask = torch.ones_like(embedding.weight, requires_grad=False)
    mask[idx] = 0
    embedding.weight.register_hook(lambda grad: grad * (mask == 0).float())

class ContentRecommender:
    '''
    Hybrid content-based and collaborative recommender system
    '''
    def __init__(self, model, user_mapping, games_df, game_features, game_mapping):
        self.model = model
        self.games_df = games_df
        self.game_features = game_features
        self.game_mapping = game_mapping
        self.user_mapping = user_mapping
        self.idx_to_game = {i: title for i, title in enumerate(games_df['Title'].values)}

    def recommend(self, user_id=None, new_user_ratings=None, top_n=5):
        '''
        Function to generate recommendations for a user

        Inputs:
            - user_id: known user id in training set (optional)
            - new_user_ratings: dict of game_title to rating for cold start user (optional)
            - top_n: number of recommendations to return

        Returns:
            - list of recommended game titles
        '''
        # route the recommendation to the appropriate function
        if user_id is not None and user_id in self.user_mapping:
            return self.recommend_for_known_user(user_id, top_n)
        if new_user_ratings is not None:
            return self.recommend_new_user(new_user_ratings, top_n)
        return "Must provide either user_id or new_user_ratings."

    def recommend_new_user(self, new_user_ratings, top_n=5):
        '''
        Generate recommendations for a new user using content-based filtering

        Inputs:
            - new_user_ratings: dictionary mapping game title to rating
            - top_n: number of games to recommend

        Returns:
            - list of top_n recommended game titles
        '''
        # filter ratings to include only games present in the game mapping
        valid_ratings = {
            game: rating for game, rating in new_user_ratings.items()
            if game in self.game_mapping
        }

        if not new_user_ratings:
            # fallback to the most popular games if there's no input
            popular_indices = self.games_df.sort_values('User Ratings Count', ascending=False).index[:top_n].tolist()
            return [self.idx_to_game[i] for i in popular_indices]

        if not valid_ratings:
            return "None of the provided games exist in our database"

        # build a weighted user profile from rated games
        rated_game_indices = [self.game_mapping[game] for game in valid_ratings.keys()]
        rated_game_features = self.game_features[rated_game_indices]
        ratings_array = np.array(list(valid_ratings.values()))
        weights = np.ones_like(ratings_array) if ratings_array.max() == ratings_array.min() else (ratings_array - ratings_array.min()) / (ratings_array.max() - ratings_array.min())
        user_profile = np.average(rated_game_features, axis=0, weights=weights)
        user_profile_tensor = torch.FloatTensor(user_profile).unsqueeze(0).to(device)

        # get user embedding from game feature profile
        with torch.no_grad():
            user_emb = self.model.feature_linear(user_profile_tensor)

        # score all games for this new user
        all_game_indices = list(range(len(self.game_features)))
        game_idx_tensor = torch.LongTensor(all_game_indices).to(device)
        game_feat_tensor = torch.FloatTensor(self.game_features).to(device)
        user_tensor = user_emb.repeat(len(all_game_indices), 1)

        with torch.no_grad():
            game_emb = self.model.game_embedding(game_idx_tensor)
            feature_emb = self.model.feature_linear(game_feat_tensor)
            x = torch.cat([user_tensor, game_emb, feature_emb], dim=1)
            x = self.model.relu(self.model.fc1(x))
            x = self.model.dropout(x)
            x = self.model.relu(self.model.fc2(x))
            x = self.model.dropout(x)
            scores = self.model.fc3(x).squeeze()

        # prevent recommending already-rated games
        for idx in rated_game_indices:
            scores[idx] = -float('inf')

        top_indices = torch.topk(scores, top_n).indices.cpu().numpy()
        return [self.idx_to_game[i] for i in top_indices]

    def recommend_for_known_user(self, user_id, top_n=5):
        '''
        Generate recommendations for a known user via collaborative filtering

        Inputs:
            - user_id: string id of known user
            - top_n: number of games to recommend

        Returns:
            - A list of recommended game titles
        '''
        # get the index of the user
        user_idx = self.user_mapping[user_id]

        # convert it to a tensor and move to the device
        user_idx_tensor = torch.LongTensor([user_idx]).to(device)

        # get all game indices and its corresponding feature subset
        all_game_indices = list(self.game_mapping.values())
        all_game_idx_tensor = torch.LongTensor(all_game_indices).to(device)
        subset_game_features = torch.FloatTensor(self.game_features[all_game_indices]).to(device)

        # evaluate the model
        self.model.eval()
        with torch.no_grad():
            user_tensor = user_idx_tensor.repeat(len(all_game_indices))
            
            # compute predicted scores for all games
            scores = self.model(user_tensor, all_game_idx_tensor, subset_game_features)

        # get the top scoring game indices
        top_indices = torch.topk(scores, top_n).indices.cpu().numpy()

        # convert game indices back to titles and return it
        idx_to_title = {v: k for k, v in self.game_mapping.items()}
        return [idx_to_title[i] for i in [all_game_indices[i] for i in top_indices]]

    def update_user_embedding(self, user_id, new_ratings_df, epochs=10, lr=0.01):
        '''
        Function to perform SGD updates on a user embedding

        Inputs:
            - user_id: the user to update
            - new_ratings_df: dataframe with 'game_title' and 'rating'
            - epochs: number of epochs to fine-tune
            - lr: learning rate for SGD
        '''
        # get the index of the user to update
        user_idx = self.user_mapping[user_id]

        # make a copy of the ratings to avoid modifying original
        new_ratings_df = new_ratings_df.copy()

        # map user and game ids to their corresponding indices
        new_ratings_df['user_idx'] = user_idx
        new_ratings_df['game_idx'] = new_ratings_df['game_title'].map(self.game_mapping)

        # drop rows with missing game mappings
        new_ratings_df = new_ratings_df.dropna(subset=['game_idx'])

        # ensure game indices are integers
        new_ratings_df['game_idx'] = new_ratings_df['game_idx'].astype(int)

        # create dataset and dataloader from new ratings
        dataset = RecommendationDataset(new_ratings_df, self.game_features)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # enable training for user embeddings only
        self.model.user_embedding.weight.requires_grad = True

        # apply selective gradient mask so only this user's embedding updates
        freeze_all_but_one(self.model.user_embedding, user_idx)

        # use SGD optimizer on user embeddings
        optimizer = torch.optim.SGD([self.model.user_embedding.weight], lr=lr)

        # set model to train mode
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                # move batch to device
                user_idx_batch = batch['user_idx'].to(device)
                game_idx = batch['game_idx'].to(device)
                game_features = batch['game_features'].to(device)
                ratings = batch['rating'].to(device)

                # forward + backward + step
                optimizer.zero_grad()
                preds = self.model(user_idx_batch, game_idx, game_features)
                loss = torch.nn.MSELoss()(preds.view(-1), ratings.view(-1))
                loss.backward()
                optimizer.step()

def main():
    # load user and model data
    user_df = pd.read_csv('./data/fake_user_data.csv')
    held_out_user = 'hollandmichael'
    held_out_user_df = user_df[user_df['user_id'] == held_out_user].copy()

    # load artifacts
    cleaned_games_df = pd.read_csv('data/inference_data/cleaned_games_df.csv')
    game_features = np.load('data/inference_data/game_features.npy')
    with open('data/inference_data/user_mapping.json') as f:
        user_mapping = json.load(f)
    with open('data/inference_data/game_mapping.json') as f:
        game_mapping = json.load(f)

    # restore model
    num_users = len(user_mapping)
    num_games = len(game_mapping)
    num_features = game_features.shape[1]
    model = HybridRecommender(num_users, num_games, num_features)
    model.load_state_dict(torch.load('./models/deep_learning_model.pth'))
    model.to(device)

    # initialize recommender
    recommender = ContentRecommender(model, user_mapping, cleaned_games_df, game_features, game_mapping)

    # update user embedding with few new ratings
    new_ratings = held_out_user_df.sample(3).copy()
    new_ratings['user_id'] = 'davisjulie'
    recommender.update_user_embedding('davisjulie', new_ratings)

    # print updated user recs
    recs = recommender.recommend(user_id='davisjulie')
    print("Updated user recs:", recs)

    # simulate cold start
    cold_user_ratings = dict(held_out_user_df[['game_title', 'rating']].values)
    cold_recs = recommender.recommend(new_user_ratings=cold_user_ratings)
    print("New user recs:", cold_recs)

if __name__ == '__main__':
    main()
