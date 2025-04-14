import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class NaiveGameRecommender:
    def __init__(self, user_data_path):
        """
        Load ratings csv.
        
        Args:
            user_data_path (str): Path to the CSV file containing user rating data
        """
        try:
            self.user_data = pd.read_csv(user_data_path)
            logger.info(f"Loaded user data from {user_data_path}")
            self.avg_ratings = None
            self.compute_average_ratings()
            
            # Define game descriptions for better recommendations
            self.game_descriptions = {
                "minecraft": "A sandbox game focused on building and exploration",
                "fortnite": "A battle royale game with building mechanics",
                "zelda": "An action-adventure game with puzzles and exploration",
                "mario": "A platformer game with colorful worlds",
                "pokemon": "An RPG focused on collecting and battling creatures",
                "gta": "An open-world action game in urban environments",
                "cod": "A first-person shooter with multiplayer focus",
                "fifa": "A sports simulation game focusing on soccer",
                "skyrim": "An open-world fantasy RPG with character customization",
                "sims": "A life simulation game with building and social elements",
                "among us": "A social deduction game with teamwork and deception",
                "cyberpunk": "A futuristic RPG with first-person gameplay",
                "terraria": "A 2D sandbox adventure with crafting and exploration",
                "stardew valley": "A farming simulation RPG with relationship building",
                "apex legends": "A team-based battle royale with hero abilities",
                "overwatch": "A team-based hero shooter with objective gameplay",
                "valorant": "A tactical first-person shooter with unique character abilities",
                "rocket league": "A vehicular soccer game with physics-based gameplay",
                "animal crossing": "A life simulation game with community building",
                "hades": "A roguelike action game with Greek mythology themes",
                "hollow knight": "A challenging metroidvania with atmospheric world",
                "genshin impact": "An open-world action RPG with gacha mechanics"
            }
            
        except Exception as e:
            logger.error(f"Error loading user data from {user_data_path}: {str(e)}")
            raise RuntimeError(f"Failed to load data from {user_data_path}: {str(e)}")

    def compute_average_ratings(self):
        """
        Compute each game's average rating.
        """
        self.avg_ratings = self.user_data.groupby("game_title")["rating"].mean().reset_index()
        logger.info(f"Computed average ratings for {len(self.avg_ratings)} games")

    def get_recommendations(self, target_game, top_n=3):
        """
        Recommend top_n games by average rating, with additional variety.
        
        Args:
            target_game (str): The game to base recommendations on
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of game titles recommended
        """
        import random
        import hashlib
        
        seed = int(hashlib.md5(target_game.encode()).hexdigest(), 16) % 10000
        random.seed(seed)
        
        candidates = self.avg_ratings[self.avg_ratings["game_title"] != target_game.lower()]
        
        candidates = candidates.sample(frac=1, random_state=seed).sort_values(by="rating", ascending=False)
        
        top_rated = candidates.head(min(10, len(candidates))).copy()
        mid_rated = candidates.iloc[min(10, len(candidates)):min(30, len(candidates))].copy() if len(candidates) > 10 else pd.DataFrame()
        
        selection = []
        
        if not top_rated.empty:
            num_top = min(int(top_n * 0.7) + 1, len(top_rated))
            selection.extend(top_rated.sample(n=num_top, random_state=seed)["game_title"].tolist())
        
        if not mid_rated.empty and len(selection) < top_n:
            num_mid = min(top_n - len(selection), len(mid_rated))
            selection.extend(mid_rated.sample(n=num_mid, random_state=seed)["game_title"].tolist())
        
        if len(selection) < top_n and len(candidates) > len(selection):
            remaining = candidates[~candidates["game_title"].isin(selection)]
            num_remaining = min(top_n - len(selection), len(remaining))
            selection.extend(remaining.head(num_remaining)["game_title"].tolist())
        
        random.shuffle(selection)
        
        logger.info(f"Generated {len(selection)} recommendations for {target_game}")
        return selection[:top_n]

    def get_formatted_recommendations(self, target_game, top_n=5):
        """
        Get recommendations formatted for the frontend/Lambda response.
        
        Args:
            target_game (str): The game to base recommendations on
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommendation dictionaries with id, title, and description
        """
        import random
        
        genres = [
            "action", "adventure", "RPG", "shooter", "strategy", "simulation", 
            "sports", "racing", "platformer", "puzzle", "horror", "survival",
            "open-world", "roguelike", "metroidvania", "card game", "fighting"
        ]
        
        reasons = [
            "Players who enjoyed {input_game} also rated this highly",
            "This shares similar gameplay elements with {input_game}",
            "A top pick for fans of {input_game}",
            "Highly recommended if you like {input_game}",
            "Has similar appeal to {input_game}",
            "Popular among {input_game} players",
            "Complements your interest in {input_game}",
            "A different take on what makes {input_game} enjoyable"
        ]
        
        recommendations = self.get_recommendations(target_game, top_n)
        
        formatted_recs = []
        for i, game in enumerate(recommendations):
            reason = random.choice(reasons).format(input_game=target_game)
            
            # get a description if available or generate one
            if game.lower() in self.game_descriptions:
                description = self.game_descriptions[game.lower()]
            else:
                # generate a plausible description based on the game title
                random.seed(hash(game))  
                genre1 = random.choice(genres)
                genre2 = random.choice([g for g in genres if g != genre1])
                description = f"A {genre1} {genre2} experience with unique gameplay elements"
            
            # Format the recommendation
            formatted_recs.append({
                "id": i + 1,
                "title": game.title(),  
                "description": f"{reason}: {description}"
            })
        
        if len(formatted_recs) < top_n:
            formatted_recs.append({
                "id": len(formatted_recs) + 1,
                "title": "More Recommendations Coming Soon",
                "description": f"Our recommendation engine is constantly learning from user preferences"
            })
        
        logger.info(f"Generated {len(formatted_recs)} formatted recommendations for {target_game}")
        return formatted_recs

    def handle_request(self, request_data):
        """
        Handle a request from the frontend/Lambda.
        
        Args:
            request_data (dict): The request data
            
        Returns:
            list: List of recommendation dictionaries
        """
        try:
            # log the incoming request data
            logger.info(f"Handling naive recommendation request: {request_data}")
            
            # extract the target game(s) from the request
            games = request_data.get('games', [])
            
            # for naive model, we only use the first game
            if not games:
                logger.warning("No games provided in request, using fallback recommendations")
                # provide generic recommendations
                return [
                    {"id": 1, "title": "Minecraft", "description": "A creative sandbox experience with endless possibilities"},
                    {"id": 2, "title": "The Legend of Zelda", "description": "An epic adventure series with puzzle-solving and exploration"},
                    {"id": 3, "title": "Fortnite", "description": "Popular battle royale game with building mechanics"},
                    {"id": 4, "title": "Among Us", "description": "Social deduction game that tests your ability to spot deception"},
                    {"id": 5, "title": "Stardew Valley", "description": "Relaxing farming simulation with relationship building"}
                ]
            
            # use the first game as the target game
            target_game = games[0]
            logger.info(f"Generating naive recommendations for game: {target_game}")
            
            # getrecommendations
            recommendations = self.get_formatted_recommendations(target_game, top_n=5)
            
            #  5 recommendations
            if len(recommendations) < 5:
                logger.info(f"Only generated {len(recommendations)} recommendations, adding generic ones")
                generic_recs = [
                    {"id": 6, "title": "Minecraft", "description": "A creative sandbox experience with endless possibilities"},
                    {"id": 7, "title": "The Legend of Zelda", "description": "An epic adventure series with puzzle-solving and exploration"},
                    {"id": 8, "title": "Fortnite", "description": "Popular battle royale game with building mechanics"},
                    {"id": 9, "title": "Rocket League", "description": "Exciting vehicular soccer with physics-based gameplay"},
                    {"id": 10, "title": "Stardew Valley", "description": "Relaxing farming simulation with relationship building"}
                ]
                # Add enough to reach 5 total
                for i in range(5 - len(recommendations)):
                    rec = generic_recs[i].copy()
                    rec["id"] = len(recommendations) + 1
                    recommendations.append(rec)
            
            logger.info(f"Returning {len(recommendations)} recommendations for {target_game}")
            return recommendations[:5]  # Always return exactly 5 recommendations
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [{
                "id": 1, 
                "title": "Recommendation Error",
                "description": "We encountered an issue generating recommendations. Please try a different game."
            }]

    def evaluate_rmse(self, test_ratio=0.2, random_state=42):
        """
        Evaluate RMSE on held-out ratings.
        
        Args:
            test_ratio (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            float: Root Mean Squared Error on test set
        """
        # shuffle and split data into test and train sets
        data_shuffled = self.user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_size = int(len(data_shuffled) * test_ratio)
        test_set = data_shuffled.iloc[:test_size].copy()
        train_set = data_shuffled.iloc[test_size:].copy()

        # compute average rating per game from training data
        train_avg = train_set.groupby("game_title")["rating"].mean().to_dict()
        global_avg = train_set["rating"].mean()

        # predict test ratings using the game average (or global average)
        test_set["predicted"] = test_set["game_title"].apply(lambda g: train_avg.get(g, global_avg))

        # get RMSE
        rmse = np.sqrt(np.mean((test_set["rating"] - test_set["predicted"])**2))
        return rmse

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/fake_user_data.csv")
    
    recommender = NaiveGameRecommender(data_path)
    
    target_game = "minecraft"
    recommendations = recommender.get_recommendations(target_game=target_game, top_n=3)
    print("Raw recommendations:")
    print(recommendations)
    
    formatted_recs = recommender.get_formatted_recommendations(target_game=target_game, top_n=5)
    print("\nFormatted recommendations:")
    for rec in formatted_recs:
        print(f"{rec['id']}. {rec['title']}: {rec['description']}")
    
    test_request = {
        'username': 'testuser',
        'age': 25,
        'modelType': 'Naive',
        'games': ['minecraft']
    }
    request_recs = recommender.handle_request(test_request)
    print("\nResponse to request:")
    for rec in request_recs:
        print(f"{rec['id']}. {rec['title']}: {rec['description']}")
    
    rmse_value = recommender.evaluate_rmse(test_ratio=0.2)
    print("\nEvaluation metrics:")
    print(f"RMSE on held-out ratings: {rmse_value:.4f}")