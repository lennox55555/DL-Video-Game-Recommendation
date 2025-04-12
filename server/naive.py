import pandas as pd
import numpy as np
import os
import logging

# Set up logging
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
        Recommend top_n games by average rating.
        
        Args:
            target_game (str): The game to base recommendations on
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of game titles recommended
        """
        # Normalize the target game name to lowercase
        target_game_lower = target_game.lower()
        
        # Remove target game from recommendations
        recs = self.avg_ratings[self.avg_ratings["game_title"] != target_game_lower]
        # Sort by highest rating
        recs = recs.sort_values(by="rating", ascending=False)
        # Return top N recommendations
        return recs["game_title"].head(top_n).tolist()

    def get_formatted_recommendations(self, target_game, top_n=5):
        """
        Get recommendations formatted for the frontend/Lambda response.
        
        Args:
            target_game (str): The game to base recommendations on
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommendation dictionaries with id, title, and description
        """
        # Get raw recommendations
        recommendations = self.get_recommendations(target_game, top_n)
        
        # Format recommendations for frontend
        formatted_recs = []
        for i, game in enumerate(recommendations):
            # Default description if game not in our descriptions dictionary
            description = self.game_descriptions.get(
                game, 
                "A highly rated game you might enjoy"
            )
            
            # Format the recommendation
            formatted_recs.append({
                "id": i + 1,
                "title": game.title(),  # Capitalize the game title
                "description": f"Based on your interest in {target_game}: {description}"
            })
        
        # Add a message about the naive model
        if len(formatted_recs) < top_n:
            formatted_recs.append({
                "id": len(formatted_recs) + 1,
                "title": "Naive Model",
                "description": f"Recommendations powered by naive average rating model"
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
            # Extract the target game(s) from the request
            games = request_data.get('games', [])
            
            # For naive model, we only use the first game
            if not games:
                logger.warning("No games provided in request")
                return []
            
            # Use the first game as the target game
            target_game = games[0]
            
            # Get formatted recommendations
            recommendations = self.get_formatted_recommendations(target_game, top_n=5)
            logger.info(f"Generated recommendations for {target_game}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return [{
                "id": 1,
                "title": "Error",
                "description": f"An error occurred while generating recommendations: {str(e)}"
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
        # Shuffle and split data into test and train sets
        data_shuffled = self.user_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_size = int(len(data_shuffled) * test_ratio)
        test_set = data_shuffled.iloc[:test_size].copy()
        train_set = data_shuffled.iloc[test_size:].copy()

        # Compute average rating per game from training data
        train_avg = train_set.groupby("game_title")["rating"].mean().to_dict()
        global_avg = train_set["rating"].mean()

        # Predict test ratings using the game average (or global average)
        test_set["predicted"] = test_set["game_title"].apply(lambda g: train_avg.get(g, global_avg))

        # Calculate RMSE
        rmse = np.sqrt(np.mean((test_set["rating"] - test_set["predicted"])**2))
        return rmse

# Example usage
if __name__ == '__main__':
    # Use the relative path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/fake_user_data.csv")
    
    # Initialize the recommender
    recommender = NaiveGameRecommender(data_path)
    
    # Test simple recommendations
    target_game = "minecraft"
    recommendations = recommender.get_recommendations(target_game=target_game, top_n=3)
    print("Raw recommendations:")
    print(recommendations)
    
    # Test formatted recommendations for frontend
    formatted_recs = recommender.get_formatted_recommendations(target_game=target_game, top_n=5)
    print("\nFormatted recommendations:")
    for rec in formatted_recs:
        print(f"{rec['id']}. {rec['title']}: {rec['description']}")
    
    # Test request handling
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
    
    # Test evaluation
    rmse_value = recommender.evaluate_rmse(test_ratio=0.2)
    print("\nEvaluation metrics:")
    print(f"RMSE on held-out ratings: {rmse_value:.4f}")