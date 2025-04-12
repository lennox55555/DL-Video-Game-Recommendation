from flask import Flask, request, jsonify
import logging
import json
import datetime
import os
import sys
import pandas as pd
import random
import traceback

# Import our model classes
from naive import NaiveGameRecommender
from traditional import TraditionalRecommender
from deep_learning import DeepLearningRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ec2_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")
user_data_path = os.path.join(data_dir, "user_data.csv")
fake_data_path = os.path.join(data_dir, "fake_user_data.csv")
all_games_path = os.path.join(data_dir, "all_video_games.csv")

# Load the video games data
all_games_df = None
try:
    all_games_df = pd.read_csv(all_games_path)
    logger.info(f"Successfully loaded {len(all_games_df)} games from {all_games_path}")
except Exception as e:
    logger.error(f"Error loading games data: {str(e)}")
    # Create a fallback games list if the file can't be loaded
    all_games_df = pd.DataFrame({
        'Title': [
            'Minecraft', 'Fortnite', 'The Legend of Zelda', 'Super Mario', 
            'Pokemon', 'Grand Theft Auto', 'Call of Duty', 'FIFA',
            'The Elder Scrolls: Skyrim', 'The Sims', 'Among Us', 'Cyberpunk 2077',
            'Terraria', 'Stardew Valley', 'Apex Legends', 'Overwatch',
            'Valorant', 'Rocket League', 'Animal Crossing', 'Hades',
            'Hollow Knight', 'Genshin Impact', 'Roblox', 'Fall Guys'
        ]
    })
    logger.info("Using fallback games list")

# Use the data path that exists
if os.path.exists(user_data_path):
    naive_data_path = user_data_path
    logger.info(f"Using real user data from {user_data_path}")
else:
    naive_data_path = fake_data_path
    logger.info(f"Using fake user data from {fake_data_path}")

# Initialize our recommender models
try:
    naive_recommender = NaiveGameRecommender(naive_data_path)
    logger.info("Successfully initialized Naive Game Recommender")
except Exception as e:
    logger.error(f"Error initializing Naive Game Recommender: {str(e)}")
    # Create a simple class with the same interface if initialization fails
    class FallbackRecommender:
        def handle_request(self, request_data):
            return [
                {"id": 1, "title": "Minecraft", "description": "Creative sandbox with building and exploration"},
                {"id": 2, "title": "Rocket League", "description": "Unique sports game combining soccer and vehicles"},
                {"id": 3, "title": "Stardew Valley", "description": "Relaxing farming sim with relationship building"}
            ]
    naive_recommender = FallbackRecommender()
    logger.info("Using fallback recommender due to initialization error")

# Initialize other recommenders
traditional_recommender = TraditionalRecommender()
deep_learning_recommender = DeepLearningRecommender()

app = Flask(__name__)

@app.route('/process-data', methods=['POST'])
def process_data():
    # Get the JSON data from the request
    data = request.json
    logger.info(f"Received data from Lambda: {json.dumps(data)}")
    
    try:
        # Add timestamp for when EC2 received the data
        data['ec2_timestamp'] = datetime.datetime.now().isoformat()
        
        # Extract model type and data from the request
        model_type = data.get('modelType', 'Traditional')
        
        # Add extra validation to ensure model_type is one of the allowed values
        model_type_raw = model_type  # Save the original value for logging
        if model_type not in ['Naive', 'Deep Learning', 'Traditional']:
            logger.warning(f"Invalid model type received: '{model_type}', defaulting to Traditional")
            model_type = 'Traditional'
        
        # Log both the raw and validated model types
        logger.info(f"Model type received: '{model_type_raw}', using: '{model_type}'")
        
        username = data.get('username', 'anonymous')
        # Removed age since we no longer use it
        games = data.get('games', [])
        game_ratings = data.get('gameRatings', {})
        
        # Generate recommendations based on the selected model type
        recommendations = generate_recommendations(model_type, games, game_ratings)
        
        # Add the recommendations to the response data
        data['ec2_recommendations'] = recommendations
        
        # Add processing status
        data['status'] = "processed by ec2"
        
        # Log success
        logger.info(f"Successfully processed data using {model_type} model")
        
        # Return a response
        return jsonify({
            'success': True,
            'message': f"Data processed successfully using {model_type} model",
            'recommendations': recommendations,
            'data': data
        })
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error processing data: {str(e)}",
            'data': data
        }), 500

def generate_recommendations(model_type, games, game_ratings=None):
    """
    Generate recommendations based on the selected model type.
    
    Args:
        model_type (str): The model type (Deep Learning, Traditional, or Naive)
        games (list): List of games the user likes
        game_ratings (dict, optional): Dictionary mapping games to ratings (1-10)
        
    Returns:
        list: List of game recommendation dictionaries with title and description
    """
    logger.info(f"Generating recommendations using {model_type} model for games: {games}")
    
    if not games:
        logger.warning("No games provided, returning default recommendations")
        return [
            {"id": 1, "title": "Minecraft", "description": "Creative sandbox with building and exploration"},
            {"id": 2, "title": "Rocket League", "description": "Unique sports game combining soccer and vehicles"},
            {"id": 3, "title": "Stardew Valley", "description": "Relaxing farming sim with relationship building"}
        ]
    
    # Normalize the model type string - make case-insensitive
    model_type_lower = model_type.lower() if model_type else "traditional"
    logger.info(f"Normalized model type from '{model_type}' to '{model_type_lower}' for processing")
    
    # Generate recommendations based on the model type
    if model_type_lower == "naive":
        if len(games) > 0:
            # For naive model, we only consider the first game
            logger.info(f"Using Naive model with game: {games[0]}")
            # Create a request object that naive_recommender.handle_request expects
            request_data = {
                'games': games,
                'modelType': 'Naive'
            }
            # Use the handle_request method to get formatted recommendations
            return naive_recommender.handle_request(request_data)
        else:
            logger.warning("No games provided for Naive model")
            return []
    
    elif model_type_lower == "deep learning":
        logger.info("Using Deep Learning model")
        return deep_learning_recommender.get_recommendations(games, game_ratings)
    
    else:  # Default to Traditional
        logger.info("Using Traditional model")
        return traditional_recommender.get_recommendations(games, game_ratings)

# The random games endpoint has been removed since we're now using hardcoded games in the frontend
# This simplifies the code and removes a potential source of errors

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'models_available': ['Deep Learning', 'Traditional', 'Naive'],
        'games_available': len(all_games_df) if all_games_df is not None else 0
    })

if __name__ == '__main__':
    logger.info("Starting EC2 processing server...")
    app.run(host='0.0.0.0', port=5000, debug=False)