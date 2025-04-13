import random
import numpy as np

class DeepLearningRecommender:
    """
    Deep Learning recommender that simulates a collaborative filtering neural network model.
    In a real implementation, this would use actual neural networks to predict user preferences.
    """
    
    def __init__(self):
        # Simulate learned embeddings for games
        # In a real system, these would be learned from training data
        self.game_embeddings = {
            "minecraft": np.array([0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.5, 0.8]),
            "fortnite": np.array([0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.9, 0.2]),
            "zelda": np.array([0.7, 0.4, 0.8, 0.3, 0.9, 0.1, 0.2, 0.6]),
            "mario": np.array([0.6, 0.5, 0.7, 0.2, 0.4, 0.3, 0.1, 0.8]),
            "pokemon": np.array([0.5, 0.6, 0.8, 0.4, 0.7, 0.2, 0.3, 0.5]),
            "gta": np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.6, 0.7, 0.1]),
            "cod": np.array([0.2, 0.9, 0.1, 0.7, 0.2, 0.8, 0.9, 0.3]),
            "fifa": np.array([0.3, 0.8, 0.1, 0.6, 0.2, 0.9, 0.8, 0.1]),
            "skyrim": np.array([0.9, 0.3, 0.7, 0.4, 0.8, 0.1, 0.2, 0.6]),
            "sims": np.array([0.7, 0.2, 0.8, 0.3, 0.1, 0.4, 0.2, 0.5]),
            "among us": np.array([0.5, 0.6, 0.2, 0.3, 0.1, 0.8, 0.7, 0.4]),
            "cyberpunk": np.array([0.8, 0.4, 0.6, 0.7, 0.3, 0.5, 0.2, 0.9])
        }
        
        # Potential recommendations with embeddings
        self.recommendation_embeddings = {
            "terraria": {
                "title": "Terraria",
                "description": "2D sandbox adventure with crafting and exploration",
                "embedding": np.array([0.75, 0.25, 0.85, 0.15, 0.65, 0.35, 0.55, 0.75])
            },
            "valorant": {
                "title": "Valorant",
                "description": "Tactical first-person shooter with unique agent abilities",
                "embedding": np.array([0.25, 0.95, 0.15, 0.75, 0.25, 0.85, 0.95, 0.25])
            },
            "breath of the wild": {
                "title": "Breath of the Wild",
                "description": "Open-world adventure with environmental puzzles and exploration",
                "embedding": np.array([0.75, 0.45, 0.85, 0.35, 0.95, 0.15, 0.25, 0.65])
            },
            "super mario odyssey": {
                "title": "Super Mario Odyssey",
                "description": "3D platformer with possession mechanics and collectibles",
                "embedding": np.array([0.65, 0.55, 0.75, 0.25, 0.45, 0.35, 0.15, 0.85])
            },
            "monster hunter rise": {
                "title": "Monster Hunter Rise",
                "description": "Action RPG with monster hunting and gear crafting",
                "embedding": np.array([0.55, 0.65, 0.85, 0.45, 0.75, 0.25, 0.35, 0.55])
            },
            "red dead redemption 2": {
                "title": "Red Dead Redemption 2",
                "description": "Open-world western with deep story and realistic world",
                "embedding": np.array([0.15, 0.75, 0.25, 0.85, 0.35, 0.65, 0.75, 0.15])
            },
            "battlefield 2042": {
                "title": "Battlefield 2042",
                "description": "Large-scale modern military shooter with vehicles",
                "embedding": np.array([0.25, 0.95, 0.15, 0.75, 0.25, 0.85, 0.95, 0.35])
            },
            "nba 2k22": {
                "title": "NBA 2K22",
                "description": "Realistic basketball simulation with multiple game modes",
                "embedding": np.array([0.35, 0.85, 0.15, 0.65, 0.25, 0.95, 0.85, 0.15])
            },
            "elden ring": {
                "title": "Elden Ring",
                "description": "Open-world action RPG with challenging combat",
                "embedding": np.array([0.85, 0.35, 0.75, 0.45, 0.85, 0.15, 0.25, 0.65])
            },
            "animal crossing": {
                "title": "Animal Crossing: New Horizons",
                "description": "Life simulation where you build an island community",
                "embedding": np.array([0.75, 0.25, 0.85, 0.35, 0.15, 0.45, 0.25, 0.55])
            },
            "fall guys": {
                "title": "Fall Guys",
                "description": "Colorful battle royale with mini-games and obstacles",
                "embedding": np.array([0.55, 0.65, 0.25, 0.35, 0.15, 0.85, 0.75, 0.45])
            },
            "deus ex: mankind divided": {
                "title": "Deus Ex: Mankind Divided",
                "description": "Cyberpunk RPG with player choice and multiple approaches",
                "embedding": np.array([0.85, 0.45, 0.65, 0.75, 0.35, 0.55, 0.25, 0.95])
            },
            "doom eternal": {
                "title": "Doom Eternal",
                "description": "Fast-paced FPS with intense combat and movement",
                "embedding": np.array([0.3, 0.95, 0.2, 0.8, 0.15, 0.75, 0.9, 0.25])
            },
            "hades": {
                "title": "Hades",
                "description": "Action roguelike with Greek mythology and narrative focus",
                "embedding": np.array([0.7, 0.5, 0.75, 0.35, 0.8, 0.3, 0.4, 0.7])
            },
            "stardew valley": {
                "title": "Stardew Valley",
                "description": "Farming RPG with relationship building and exploration",
                "embedding": np.array([0.75, 0.3, 0.85, 0.2, 0.4, 0.5, 0.3, 0.6])
            },
            "hollow knight": {
                "title": "Hollow Knight",
                "description": "Challenging metroidvania with atmospheric world",
                "embedding": np.array([0.65, 0.4, 0.8, 0.25, 0.75, 0.35, 0.2, 0.7])
            },
            "final fantasy xiv": {
                "title": "Final Fantasy XIV",
                "description": "Story-rich MMORPG with diverse classes and content",
                "embedding": np.array([0.6, 0.5, 0.75, 0.4, 0.7, 0.3, 0.35, 0.65])
            },
            "it takes two": {
                "title": "It Takes Two",
                "description": "Cooperative adventure with varied gameplay mechanics",
                "embedding": np.array([0.5, 0.6, 0.65, 0.35, 0.45, 0.5, 0.4, 0.7])
            },
            "ghostwire: tokyo": {
                "title": "Ghostwire: Tokyo",
                "description": "Supernatural action-adventure in modern Japan",
                "embedding": np.array([0.6, 0.5, 0.55, 0.65, 0.4, 0.5, 0.35, 0.75])
            },
            "inscryption": {
                "title": "Inscryption",
                "description": "Horror card game with puzzle and roguelike elements",
                "embedding": np.array([0.7, 0.35, 0.6, 0.45, 0.55, 0.4, 0.3, 0.65])
            },
            "splatoon 3": {
                "title": "Splatoon 3",
                "description": "Colorful shooter where you paint to claim territory",
                "embedding": np.array([0.4, 0.75, 0.35, 0.6, 0.3, 0.65, 0.8, 0.4])
            },
            "deathloop": {
                "title": "Deathloop",
                "description": "First-person shooter with time loop mechanics",
                "embedding": np.array([0.45, 0.75, 0.3, 0.7, 0.35, 0.65, 0.55, 0.6])
            },
            "subnautica": {
                "title": "Subnautica",
                "description": "Underwater survival with exploration and base building",
                "embedding": np.array([0.75, 0.3, 0.8, 0.25, 0.65, 0.4, 0.5, 0.7])
            }
        }
        
        # Default recommendations if we can't generate personalized ones
        self.default_recommendations = [
            {"id": 1, "title": "Minecraft", "description": "Creative sandbox with building and exploration"},
            {"id": 2, "title": "Rocket League", "description": "Unique sports game combining soccer and vehicles"},
            {"id": 3, "title": "Among Us", "description": "Social deduction game with teamwork and deception"},
            {"id": 4, "title": "Stardew Valley", "description": "Relaxing farming sim with relationship building"},
            {"id": 5, "title": "Fortnite", "description": "Popular battle royale with building mechanics"}
        ]
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_recommendations(self, games, ratings=None):
        """
        Get recommendations based on multiple games and their ratings using a simulated
        deep learning collaborative filtering model.
        
        Args:
            games (list): List of games the user likes
            ratings (dict, optional): Dictionary mapping games to ratings (1-10)
            
        Returns:
            list: List of game recommendation dictionaries with title and description
        """
        if not games:
            return self.default_recommendations
        
        # Use default equal ratings if not provided
        if not ratings:
            ratings = {game: 5 for game in games}
        
        # Calculate user embedding by weighted average of game embeddings
        user_embedding = np.zeros(8)  # Assuming 8-dimensional embeddings
        total_weight = 0
        
        for game in games:
            game_lower = game.lower()
            if game_lower in self.game_embeddings:
                # Get the rating (1-10) and normalize it to a weight
                weight = ratings.get(game, 5) / 10.0
                user_embedding += weight * self.game_embeddings[game_lower]
                total_weight += weight
        
        # If we couldn't create a user embedding, return defaults
        if total_weight == 0:
            return self.default_recommendations
        
        # Normalize the user embedding
        user_embedding /= total_weight
        
        # Calculate similarity scores for all potential recommendations
        recommendations = []
        for game_id, game_info in self.recommendation_embeddings.items():
            # Skip if the user already has this game
            if game_id in [g.lower() for g in games]:
                continue
            
            similarity = self.cosine_similarity(user_embedding, game_info["embedding"])
            
            # Add some randomness to avoid the same recommendations every time
            # In a real system, you might use exploration strategies like epsilon-greedy
            similarity = similarity * 0.9 + random.random() * 0.1
            
            recommendations.append({
                "game_id": game_id,
                "title": game_info["title"],
                "description": game_info["description"],
                "similarity": similarity
            })
        
        # Sort by similarity (descending)
        recommendations.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Format the final recommendations
        final_recommendations = []
        for i, rec in enumerate(recommendations[:5]):  # Take top 5
            # Create a description that mentions the deep learning model
            description = f"AI model recommendation ({rec['similarity']:.2f} similarity): {rec['description']}"
            
            # If this game is particularly similar to one of the user's games, mention it
            best_match_game = None
            best_match_similarity = 0
            
            for game in games:
                game_lower = game.lower()
                if game_lower in self.game_embeddings:
                    game_similarity = self.cosine_similarity(
                        self.game_embeddings[game_lower],
                        self.recommendation_embeddings[rec["game_id"]]["embedding"]
                    )
                    
                    if game_similarity > best_match_similarity:
                        best_match_game = game
                        best_match_similarity = game_similarity
            
            if best_match_game and best_match_similarity > 0.8:
                description = f"Based on your {ratings.get(best_match_game, 5)}/10 rating of {best_match_game}: {rec['description']}"
            
            final_recommendations.append({
                "id": i + 1,
                "title": rec["title"],
                "description": description
            })
        
        # Add default recommendations if we don't have enough
        if len(final_recommendations) < 3:
            needed = 3 - len(final_recommendations)
            for i in range(needed):
                if i < len(self.default_recommendations):
                    rec = self.default_recommendations[i].copy()
                    rec["id"] = len(final_recommendations) + 1
                    final_recommendations.append(rec)
        
        return final_recommendations