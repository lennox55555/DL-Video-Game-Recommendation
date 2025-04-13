import random

class TraditionalRecommender:
    """
    Traditional recommender class that uses content-based filtering to recommend games.
    This model considers multiple game selections and their ratings.
    """
    
    def __init__(self):
        # Game metadata with genres and features
        self.game_metadata = {
            "minecraft": {
                "genres": ["sandbox", "crafting", "survival", "building", "exploration"],
                "features": ["procedural generation", "multiplayer", "creative", "peaceful", "resource gathering"],
                "similar_games": ["terraria", "factorio", "no man's sky", "valheim", "starbound"]
            },
            "fortnite": {
                "genres": ["battle royale", "shooter", "multiplayer", "action", "third-person"],
                "features": ["building", "fast-paced", "competitive", "seasonal", "social"],
                "similar_games": ["apex legends", "pubg", "call of duty: warzone", "fall guys", "realm royale"]
            },
            "zelda": {
                "genres": ["action-adventure", "rpg", "puzzle", "exploration", "fantasy"],
                "features": ["story-driven", "open world", "dungeon crawling", "puzzle solving", "combat"],
                "similar_games": ["genshin impact", "immortals fenyx rising", "okami", "hob", "oceanhorn"]
            },
            "mario": {
                "genres": ["platformer", "family-friendly", "adventure", "casual", "puzzle"],
                "features": ["colorful", "accessible", "varied gameplay", "skill-based", "collectibles"],
                "similar_games": ["crash bandicoot", "rayman legends", "spyro", "a hat in time", "sonic mania"]
            },
            "pokemon": {
                "genres": ["jrpg", "turn-based", "collection", "adventure", "multiplayer"],
                "features": ["monster collecting", "team building", "exploration", "trading", "battling"],
                "similar_games": ["monster hunter stories", "ni no kuni", "temtem", "monster sanctuary", "yokai watch"]
            },
            "gta": {
                "genres": ["open world", "action", "crime", "third-person", "shooter"],
                "features": ["driving", "story-driven", "mission-based", "realistic", "urban"],
                "similar_games": ["saints row", "watch dogs", "mafia", "sleeping dogs", "just cause"]
            },
            "cod": {
                "genres": ["first-person shooter", "action", "military", "multiplayer", "war"],
                "features": ["fast-paced", "competitive", "realistic weapons", "team-based", "progression"],
                "similar_games": ["battlefield", "insurgency", "rainbow six siege", "halo", "titanfall"]
            },
            "fifa": {
                "genres": ["sports", "simulation", "multiplayer", "competitive", "team-based"],
                "features": ["realistic", "licensed teams", "career mode", "online matches", "annual release"],
                "similar_games": ["pro evolution soccer", "nba 2k", "madden nfl", "nhl", "f1"]
            },
            "skyrim": {
                "genres": ["open world", "rpg", "fantasy", "action", "adventure"],
                "features": ["character customization", "vast world", "quests", "modding", "exploration"],
                "similar_games": ["fallout", "the witcher", "dragon age", "kingdom come: deliverance", "outward"]
            },
            "sims": {
                "genres": ["simulation", "life sim", "casual", "building", "customization"],
                "features": ["character creation", "home building", "social gameplay", "sandbox", "goals"],
                "similar_games": ["animal crossing", "stardew valley", "house flipper", "cities: skylines", "two point hospital"]
            },
            "among us": {
                "genres": ["social deduction", "party", "multiplayer", "casual", "strategy"],
                "features": ["deception", "teamwork", "voting", "tasks", "communication"],
                "similar_games": ["town of salem", "unfortunate spacemen", "secret neighbor", "werewolf online", "dread hunger"]
            },
            "cyberpunk": {
                "genres": ["rpg", "open world", "sci-fi", "first-person", "shooter"],
                "features": ["futuristic", "story-driven", "character customization", "decision making", "urban"],
                "similar_games": ["deus ex", "the ascent", "ghostrunner", "observer", "cloudpunk"]
            }
        }
        
        # Library of available games for recommendations
        self.game_library = {
            "terraria": {
                "title": "Terraria",
                "description": "2D sandbox adventure with crafting, building, and exploration",
                "genres": ["sandbox", "crafting", "survival", "building", "exploration"],
                "features": ["procedural generation", "multiplayer", "boss battles", "resource gathering", "2D"]
            },
            "stardew valley": {
                "title": "Stardew Valley",
                "description": "Farming simulation RPG with town building and relationships",
                "genres": ["simulation", "rpg", "farming", "life sim", "pixel art"],
                "features": ["character customization", "relationship building", "seasonal gameplay", "resource management", "relaxing"]
            },
            "apex legends": {
                "title": "Apex Legends",
                "description": "Team-based battle royale with unique character abilities",
                "genres": ["battle royale", "first-person shooter", "multiplayer", "action", "hero shooter"],
                "features": ["character abilities", "team play", "fast-paced", "competitive", "free-to-play"]
            },
            "the witcher 3": {
                "title": "The Witcher 3",
                "description": "Epic fantasy RPG with deep storytelling and rich world",
                "genres": ["rpg", "fantasy", "open world", "action", "adventure"],
                "features": ["story-driven", "choice-based", "monster hunting", "character development", "vast world"]
            },
            "genshin impact": {
                "title": "Genshin Impact",
                "description": "Open-world action RPG with elemental combat and gacha mechanics",
                "genres": ["action", "rpg", "open world", "fantasy", "adventure"],
                "features": ["elemental combat", "character switching", "exploration", "gacha", "anime aesthetic"]
            },
            "rainbow six siege": {
                "title": "Rainbow Six Siege",
                "description": "Tactical team-based shooter with destructible environments",
                "genres": ["first-person shooter", "tactical", "multiplayer", "action", "competitive"],
                "features": ["destructible environments", "team play", "operator selection", "gadgets", "strategy"]
            },
            "cities: skylines": {
                "title": "Cities: Skylines",
                "description": "Modern city-building simulation with deep management systems",
                "genres": ["simulation", "management", "building", "strategy", "sandbox"],
                "features": ["traffic simulation", "zoning", "utilities management", "natural disasters", "modding"]
            },
            # And many more... would add more games here in a real implementation
            "factorio": {
                "title": "Factorio",
                "description": "Complex factory building and automation game",
                "genres": ["simulation", "strategy", "sandbox", "management", "indie"],
                "features": ["automation", "resource management", "tech tree", "enemy waves", "optimization"]
            },
            "hades": {
                "title": "Hades",
                "description": "Action roguelike with Greek mythology and narrative focus",
                "genres": ["roguelike", "action", "hack and slash", "indie", "isometric"],
                "features": ["story progression", "permanent upgrades", "god powers", "character relationships", "fast-paced combat"]
            },
            "hollow knight": {
                "title": "Hollow Knight",
                "description": "Atmospheric metroidvania with challenging combat and exploration",
                "genres": ["metroidvania", "action", "platformer", "indie", "adventure"],
                "features": ["hand-drawn art", "challenging combat", "exploration", "upgrades", "rich lore"]
            },
            "monster hunter world": {
                "title": "Monster Hunter World",
                "description": "Action RPG focused on hunting massive creatures and crafting gear",
                "genres": ["action", "rpg", "multiplayer", "hunting", "adventure"],
                "features": ["large monsters", "weapon mastery", "crafting system", "environmental interaction", "cooperative"]
            },
            "overwatch": {
                "title": "Overwatch",
                "description": "Team-based hero shooter with diverse characters and abilities",
                "genres": ["first-person shooter", "multiplayer", "hero shooter", "team-based", "action"],
                "features": ["hero abilities", "team composition", "objective-based", "competitive", "casual modes"]
            },
            "control": {
                "title": "Control",
                "description": "Supernatural third-person action game with telekinetic powers",
                "genres": ["action", "third-person shooter", "supernatural", "mystery", "sci-fi"],
                "features": ["telekinesis", "destructible environments", "bizarre setting", "narrative focus", "upgradable abilities"]
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
    
    def get_recommendations(self, games, ratings=None):
        """
        Get recommendations based on multiple games and their ratings.
        
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
        
        # Collect genre and feature preferences weighted by ratings
        genre_preferences = {}
        feature_preferences = {}
        
        for game in games:
            game_lower = game.lower()
            
            if game_lower in self.game_metadata:
                # Get the rating or default to 5
                rating_weight = ratings.get(game, 5) / 10.0  # Normalize to 0-1
                
                # Add weighted genre preferences
                for genre in self.game_metadata[game_lower]["genres"]:
                    if genre in genre_preferences:
                        genre_preferences[genre] += rating_weight
                    else:
                        genre_preferences[genre] = rating_weight
                
                # Add weighted feature preferences
                for feature in self.game_metadata[game_lower]["features"]:
                    if feature in feature_preferences:
                        feature_preferences[feature] += rating_weight
                    else:
                        feature_preferences[feature] = rating_weight
        
        # Calculate a score for each potential recommendation
        recommendations = []
        user_games_lower = [g.lower() for g in games]
        
        for game_id, game_info in self.game_library.items():
            # Skip games the user already has
            if game_id in user_games_lower:
                continue
            
            # Calculate genre match score
            genre_score = 0
            for genre in game_info["genres"]:
                if genre in genre_preferences:
                    genre_score += genre_preferences[genre]
            
            # Calculate feature match score
            feature_score = 0
            for feature in game_info["features"]:
                if feature in feature_preferences:
                    feature_score += feature_preferences[feature]
            
            # Combined score (could weight these differently)
            total_score = genre_score + feature_score
            
            if total_score > 0:
                recommendations.append({
                    "game_id": game_id,
                    "title": game_info["title"],
                    "description": game_info["description"],
                    "score": total_score
                })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Convert to the expected format with IDs
        final_recommendations = []
        for i, rec in enumerate(recommendations[:5]):  # Take top 5
            description_parts = []
            
            # Find the most relevant user game for this recommendation
            best_match_game = None
            best_match_score = 0
            
            for game in games:
                game_lower = game.lower()
                if game_lower in self.game_metadata and rec["game_id"] in self.game_metadata[game_lower]["similar_games"]:
                    match_score = ratings.get(game, 5)
                    if match_score > best_match_score:
                        best_match_game = game
                        best_match_score = match_score
            
            # Add a more personalized description
            if best_match_game:
                description = f"Based on your {ratings.get(best_match_game, 5)}/10 rating of {best_match_game}: {rec['description']}"
            else:
                description = f"Matches your genre preferences: {rec['description']}"
            
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