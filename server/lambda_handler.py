import json
import logging
import random
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Game recommendations by age group and game preference
GAME_RECOMMENDATIONS = {
    # Age group: < 13
    "kids": {
        "minecraft": [
            {"title": "Terraria", "description": "A 2D sandbox adventure with crafting and exploration"},
            {"title": "Roblox", "description": "User-generated gaming platform with endless possibilities"},
            {"title": "Portal Knights", "description": "3D sandbox RPG with crafting, combat, and world-building"}
        ],
        "fortnite": [
            {"title": "Rocket League", "description": "Soccer with rocket-powered cars and fun competitions"},
            {"title": "Fall Guys", "description": "Colorful battle royale with obstacle courses and mini-games"},
            {"title": "Apex Legends", "description": "Team-based battle royale with diverse characters"}
        ],
        "mario": [
            {"title": "Rayman Legends", "description": "Vibrant platformer with co-op gameplay"},
            {"title": "Crash Bandicoot N. Sane Trilogy", "description": "Classic platforming adventures remastered"},
            {"title": "Kirby Star Allies", "description": "Colorful platformer with friendly team mechanics"}
        ],
        "pokemon": [
            {"title": "Monster Hunter Stories", "description": "Turn-based RPG with monster collection"},
            {"title": "Ni No Kuni", "description": "Beautiful RPG with creature collection and battling"},
            {"title": "Yokai Watch", "description": "RPG about collecting and battling supernatural creatures"}
        ]
    },
    
    # Age group: 13-17
    "teens": {
        "minecraft": [
            {"title": "Satisfactory", "description": "First-person factory building and optimization game"},
            {"title": "No Man's Sky", "description": "Space exploration and survival with procedural planets"},
            {"title": "Valheim", "description": "Viking-themed survival and exploration game"}
        ],
        "fortnite": [
            {"title": "Valorant", "description": "Tactical character-based shooter with unique abilities"},
            {"title": "Overwatch", "description": "Team-based shooter with diverse heroes and objectives"},
            {"title": "Call of Duty: Warzone", "description": "Fast-paced battle royale in the CoD universe"}
        ],
        "zelda": [
            {"title": "Genshin Impact", "description": "Open-world action RPG with elemental combat"},
            {"title": "Immortals Fenyx Rising", "description": "Mythological adventure with puzzles and combat"},
            {"title": "Okami HD", "description": "Beautiful action-adventure with Japanese mythology"}
        ],
        "gta": [
            {"title": "Watch Dogs", "description": "Open-world hacking and action game in a modern city"},
            {"title": "Saints Row", "description": "Over-the-top open-world action with humor"},
            {"title": "Just Cause 4", "description": "Explosive open-world action with physics-based tools"}
        ]
    },
    
    # Age group: 18+
    "adults": {
        "skyrim": [
            {"title": "The Witcher 3", "description": "Expansive fantasy RPG with rich storytelling and choices"},
            {"title": "Dragon Age: Inquisition", "description": "Team-based fantasy RPG with deep lore"},
            {"title": "Kingdom Come: Deliverance", "description": "Historical RPG with realistic medieval setting"}
        ],
        "cyberpunk": [
            {"title": "Deus Ex: Mankind Divided", "description": "Immersive cyberpunk RPG with multiple approaches"},
            {"title": "Detroit: Become Human", "description": "Narrative-driven game about AI consciousness"},
            {"title": "The Ascent", "description": "Cyberpunk action-RPG with twin-stick shooting"}
        ],
        "cod": [
            {"title": "Battlefield 2042", "description": "Large-scale modern military shooter with vehicles"},
            {"title": "Hell Let Loose", "description": "Realistic WWII team-based tactical shooter"},
            {"title": "Insurgency: Sandstorm", "description": "Tactical modern military shooter with intensity"}
        ],
        "fifa": [
            {"title": "NBA 2K", "description": "Comprehensive basketball simulation with multiple modes"},
            {"title": "F1 2023", "description": "Authentic Formula 1 racing with team management"},
            {"title": "MLB The Show", "description": "Detailed baseball simulation with career and team modes"}
        ]
    },
    
    # Default recommendations for any age group
    "default": [
        {"title": "Minecraft", "description": "Creative sandbox with building and exploration"},
        {"title": "Rocket League", "description": "Unique sports game combining soccer and vehicles"},
        {"title": "Among Us", "description": "Social deduction game with teamwork and deception"},
        {"title": "Stardew Valley", "description": "Relaxing farming sim with relationship building"}
    ]
}

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Check if this is a WebSocket event
    route_key = event.get('requestContext', {}).get('routeKey')
    
    if route_key == '$connect':
        return handle_connect(event)
    elif route_key == '$disconnect':
        return handle_disconnect(event)
    elif route_key == 'sendUserData':
        return handle_user_data(event)
    else:
        # Handle direct Lambda invocation (e.g., for testing)
        username = event.get('username', 'anonymous')
        age = event.get('age', 25)
        games = event.get('games', [])
        recommendations = generate_recommendations(username, age, games)
        
        return {
            'statusCode': 200,
            'recommendations': recommendations
        }

def handle_connect(event):
    # Log new WebSocket connection
    connection_id = event['requestContext']['connectionId']
    logger.info(f"New connection: {connection_id}")
    return {'statusCode': 200}

def handle_disconnect(event):
    # Log WebSocket disconnection
    connection_id = event['requestContext']['connectionId']
    logger.info(f"Connection closed: {connection_id}")
    return {'statusCode': 200}

def handle_user_data(event):
    # Extract connection ID and user data
    connection_id = event['requestContext']['connectionId']
    endpoint_url = f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
    
    # Parse the body as JSON
    body = json.loads(event.get('body', '{}'))
    logger.info(f"Received user data: {body}")
    
    # Extract user information
    username = body.get('username', 'anonymous')
    age = body.get('age', 25)
    games = body.get('games', [])
    
    # Generate recommendations
    recommendations = generate_recommendations(username, age, games)
    
    # Send response back through WebSocket
    response_data = {
        'statusCode': 200,
        'recommendations': recommendations
    }
    
    try:
        # Send the response back through the WebSocket
        api_gateway = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        api_gateway.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(response_data)
        )
        logger.info(f"Successfully sent recommendations to {connection_id}")
    except Exception as e:
        logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
    
    return {'statusCode': 200}

def generate_recommendations(username, age, games):
    recommendations = []
    
    # Determine age group
    if age < 13:
        age_group = "kids"
    elif age < 18:
        age_group = "teens"
    else:
        age_group = "adults"
    
    # If user has selected games, use them for recommendations
    if games:
        for game in games:
            # Normalize game name to lowercase
            game_lower = game.lower()
            
            # Try to find recommendations for this specific game in the user's age group
            if game_lower in GAME_RECOMMENDATIONS.get(age_group, {}):
                game_recs = GAME_RECOMMENDATIONS[age_group][game_lower]
                
                # Add a random recommendation for this game
                if game_recs:
                    rec = random.choice(game_recs)
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': rec['title'],
                        'description': f"Based on your interest in {game}: {rec['description']}"
                    })
    
    # If we don't have enough recommendations, add some default ones
    if len(recommendations) < 3:
        # How many more recommendations we need
        needed = 3 - len(recommendations)
        
        # Add default recommendations
        default_recs = GAME_RECOMMENDATIONS['default'].copy()
        random.shuffle(default_recs)
        
        for i in range(min(needed, len(default_recs))):
            recommendations.append({
                'id': len(recommendations) + 1,
                'title': default_recs[i]['title'],
                'description': f"Popular with {age_group}: {default_recs[i]['description']}"
            })
    
    # Add a personalized age-specific recommendation
    age_specific_rec = {
        'id': len(recommendations) + 1,
        'title': f"Age Group: {age_group.title()}",
        'description': f"Recommended especially for {username} at age {age}"
    }
    
    recommendations.append(age_specific_rec)
    
    # Return at most 5 recommendations
    return recommendations[:5]