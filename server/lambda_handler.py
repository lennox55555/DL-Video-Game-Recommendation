import json
import logging
import random
import boto3
import requests
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_HOST = ""  
EC2_PORT = 0000 
EC2_ENDPOINT = f"http://{EC2_HOST}:{EC2_PORT}/process-data"

GAME_RECOMMENDATIONS = {
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
    
    "default": [
        {"title": "Minecraft", "description": "Creative sandbox with building and exploration"},
        {"title": "Rocket League", "description": "Unique sports game combining soccer and vehicles"},
        {"title": "Among Us", "description": "Social deduction game with teamwork and deception"},
        {"title": "Stardew Valley", "description": "Relaxing farming sim with relationship building"}
    ]
}

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    route_key = event.get('requestContext', {}).get('routeKey')
    
    if route_key == '$connect':
        return handle_connect(event)
    elif route_key == '$disconnect':
        return handle_disconnect(event)
    elif route_key == 'sendUserData':
        return handle_user_data(event)
    else:
        username = event.get('username', 'anonymous')
        age = event.get('age', 25)
        games = event.get('games', [])
        recommendations = generate_recommendations(username, age, games)
        
        message_tagged = forward_to_ec2({
            'username': username,
            'age': age,
            'games': games
        })
        
        return {
            'statusCode': 200,
            'recommendations': recommendations,
            'ec2_processed': message_tagged
        }

def handle_connect(event):
    connection_id = event['requestContext']['connectionId']
    logger.info(f"New connection: {connection_id}")
    return {'statusCode': 200}

def handle_disconnect(event):
    connection_id = event['requestContext']['connectionId']
    logger.info(f"Connection closed: {connection_id}")
    return {'statusCode': 200}

def handle_user_data(event):
    connection_id = event['requestContext']['connectionId']
    endpoint_url = f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
    
    body = json.loads(event.get('body', '{}'))
    logger.info(f"Received user data: {body}")
    
    username = body.get('username', 'anonymous')
    age = body.get('age', 25)
    games = body.get('games', [])
    
    message_tagged = forward_to_ec2(body)
    
    recommendations = generate_recommendations(username, age, games)
    
    response_data = {
        'statusCode': 200,
        'recommendations': recommendations,
        'ec2_processed': message_tagged
    }
    
    try:
        api_gateway = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        api_gateway.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(response_data)
        )
        logger.info(f"Successfully sent recommendations to {connection_id}")
    except Exception as e:
        logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
    
    return {'statusCode': 200}

def forward_to_ec2(message_data):
    """
    Forward the message to the EC2 instance and tag it
    
    Args:
        message_data (dict): The message data to forward
        
    Returns:
        bool: True if successfully processed by EC2, False otherwise
    """
    try:
        data_to_forward = message_data.copy()
        
        data_to_forward['timestamp'] = datetime.datetime.now().isoformat()
        
        data_to_forward['status'] = "passing to ec2"
        
        logger.info(f"Forwarding data to EC2: {json.dumps(data_to_forward)}")
        
        response = requests.post(
            EC2_ENDPOINT,
            json=data_to_forward,
            headers={'Content-Type': 'application/json'},
            timeout=100
        )
        
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"Successfully forwarded message to EC2: {response_data}")
            data_to_forward['status'] = "passed to ec2"
            return True
        else:
            logger.error(f"Failed to forward message to EC2. Status code: {response.status_code}")
            data_to_forward['status'] = "ec2 forwarding failed"
            return False
            
    except Exception as e:
        logger.error(f"Error forwarding message to EC2: {str(e)}")
        return False

def generate_recommendations(username, age, games):
    recommendations = []
    
    # determine age group
    if age < 13:
        age_group = "kids"
    elif age < 18:
        age_group = "teens"
    else:
        age_group = "adults"
    
    if games:
        for game in games:
            game_lower = game.lower()
            
            if game_lower in GAME_RECOMMENDATIONS.get(age_group, {}):
                game_recs = GAME_RECOMMENDATIONS[age_group][game_lower]
                
                if game_recs:
                    rec = random.choice(game_recs)
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': rec['title'],
                        'description': f"Based on your interest in {game}: {rec['description']}"
                    })
    
    if len(recommendations) < 3:
        needed = 3 - len(recommendations)
        
        default_recs = GAME_RECOMMENDATIONS['default'].copy()
        random.shuffle(default_recs)
        
        for i in range(min(needed, len(default_recs))):
            recommendations.append({
                'id': len(recommendations) + 1,
                'title': default_recs[i]['title'],
                'description': f"Popular with {age_group}: {default_recs[i]['description']}"
            })
    
    age_specific_rec = {
        'id': len(recommendations) + 1,
        'title': f"Age Group: {age_group.title()}",
        'description': f"Recommended especially for {username} at age {age}"
    }
    
    recommendations.append(age_specific_rec)
    
    return recommendations[:5]