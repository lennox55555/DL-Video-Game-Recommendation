import json
import logging
import boto3
import requests
import datetime
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_HOST = "ec2-44-211-30-27.compute-1.amazonaws.com"  
EC2_PORT = 5000  
EC2_ENDPOINT = f"http://{EC2_HOST}:{EC2_PORT}/process-data"
EC2_RANDOM_GAMES_ENDPOINT = f"http://{EC2_HOST}:{EC2_PORT}/random-games"


def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    route_key = event.get('requestContext', {}).get('routeKey')
    
    if route_key == '$connect':
        return handle_connect(event)
    elif route_key == '$disconnect':
        return handle_disconnect(event)
    elif route_key == 'sendUserData':
        return handle_user_data(event)
    elif route_key == 'getGameOptions':
        return handle_game_options(event)
    else:
        # direct API invocation
        username = event.get('username', 'anonymous')
        # removed age since we no longer use it
        model_type = event.get('modelType', 'Traditional')
        games = event.get('games', [])
        game_ratings = event.get('gameRatings', {})
        
        # log and validate the model type
        logger.info(f"Direct API: Model type from request: '{model_type}'")
        
        # make sure we use the correct model type (Naive, Deep Learning, or Traditional)
        if model_type not in ['Naive', 'Deep Learning', 'Traditional']:
            logger.warning(f"Invalid model type: '{model_type}', defaulting to Traditional")
            model_type = 'Traditional'
        
        # forward to EC2 for processing
        ec2_response = forward_to_ec2({
            'username': username,
            'modelType': model_type,
            'games': games,
            'gameRatings': game_ratings
        })
        
        # return the EC2 response
        return {
            'statusCode': 200,
            'recommendations': ec2_response.get('recommendations', []),
            'ec2_processed': ec2_response.get('success', False),
            'model_used': model_type
        }

def handle_connect(event):
    connection_id = event['requestContext']['connectionId']
    logger.info(f"New connection: {connection_id}")
    return {'statusCode': 200}

def handle_disconnect(event):
    connection_id = event['requestContext']['connectionId']
    logger.info(f"Connection closed: {connection_id}")
    return {'statusCode': 200}

def handle_game_options(event):
    """
    Handle a request for game options from the frontend.
    This is a simplified version that just returns a hardcoded list of games,
    without making requests to EC2.
    
    Args:
        event (dict): The WebSocket event
        
    Returns:
        dict: Response with status code
    """
    connection_id = event['requestContext']['connectionId']
    endpoint_url = f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
    
    try:
        body_str = event.get('body', '{}')
        logger.info(f"Raw game options request body: {body_str}")
        
        try:
            body = json.loads(body_str)
            logger.info(f"Parsed game options request: {body}")
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in request body: {body_str}")
            body = {}
        
        # get count of games to return (default to 9)
        count = int(body.get('count', 9))
        # ensure count is within reasonable limits
        count = max(1, min(count, 20))
        
        logger.info(f"Returning {count} default games for connection {connection_id}")
        
        # use hardcoded fallback games to avoid EC2 request
        default_games = [
            'Minecraft', 'Fortnite', 'Zelda', 'Mario',
            'Pokemon', 'GTA', 'COD', 'FIFA',
            'Skyrim'
        ]
        
        # prepare response data
        response_data = {
            'statusCode': 200,
            'success': True,
            'games': default_games[:count],
            'count': min(count, len(default_games)),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # log the response
        logger.info(f"Sending response with {len(response_data['games'])} default games")
        
        # send game options back to the client via the WebSocket
        api_gateway = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
        api_gateway.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(response_data)
        )
        logger.info(f"Successfully sent default games to {connection_id}")
        
    except Exception as e:
        logger.error(f"Error handling game options request: {str(e)}")
        
        # set simple fallback games
        fallback_games = [
            'Minecraft', 'Fortnite', 'Zelda', 'Mario',
            'Pokemon', 'GTA', 'COD', 'FIFA',
            'Skyrim'
        ]
        
        try:
            # send error response with default games
            error_response = {
                'statusCode': 500,
                'success': False,
                'message': f"Error processing request",
                'games': fallback_games,
                'count': len(fallback_games)
            }
            
            # log what we're sending back
            logger.info(f"Sending error response with fallback games")
            
            api_gateway = boto3.client('apigatewaymanagementapi', endpoint_url=endpoint_url)
            api_gateway.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps(error_response)
            )
            logger.info(f"Successfully sent error response to {connection_id}")
        except Exception as send_error:
            logger.error(f"Failed to send error response: {str(send_error)}")
    
    return {'statusCode': 200}

def handle_user_data(event):
    connection_id = event['requestContext']['connectionId']
    endpoint_url = f"https://{event['requestContext']['domainName']}/{event['requestContext']['stage']}"
    
    body = json.loads(event.get('body', '{}'))
    logger.info(f"Received user data: {body}")
    
    username = body.get('username', 'anonymous')
    # removed age since we no longer use it
    model_type = body.get('modelType', 'Traditional')
    games = body.get('games', [])
    game_ratings = body.get('gameRatings', {})
    
    # log the model type to diagnose issues
    logger.info(f"Model type from client: '{model_type}'")
    
    # make sure we use the correct model type (Naive, Deep Learning, or Traditional)
    if model_type not in ['Naive', 'Deep Learning', 'Traditional']:
        logger.warning(f"Invalid model type: '{model_type}', defaulting to Traditional")
        model_type = 'Traditional'
    
    # create a new copy of the body with the validated model type
    data_to_send = body.copy()
    data_to_send['modelType'] = model_type
    
    # forward to EC2 for processing
    ec2_response = forward_to_ec2(data_to_send)
    
    # extract recommendations from EC2 response
    recommendations = ec2_response.get('recommendations', [])
    logger.info(f"Got recommendations from EC2 for user {username} using {model_type} model")
    
    # prepare response data
    response_data = {
        'statusCode': 200,
        'recommendations': recommendations,
        'ec2_processed': ec2_response.get('success', False),
        'model_used': model_type
    }
    
    # send recommendations back to the client via the WebSocket
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
    Forward the message to the EC2 instance and get recommendations
    
    Args:
        message_data (dict): The message data to forward
        
    Returns:
        dict: Response from EC2 or error object if something failed
    """
    try:
        data_to_forward = message_data.copy()
        
        # add timestamp and status
        data_to_forward['timestamp'] = datetime.datetime.now().isoformat()
        data_to_forward['status'] = "passing to ec2"
        
        logger.info(f"Forwarding data to EC2: {json.dumps(data_to_forward)}")
        
        response = requests.post(
            EC2_ENDPOINT,
            json=data_to_forward,
            headers={'Content-Type': 'application/json'},
            timeout=5  
        )
        
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"Successfully received data from EC2")
            return {
                'success': True,
                'message': response_data.get('message', 'Successfully processed by EC2'),
                'recommendations': response_data.get('recommendations', []),
                'data': response_data.get('data', {})
            }
        else:
            logger.error(f"Failed to get response from EC2. Status code: {response.status_code}")
            return {
                'success': False,
                'message': f"EC2 returned status code {response.status_code}",
                'error': f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        logger.error("Timeout while connecting to EC2")
        return {
            'success': False,
            'message': "Timeout while connecting to EC2",
            'error': "timeout"
        }
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while connecting to EC2")
        return {
            'success': False,
            'message': "Unable to connect to EC2 server",
            'error': "connection_error"
        }
    except Exception as e:
        logger.error(f"Error forwarding message to EC2: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'error': "general_error"
        }