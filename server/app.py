from flask import Flask, request, jsonify
import logging
import json
import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ec2_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/process-data', methods=['POST'])
def process_data():
    data = request.json
    logger.info(f"Received data from Lambda: {json.dumps(data)}")
    
    try:
        data['ec2_timestamp'] = datetime.datetime.now().isoformat()
        
        data['status'] = "processed by ec2"
        
        logger.info(f"Successfully processed data: {json.dumps(data)}")
        
        return jsonify({
            'success': True,
            'message': "Data processed successfully",
            'data': data
        })
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error processing data: {str(e)}",
            'data': data
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting EC2 processing server...")
    app.run(host='0.0.0.0', port=0000, debug=False)