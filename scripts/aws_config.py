import os
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

class AWSConfigManager:
    """Class to manage AWS configuration from environment variables."""
    
    def __init__(self, env_path=None):
        """Initialize the AWS Config Manager.
        
        Args:
            env_path (str, optional): Path to .env file. If None, will look for .env in standard locations.
        """
        self.logger = logging.getLogger(__name__)
        
        if env_path:
            self.env_path = Path(env_path)
        else:
            project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.env_path = project_root / '.env'
        
        self._create_env_template()
    
    def _create_env_template(self):
        """Create a template .env file if it doesn't exist."""
        if not self.env_path.exists():
            self.logger.info(f"Creating template .env file at {self.env_path}")
            with open(self.env_path, 'w') as f:
                f.write("# AWS Credentials\n")
                f.write("AWS_ACCESS_KEY_ID=your_access_key_here\n")
                f.write("AWS_SECRET_ACCESS_KEY=your_secret_key_here\n")
                f.write("AWS_REGION=us-east-1\n")
                f.write("AWS_BUCKET_NAME=videogame-data-bucket\n")
            self.logger.info("Please edit the .env file with your AWS credentials")
    
    def configure(self):
        """Configure AWS CLI with credentials from .env file."""
        self.logger.info("Configuring AWS...")
        
        load_dotenv(self.env_path)
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        if not aws_access_key or aws_access_key == "your_access_key_here":
            self.logger.error("AWS Access Key not set in .env file")
            return False
        
        if not aws_secret_key or aws_secret_key == "your_secret_key_here":
            self.logger.error("AWS Secret Key not set in .env file")
            return False
        
        try:
            subprocess.run(["aws", "configure", "set", "aws_access_key_id", aws_access_key], check=True)
            subprocess.run(["aws", "configure", "set", "aws_secret_access_key", aws_secret_key], check=True)
            subprocess.run(["aws", "configure", "set", "region", aws_region], check=True)
            subprocess.run(["aws", "configure", "set", "output", "json"], check=True)
            
            self.logger.info("AWS configured successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error configuring AWS: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    aws_config = AWSConfigManager()
    aws_config.configure()