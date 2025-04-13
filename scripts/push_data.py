import os
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError

class DataPusher:
    """Class to upload data to S3 from the local data directory."""
    
    def __init__(self, env_path=None, data_dir=None):
        """Initialize the DataPusher.
        
        Args:
            env_path (str, optional): Path to .env file. If None, will look for .env in standard locations.
            data_dir (str, optional): Path to data directory. If None, will use project_root/data.
        """
        self.logger = logging.getLogger(__name__)
        
        if env_path:
            self.env_path = Path(env_path)
        else:
            project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.env_path = project_root / '.env'
        
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = project_root / 'data'
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def push_data(self, file_name, object_key=None):
        """Upload a file to S3 bucket.
        
        Args:
            file_name (str): Local filename in the data directory.
            object_key (str, optional): Desired object key in the S3 bucket. Defaults to file_name.
        
        Returns:
            bool: True if upload is successful, False otherwise.
        """
        self.logger.info("Uploading data to S3...")
        
        load_dotenv(self.env_path)
        
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        
        if not bucket_name:
            self.logger.warning("AWS_BUCKET_NAME not set in .env file")
            bucket_name = input("Enter your S3 bucket name: ")
        
        file_path = self.data_dir / file_name
        object_key = object_key or file_name
        
        try:
            s3_client = boto3.client('s3')
            
            self.logger.info(f"Uploading {file_path} to {bucket_name} as {object_key}")
            s3_client.upload_file(str(file_path), bucket_name, object_key)
            
            self.logger.info("Upload successful.")
            return True
        
        except ClientError as e:
            self.logger.error(f"Error uploading data to S3: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return False

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_pusher = DataPusher()
    data_pusher.push_data("fake_user_data.csv")
    print("User data addition")

if __name__ == "__main__":
    main()