import os
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError

class DataPuller:
    """Class to pull data from S3 and save it to the data directory."""

    def __init__(self, env_path=None, data_dir=None):
        """Initialize the DataPuller.
        
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

        self.data_dir.mkdir(exist_ok=True)
    
    def _get_bucket_name(self):
        """Get the S3 bucket name from environment variables.
        
        Returns:
            str: The bucket name
        """
        load_dotenv(self.env_path)
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        
        if not bucket_name:
            self.logger.warning("AWS_BUCKET_NAME not set in .env file")
            bucket_name = input("Enter your S3 bucket name: ")
            
        return bucket_name
    
    def pull_all_csv_files(self):
        """Download all CSV files from S3 bucket.
        
        Returns:
            bool: True if all files were downloaded successfully, False otherwise.
        """
        self.logger.info("Downloading all CSV files from S3...")
        
        bucket_name = self._get_bucket_name()
        
        try:
            s3_client = boto3.client('s3')
            
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' not in response:
                self.logger.warning(f"No files found in bucket {bucket_name}")
                return False
            
            csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.csv')]
            
            if not csv_files:
                self.logger.warning(f"No CSV files found in bucket {bucket_name}")
                return False
            
            self.logger.info(f"Found {len(csv_files)} CSV files in bucket {bucket_name}")
            
            all_successful = True
            for file_key in csv_files:
                local_file_path = self.data_dir / file_key
                
                try:
                    self.logger.info(f"Downloading {file_key} from {bucket_name} to {local_file_path}")
                    s3_client.download_file(bucket_name, file_key, str(local_file_path))
                    self.logger.info(f"Successfully downloaded {file_key}")
                except Exception as e:
                    self.logger.error(f"Error downloading {file_key}: {e}")
                    all_successful = False
            
            return all_successful
            
        except ClientError as e:
            self.logger.error(f"Error accessing S3 bucket: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return False
    
    def pull_data(self, object_key="all_video_games.csv"):
        """Download specific data file from S3 bucket.
        
        Args:
            object_key (str): The object key (file name) in the S3 bucket.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        self.logger.info(f"Downloading {object_key} from S3...")

        bucket_name = self._get_bucket_name()
        local_file_path = self.data_dir / object_key

        try:
            s3_client = boto3.client('s3')

            self.logger.info(f"Downloading {object_key} from {bucket_name} to {local_file_path}")
            s3_client.download_file(bucket_name, object_key, str(local_file_path))

            self.logger.info(f"Data downloaded successfully to {local_file_path}")
            return True

        except ClientError as e:
            self.logger.error(f"Error downloading data from S3: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_puller = DataPuller()
    data_puller.pull_all_csv_files()