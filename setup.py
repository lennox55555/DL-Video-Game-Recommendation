from setuptools import setup, find_packages
import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = project_root / 'scripts'
sys.path.insert(0, str(scripts_dir))

class ProjectSetup:
    """Class to handle project setup operations."""
    
    @staticmethod
    def ensure_dependencies():
        """Ensure required dependencies are installed."""
        try:
            logger.info("Installing required dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "boto3", "python-dotenv"],
                check=True,
                capture_output=True
            )
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            return False

    @staticmethod
    def train_traditional_model():
        """Train the traditional (naive) recommendation model."""
        try:
            from naive import NaiveGameRecommender
            import os
            
            logger.info("Training traditional (naive) recommendation model...")
            data_path = os.path.join(project_root, "data/fake_user_data.csv")
            
            if not os.path.exists(data_path):
                logger.error(f"Data file not found at {data_path}")
                return False
                
            recommender = NaiveGameRecommender(data_path)
            
            # Training is implicit in initialization, but we'll evaluate to verify
            metrics = recommender.evaluate_metrics(test_ratio=0.2, k=10)
            
            logger.info("Traditional Model Evaluation Metrics:")
            logger.info(f"MSE: {metrics['MSE']:.4f}")
            logger.info(f"RMSE: {metrics['RMSE']:.4f}")
            logger.info(f"R^2: {metrics['R^2']:.4f}")
            logger.info(f"MAP@10: {metrics['MAP@10']:.4f}")
            logger.info(f"NDCG@10: {metrics['NDCG@10']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training traditional model: {e}")
            return False
    
    @staticmethod
    def train_deep_learning_model():
        """Train the deep learning recommendation model."""
        try:
            import sys
            import subprocess
            
            logger.info("Training deep learning recommendation model...")
            dl_script_path = os.path.join(scripts_dir, "deep_learning_training.py")
            
            if not os.path.exists(dl_script_path):
                logger.error(f"Deep learning training script not found at {dl_script_path}")
                return False
                
            # Prepare the environment
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
            
            # Run the training script
            result = subprocess.run(
                [sys.executable, dl_script_path],
                check=False,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                logger.info("Deep learning model training completed successfully")
                return True
            else:
                logger.error(f"Deep learning model training failed with error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error training deep learning model: {e}")
            return False

    @classmethod
    def run_setup(cls):
        """Run the project setup by configuring AWS and downloading data."""
        try:
            from aws_config import AWSConfigManager
            from pull_data import DataPuller

            logger.info("Setting up DL-Stock-Picker project...")

            # config AWS
            aws_config = AWSConfigManager()
            aws_success = aws_config.configure()

            if aws_success:
                # pull data
                data_puller = DataPuller()
                data_success = data_puller.pull_all_csv_files()
                
                if data_success:
                    logger.info("Data download completed successfully")
                    
                    # create dict
                    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
                    os.makedirs(os.path.join(project_root, "data/inference_data"), exist_ok=True)
                    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
                    
                    # train traditional model
                    traditional_success = cls.train_traditional_model()
                    
                    # train deep learning model
                    deeplearning_success = cls.train_deep_learning_model()
                    
                    if traditional_success and deeplearning_success:
                        logger.info("Project setup completed successfully with both models trained")
                    elif traditional_success:
                        logger.info("Project setup completed with only traditional model trained")
                    elif deeplearning_success:
                        logger.info("Project setup completed with only deep learning model trained")
                    else:
                        logger.warning("Project setup completed but model training failed")
                else:
                    logger.warning("Data download failed. Project setup incomplete.")
            else:
                logger.warning("AWS configuration failed. Data download skipped.")

        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.info("Please place aws_config.py and pull_data.py in the scripts directory")

if __name__ == "__main__":
    if ProjectSetup.ensure_dependencies():
        ProjectSetup.run_setup()
    else:
        logger.error("Failed to install dependencies. Setup aborted.")

setup(
    name="dl-stock-picker",
    version="0.1.0",
    description="DL-Stock-Picker - A Deep Learning Stock Picking Project",
    author="Lennox Anderson",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "boto3",
        "python-dotenv",
    ],
)