#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    @classmethod
    def run_setup(cls):
        """Run the project setup by configuring AWS and downloading data."""
        try:
            from aws_config import AWSConfigManager
            from pull_data import DataPuller

            logger.info("Setting up DL-Stock-Picker project...")

            aws_config = AWSConfigManager()
            aws_success = aws_config.configure()

            if aws_success:
                data_puller = DataPuller()
                data_success = data_puller.pull_all_csv_files()
                
                if data_success:
                    logger.info("Project setup completed successfully")
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