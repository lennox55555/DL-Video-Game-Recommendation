# 🎮 GameQuest: A Deep Learning Powered Game Recommendation System
  ![Background](https://media.giphy.com/media/l378BzHA5FwWFXVSg/giphy.gif) 

## Getting Started

Follow these instructions to set up the DL-Stock-Picker project on your local machine.

### Prerequisites

- Python 3.6 or higher
- AWS account with appropriate permissions for S3
- AWS CLI installed on your machine

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/lennox55555/DL-Video-Game-Recommendation.git
cd DL-Video-Game-Recommendation
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up AWS credentials**

Create a `.env` file in the project root directory with the following content:

```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=your_preferred_region
AWS_BUCKET_NAME=your_s3_bucket_name
```

Replace the placeholder values with your actual AWS credentials.

4. **Run setup.py**

```bash
python setup.py
```

This will:
- Configure AWS with your credentials
- Download the necessary data files from S3 to your local data directory

You can ignore the "error: no commands supplied" message at the end of the setup script. This is a standard message from setuptools and doesn't affect the setup process.

You should see `all_video_games.csv` or other project data files.

## Project Structure

```
DL-Stock-Picker/
├── .env                    # AWS credentials (create this file manually)
├── setup.py                # Main setup file
├── scripts/                # Utility scripts
│   ├── aws_config.py       # AWS configuration
│   └── pull_data.py        # Data download utilities
├── data/                   # Downloaded data files
│   └── all_video_games.csv # Game data (downloaded during setup)
├── models/                 # ML models
├── client/                 # Frontend code
└── server/                 # Backend code
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.