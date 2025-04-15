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

This script performs the complete project initialization process:
- Installs required dependencies (boto3, python-dotenv)
- Configures AWS using credentials from your .env file
- Downloads all necessary data files from the configured S3 bucket
- Creates required directories for data, inference_data, and models
- Trains three recommendation models:
  1. The Naive model that provides basic game recommendations
  2. The Traditional ML model using a RandomForest regressor with hyperparameter tuning
  3. The Deep Learning model using Neural Collaborative Filtering architecture
- Evaluates all models and reports performance metrics

The setup.py script serves as the entry point for new users, automating the entire setup process to get you from a fresh clone to a fully functioning recommendation system in one command.

You can ignore the "error: no commands supplied" message at the end of the setup script. This is a standard message from setuptools and doesn't affect the setup process.

5. **Run main.py**
```bash
python main.py
```

The main.py script functions as the inference engine for all recommendation models:
- Provides a command-line interface for running recommendation models
- Supports four modes: "naive", "traditional", "deep_learning", or "all" models
- Accepts a target game name and returns similar games as recommendations
- Can display detailed evaluation metrics for model performance
- Handles all path management and file access
- For deep learning inference, it creates necessary directories and files if they don't exist
- Passes the correct model path (deep_learning_model_500_combined.pth) to the inference script
- For traditional model, loads the trained RandomForest model from models/traditional_model.pkl

Options available:
```bash
python main.py --model [naive|traditional|deep_learning|all] --game [game_name] --top_n [number] --metrics
```

By default, it will run both models and return 5 recommendations for "minecraft" if no arguments are provided.




## Project Structure

```
DL-Video-Game-Recomendation/
├── .env                    # AWS credentials (create this file manually)
├── setup.py                # Setup file (AWS config, pulling data, and model training)
├── scripts/                # Utility scripts and scripts used for training models
│   ├── aws_config.py       # AWS configuration
│   └── pull_data.py        # Data download utilities
│   └── naive.py            # naive model 
│   └── deep_learning_inference.py        # inference script for the deep learning model
│   └── deep_learning_training.py        # training script for the DL model
│   └── push_data.py        # script to push data into the AWS S3 Bucket
│   └── traditional_inference.py        # script to run inference on the trad model
│   └── traditional_training.py        # script to train the traditional model
├── data/                   # Downloaded data files
│   └── all_video_games.csv # Game data (downloaded during setup)
├── models/                 # ML models
├── client/                 # Frontend code
└── server/                 # Backend code (this is deployed on the EC2 instance)
└── notebooks/              # notebook for experimentation and quick code execution
```

## Problem Addressed 

Most video game recommendation systems are integrated within proprietary platforms like the PlayStation Store or Steam. These systems often exhibit inherent biases, as companies tend to promote specific games. Additionally, the recommendations are typically limited to games available on their respective platforms (e.g., Steam focuses exclusively on PC games within its ecosystem or  titles they have rights to) and are tailored solely to their user base. In contrast, our approach stands out because it leverages data from Metacritic, a widely trusted platform where gamers and critics independently share ratings and reviews for a diverse range of games across different platforms and consoles. This dataset spans video games dating back to 1995, many of which are unlikely to be featured on modern proprietary platforms like the PlayStation Store. While we explored datasets related to Steam users in Australia and Steam-specific games, we opted for a platform-agnostic approach that relies on authentic reviews from critics and gamers across various platforms and consoles, ensuring broader coverage and reduced bias. While our goal is to offer a more comprehensive and unbiased approach to recommending video games, we acknowledge that it is impossible to fully account for potential biases in the critics' and gamers' reviews.

## Data Sources
**Video Game Dataset**
- Source: [Video Game Dataset](<https://www.kaggle.com/datasets/beridzeg45/video-games/data>)
- Author: Beridze Giorgi
- Description: "The dataset contains all the video games that have been featured on Metacritic.com from 1995 to January 2024. It includes over 14,000 unique video game titles across all platforms and genres"
- Usability: The dataset is structured in a tabular format, with a usabilty score of 10.0 
- License: There is no specified license here. The only thing mentioned is "Other (specified in description)", but the description does not contain details about the dataset's license
- Sourcing: "The data was collected using Python's Selenium and BeautifulSoup libraries" by scraping Metacritic.com website 
- Details: The dataset contains information about video games such as  Title, Release Date,Developer,Publisher, Genres, Product Rating,User Score,User Ratings Count etc..

This dataset is used in our project to supply items information.

**Metacritic PC Games Reviews**
- Source: [Video Game Users & Critics Dataset](<https://www.kaggle.com/datasets/beridzeg45/metacritic-pc-games-reviews/data>)
- Author: Beridze Giorgi
- Description: "This dataset is a collection of 512 thousand reviews for 5449 different games gathered from Metacritic.com"
- Usability: The dataset is structured in a tabular format, with a usabilty score of 10.0 
- License: There is no specified license here. The only thing mentioned is "Other (specified in description)", but the description does not contain details about the dataset's license
- Sourcing: "The data was collected using Python's Selenium and BeautifulSoup libraries" by scraping Metacritic.com website 
- Details: The dataset contains information about video games' reviews such as Game Title,Game Poster,Game Release Date,Game Developer,Genre,Platforms,Product Rating,Overall Metascore,Overall User Rating,Reviewer Name,Reviewer Type,Rating Given By The Reviewer,Review Date,Review Text
- Note: Since the dataset contains reviews from both critics and players, we are making the assumptions that both types of reviewers have played or interacted with the game long enough to be able to submit their review.

This dataset is used in our project to supply users information such as the user's rating of the different games (we are using the Rating Given By The Reviewer and normalizing it between 0 and 10 as the rating and the Reviewer Name as the user_id). 

## Review of Relevant Previous Efforts on this Dataset
When it comes to previous efforts done on the datasets, only a few notebooks have been pushed to the Kaggle dataset repo. All of these notebooks are performing EDA on the datasets, but none of them have gone to the extent of creating models to recommend items. One of the notebooks implements an ML model, but that is used to predict game features rather than recommending games. 

## Model evaluation process & Metric Selection 


To evaluate the effectiveness of the recommendation models, we employed a hybrid evaluation strategy focusing on both regression accuracy and ranking quality. The dataset is first split into a training and testing set, ensuring that only users seen during training are included in the test set to reflect a realistic inference scenario. Since the model is trained to predict the explicit user rating for a game (e.g., 8.5 out of 10), we first use Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to quantify how close the predicted ratings are to the actual user ratings. We also include the R² Score, which captures how much variance in user ratings the model can explain, offering a more interpretable metric for overall model fit. However, since the ultimate goal of the system is to recommend the most relevant games to users, we also evaluate how well the model ranks items using Mean Average Precision at K (MAP@10) and Normalized Discounted Cumulative Gain at K (NDCG@10). These ranking metrics assess how many of the top-K recommended games are actually relevant (MAP) and how highly ranked the relevant games are (NDCG), placing greater emphasis on correct recommendations near the top of the list. This hybrid evaluation setup ensures the model not only predicts ratings accurately but also delivers high-quality, position-aware recommendations in real-world usage



The model is evaluated using offline metrics on a held-out test set that contains user-game interactions not seen during training. For each user, the model generates a ranked list of top-K recommended games, excluding items the user has already interacted with. The predicted list is compared against the actual set of games rated by the user in the test set. The evaluation script computes four ranking-based metrics: Precision@K, Recall@K, Mean Average Precision (MAP@K), and Normalized Discounted Cumulative Gain (NDCG@K). These metrics were chosen to assess the model's ability to rank relevant items at the top of the recommendation list. Precision@K captures how many of the top-K items are actually relevant, Recall@K measures coverage of the relevant items, MAP@K rewards both relevance and ordering, and NDCG@K emphasizes the importance of placing relevant items earlier in the list. Together, they provide a comprehensive view of recommendation quality beyond simple rating prediction

To evaluate the effectiveness of the recommendation models, we employed a hybrid evaluation strategy focusing on both regression accuracy and ranking quality. The dataset is first split into a training and testing set, ensuring that only users seen during training are included in the test set to reflect a realistic inference scenario. Since the model is trained to predict the explicit user rating for a game (e.g., 8.5 out of 10), we first use Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to quantify how close the predicted ratings are to the actual user ratings. We also include the R² Score, which captures how much variance in user ratings the model can explain, offering a more interpretable metric for overall model fit. However, since the ultimate goal of the system is to recommend the most relevant games to users, we also evaluate how well the model ranks items using Mean Average Precision at K (MAP@10) and Normalized Discounted Cumulative Gain at K (NDCG@10). These ranking metrics assess how many of the top-K recommended games are actually relevant (MAP) and how highly ranked the relevant games are (NDCG), placing greater emphasis on correct recommendations near the top of the list. This hybrid evaluation setup ensures the model not only predicts ratings accurately but also delivers high-quality, position-aware recommendations in real-world usage



## Modeling Approach 

### Naive 

This system reads the CSV of video games ratings and calculates the average rating for each game. When the user picks a game they like on the web app, the model suggests other high-scoring games while mixing in a bit of randomness to add variety and extra details like a short description. It also checks how good its suggestions are by comparing its predicted scores to real ratings using a simple error metric (RMSE).

### Traditional Model 


### Deep Learning Model



The model uses a Neural Collaborative Filtering (NCF) architecture with separate GMF and MLP embedding layers. GMF computes the element-wise product of user and item embeddings to capture linear interactions. MLP concatenates the same embeddings and passes them through two fully connected layers with ReLU and dropout to model non-linear interactions. Outputs from both paths are concatenated and passed through a final linear layer to predict ratings. The model is trained using MSE loss and the Adam optimizer. Training uses a custom PyTorch Dataset and DataLoader, with 20% of the user-game interaction data used for training. The model runs for 50 epochs and saves weights after training for inference. The reason we are using 20% of the user-game interaction data for training is because the dataset is extremely big and our machines kept crashing (20% of the data still contains +100,000 data points). The model is trained to predict ratings for video games and thus we have chosen to use MSE as the main metric for training.

The model uses a Neural Collaborative Filtering (NCF) architecture with separate GMF and MLP embedding layers. GMF computes the element-wise product of user and item embeddings to capture linear interactions. MLP concatenates the same embeddings and passes them through two fully connected layers with ReLU and dropout to model non-linear interactions. Outputs from both paths are concatenated and passed through a final linear layer to predict ratings. The model is trained using MSE loss and the Adam optimizer. Training uses a custom PyTorch Dataset and DataLoader, with 20% of the user-game interaction data used for training. The model runs for 50 epochs and saves weights after training for inference. The reason we are using 20% of the user-game interaction data for training is because the dataset is extremely big and our machines kept crashing (20% of the data still contains +100,000 data points). The model is trained to predict ratings for video games and thus we have chosen to use MSE as the main metric for training.


## Data Processing pipeline 

The preprocessing pipeline first cleans the video game dataset by handling missing values, standardizing genres, and parsing nested platform information. Genres are converted to lowercase, stripped of whitespace, deduplicated, and one-hot encoded using MultiLabelBinarizer. Platform information is extracted from stringified dictionaries into separate lists of platform names and metascores, which are then expanded into individual columns. Additional features like developer, publisher, product rating, and user score are filled or imputed. The Release Date is converted into a Release Year integer. The final game dataset includes binary genre indicators and platform-specific metascore columns. In parallel, the user interaction dataset is processed by mapping user IDs and game titles to index-based IDs (user_idx, game_idx) using dictionaries for embedding compatibility. Cleaned datasets and mapping files are saved for downstream model use.


## Models evaluated and Model Selected 
**Naive Approach results**

- MSE: 8.4607
- RMSE: 2.9087
- R^2: -0.3020
- MAP@10: 0.0007
- NDCG@10: 0.0017

**Traditional Approach results**



**DL Apporach results**
We explored multiple deep learning approaches to recommendation, evaluating how different training objectives affect both rating accuracy and recommendation quality. The first model followed the standard Neural Collaborative Filtering (NCF) architecture trained with Mean Squared Error (MSE) loss to predict explicit user ratings. This model achieved excellent regression performance (MSE: 0.0177, RMSE: 0.1330, R²: 0.9978), but its ability to rank relevant games was limited (MAP@10: 0.0199, NDCG@10: 0.0457), revealing a disconnect between accurate rating prediction and practical recommendation quality. To address this, we trained a second version of the model using Bayesian Personalized Ranking (BPR) loss, optimizing directly for pairwise ranking. This model dramatically improved ranking metrics (MAP@10: 0.9253, NDCG@10: 0.9435) but lost accuracy in rating prediction (RMSE: 8.04), making it unsuitable for regression-based scoring. Observing the complementary strengths of both models, we introduced a hybrid training objective combining the two loss functions: α * MSE + (1 - α) * BPR. This combined approach delivered strong performance across the board, balancing rating accuracy (MSE: 0.2047, RMSE: 0.4525, R²: 0.9740) with robust ranking metrics (MAP@10: 0.8955, NDCG@10: 0.9236). A hybrid approach with both content-based and collaborative filtering might be better for our use case. However, due to time constraints, we decided to go with the model combining both losses for our deep learning approach.

Compare traditional and DL to the naive approach

## Demo of Deployed App 
[Insert link to the deployed app]

## Results and Conclusions 
Quick TLDR of what's in the README 

## Ethics Statement

Our project seeks to address biases in proprietary video game recommendation systems by leveraging data from Metacritic, a trusted platform for independent reviews from gamers and critics. While our approach is platform-agnostic and aims to provide broader coverage, we acknowledge that the data may still contain subjective biases inherent in user and critic reviews.

The datasets used, Video Game Dataset and Metacritic PC Games Reviews, were sourced from Kaggle and collected via web scraping. Since their licensing terms are unclear, we have taken care to use them responsibly and encourage others to verify usage rights before applying them commercially.

We are committed to transparency, ethical data usage, and improving bias mitigation strategies in future iterations of this project. By documenting our methods and assumptions, we aim to build trust while contributing to unbiased video game recommendations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

