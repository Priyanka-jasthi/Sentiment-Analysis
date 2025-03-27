# Sentiment-Analysis using twitter data

## Project Overview
This project focuses on sentiment analysis of tweets using natural language processing (NLP) and machine learning techniques. The goal is to classify tweets into three categories: **positive, negative, or neutral** sentiments.

**Dataset Story**
'tweet_labeled.csv' contains the tweets made in Twitter 2022, the dates of the tweets, and the labels as -1, 0 and 1 within the scope of the emotion contained in the tweets. 'tweets_21.csv' contains tweets from 2021.

tweets_labeled.csv
tweet_id: id information of tweet
tweet: tweet content
date: date and time of tweet
label: Tag information based on the sentiment of the tweet (-1:negative, 0:neutral, 1:positive)

tweets_21.csv
tweet_id: id information of tweet
tweet: tweet content
date: date and time of tweet

## Features
- Preprocessing of tweet text (removal of stopwords, special characters, etc.)
- Sentiment classification using **TextBlob**
- NLP-based feature extraction
- Machine learning model development for sentiment prediction

## Technologies Used
- **Python** (for data processing and model development)
- **TextBlob** (for sentiment analysis)
- **scikit-learn** (for machine learning models)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for visualization)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/twitter-sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd twitter-sentiment-analysis
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the sentiment analysis script:
   ```bash
   python sentiment_analysis.py
   ```
2. View results in the generated output file or visualize sentiment distributions.

## Dataset
- The dataset consists of tweets labeled as **positive, negative, or neutral**.
- Preprocessed for model training and evaluation.

## Future Improvements
- Implement deep learning-based sentiment analysis.
- Deploy the model using **Flask/Streamlit** for real-time predictions.
- Integrate Twitter API for live sentiment analysis.


