# Sentiment Analysis on Social Media Text

## Project Overview
This project classifies tweets into Positive, Negative, or Neutral sentiments using the Sentiment140 dataset. It includes text preprocessing, feature engineering, machine learning model building, and a Flask-based web interface.

## Dataset
- **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/jonathanoheix/sentiment140)
- **Records**: 1.6 million tweets
- **Classes**: 0 (Negative), 2 (Neutral), 4 (Positive)

## Features
- Cleaned and preprocessed text
- Polarity score (TextBlob)
- Text length

## Models
- Logistic Regression
- Random Forest Classifier

## How to Run
1. Train the model: `python main.py`
2. Launch the app: `python app.py`
3. Visit `http://127.0.0.1:5000` in your browser

## Dependencies
See `requirements.txt`
