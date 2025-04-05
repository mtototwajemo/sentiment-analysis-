# Sentiment Analysis App

A Flask-based web application for sentiment analysis of text data. Upload a dataset (CSV, Excel, TXT, JSON), train a logistic regression model, and predict sentiments with detailed metrics and visualizations.

## Features
- Upload datasets in multiple formats (.csv, .xlsx, .txt, .json)
- Automatically detect text and sentiment columns
- Train a sentiment analysis model with accuracy, precision, recall, and F1-score
- Display sentiment counts (positive, negative, neutral) before and after prediction
- Generate EDA plots (sentiment distribution, word cloud, text length)
- Predict sentiment for custom text input
- Download processed results as CSV

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/mtototwajemo/sentiment-analysis-app.git
   cd sentiment-analysis-app