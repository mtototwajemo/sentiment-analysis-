# Sentiment Analysis App

A Flask-based web app for sentiment analysis with advanced features.

## Features
1. **Upload CSV Data**: Supports CSV, Excel, TXT, JSON with dynamic column detection.
2. **Data Preview**: Displays data types, null values, and class distribution.
3. **Text Preprocessing**: Removes stopwords, punctuation, numbers, URLs, emojis; applies stemming.
4. **EDA**: Bar chart of sentiment classes, text length distribution, word clouds per sentiment.
5. **Train/Test Split**: Configurable split ratio (80-20, 70-30).
6. **Model Selection**: Choose Logistic Regression, Naive Bayes, or SVM.
7. **Model Training & Evaluation**: Shows accuracy, precision, recall, F1-score, confusion matrix.
8. **Cross Validation**: Displays average score and standard deviation.
9. **Text Prediction**: Predict sentiment for user-input text.
10. **Download Results**: Export predictions as CSV.
11. **Visualizations**: Word clouds, bar plots, confusion matrix.
12. **Explainability**: Feature importance via LIME (top TF-IDF words).

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/mtototwajemo/sentiment-analysis-app.git
   cd sentiment-analysis-app