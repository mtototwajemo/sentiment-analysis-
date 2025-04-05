from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import io
import base64
import joblib
import logging
import json
import chardet
import os

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Utility Functions
def clean_text(text, remove_stopwords=True, lemmatize=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def detect_and_map_sentiment(df, sentiment_column):
    if sentiment_column not in df.columns:
        return None
    values = df[sentiment_column].dropna().astype(str).str.lower()
    unique_values = set(values)
    try:
        numeric_values = pd.to_numeric(values, errors='coerce')
        if numeric_values.notna().all():
            min_val, max_val = numeric_values.min(), numeric_values.max()
            if min_val >= 1 and max_val <= 5:
                return lambda x: 'negative' if pd.to_numeric(x) <= 2 else ('neutral' if pd.to_numeric(x) == 3 else 'positive')
            elif min_val >= 0 and max_val <= 1:
                return lambda x: 'negative' if pd.to_numeric(x) < 0.4 else ('neutral' if pd.to_numeric(x) <= 0.6 else 'positive')
            else:
                terciles = np.percentile(numeric_values, [33, 66])
                return lambda x: 'negative' if pd.to_numeric(x) <= terciles[0] else ('neutral' if pd.to_numeric(x) <= terciles[1] else 'positive')
    except:
        pass
    common_labels = {'positive', 'negative', 'neutral', 'pos', 'neg', 'neu', 'good', 'bad'}
    if unique_values.issubset(common_labels):
        return lambda x: 'positive' if str(x).lower() in ['positive', 'pos', 'good'] else ('negative' if str(x).lower() in ['negative', 'neg', 'bad'] else 'neutral')
    if len(unique_values) <= 5:
        return lambda x: str(x).lower()
    return None

def detect_columns(df):
    text_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description', 'news', 'tweet'])]
    sentiment_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sentiment', 'label', 'target', 'class', 'score', 'rating'])]
    text_column = text_candidates[0] if text_candidates else df.select_dtypes(include=['object']).columns[0]
    sentiment_column = sentiment_candidates[0] if sentiment_candidates else None
    return text_column, sentiment_column

def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, delimiter='\t', encoding=encoding, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            return pd.DataFrame(data)
        else:
            return None
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def train_sentiment_model(df_train, text_column, sentiment_column, _mapping_func=None):
    if sentiment_column is None or _mapping_func is None:
        return None, None, 0, 0, 0, 0, {}, {}
    df_train = df_train.dropna(subset=[text_column])
    df_train['cleaned_text'] = df_train[text_column].apply(lambda x: clean_text(x))
    df_train['sentiment_mapped'] = df_train[sentiment_column].apply(_mapping_func)
    
    # Sentiment counts before review
    sentiment_counts_before = df_train['sentiment_mapped'].value_counts().to_dict()
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_train['cleaned_text'])
    y = df_train['sentiment_mapped']
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Sentiment counts after review (predicted)
    sentiment_counts_after = pd.Series(y_pred).value_counts().to_dict()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    return model, vectorizer, accuracy, precision, recall, f1, sentiment_counts_before, sentiment_counts_after, report

def load_model_and_vectorizer():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        return None, None
    return model, vectorizer

def predict_sentiment(texts, model, vectorizer, threshold=0.5):
    cleaned_texts = [clean_text(text) for text in texts]
    X = vectorizer.transform(cleaned_texts)
    probs = model.predict_proba(X)
    predictions = []
    for prob in probs:
        max_prob = max(prob)
        if max_prob < threshold and 'neutral' in model.classes_:
            predictions.append('neutral')
        else:
            predictions.append(model.classes_[np.argmax(prob)])
    return predictions, probs  # Fixed: Proper return statement

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf8')

def generate_eda_plots(df, text_column, sentiment_column, mapping_func):
    plots = {}
    if mapping_func and sentiment_column in df.columns:
        df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
        # Sentiment Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='sentiment_mapped', data=df, ax=ax, palette=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.set_title("Sentiment Distribution")
        plots['sentiment_dist'] = plot_to_base64(fig)
        # Word Cloud for Positive Sentiment
        positive_text = ' '.join(df[df['sentiment_mapped'] == 'positive'][text_column].dropna())
        if positive_text:
            wc = WordCloud(width=400, height=200, background_color='white', colormap='Greens').generate(positive_text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Positive Sentiment Word Cloud")
            plots['word_cloud'] = plot_to_base64(fig)
    # Text Length Distribution
    df['text_length'] = df[text_column].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax, color='#4ecdc4')
    ax.set_title("Text Length Distribution")
    plots['text_length'] = plot_to_base64(fig)
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    file.save(file_path)
    df = load_file(open(file_path, 'rb'))
    if df is None or df.empty:
        return render_template('index.html', error="Failed to load dataset or dataset is empty.")
    text_column, sentiment_column = detect_columns(df)
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                         text_column=text_column, sentiment_column=sentiment_column)

@app.route('/train', methods=['POST'])
def train_model():
    text_column = request.form['text_column']
    sentiment_column = request.form['sentiment_column'] if request.form['sentiment_column'] else None
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    if not os.path.exists(file_path):
        return render_template('data_preview.html', error="Error: 'uploaded_data.csv' not found. Please upload a file first.")
    df = load_file(open(file_path, 'rb'))
    mapping_func = detect_and_map_sentiment(df, sentiment_column) if sentiment_column else None
    model, vectorizer, accuracy, precision, recall, f1, sentiment_counts_before, sentiment_counts_after, report = train_sentiment_model(df, text_column, sentiment_column, mapping_func)
    
    if model is None:
        result = "Failed to train model. Check your sentiment column."
        eda_plots = None
        sentiment_counts_before = sentiment_counts_after = {}
    else:
        df['cleaned_text'] = df[text_column].apply(clean_text)
        predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer)
        df['predicted_sentiment'] = predictions
        top_5 = df[[text_column, 'predicted_sentiment']].head().to_html(classes='table table-striped table-hover')
        result = f"Model trained successfully! Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
        eda_plots = generate_eda_plots(df, text_column, sentiment_column, mapping_func)
    
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                         text_column=text_column, sentiment_column=sentiment_column, result=result,
                         top_5=top_5, eda_plots=eda_plots, sentiment_counts_before=sentiment_counts_before,
                         sentiment_counts_after=sentiment_counts_after, report=report)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return render_template('data_preview.html', error="Model not found. Please train the model first.")
    pred, prob = predict_sentiment([text], model, vectorizer)
    result = f"Sentiment: {pred[0].capitalize()}, Probabilities: {dict(zip(model.classes_, prob[0]))}"
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    df = load_file(open(file_path, 'rb')) if os.path.exists(file_path) else pd.DataFrame()
    text_column, sentiment_column = detect_columns(df) if not df.empty else ("", "")
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                         text_column=text_column, sentiment_column=sentiment_column, result=result)

@app.route('/download')
def download_results():
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    if not os.path.exists(file_path):
        return render_template('data_preview.html', error="Error: 'uploaded_data.csv' not found. Please upload a file first.")
    df = load_file(open(file_path, 'rb'))
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return render_template('data_preview.html', error="Model not found. Please train the model first.")
    text_column, _ = detect_columns(df)
    df['cleaned_text'] = df[text_column].apply(clean_text)
    predictions, _ = predict_sentiment(df['cleaned_text'], model, vectorizer)
    df['predicted_sentiment'] = predictions
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='processed_data.csv')

if __name__ == '__main__':
    app.run(debug=True)