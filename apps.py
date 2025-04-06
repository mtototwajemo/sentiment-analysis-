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
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import io
import base64
import joblib
import logging
import json
import chardet
import os
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Utility Functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

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
        return None
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def detect_columns(df):
    text_candidates = [col for col in df.columns if any(k in col.lower() for k in ['text', 'review', 'comment'])]
    sentiment_candidates = [col for col in df.columns if any(k in col.lower() for k in ['sentiment', 'label', 'target'])]
    return text_candidates[0] if text_candidates else df.columns[0], sentiment_candidates[0] if sentiment_candidates else None

def get_data_summary(df):
    return {
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'class_dist': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
    }

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf8')

def generate_eda_plots(df, text_column, sentiment_column):
    plots = {}
    if sentiment_column in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=sentiment_column, data=df, palette=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.set_title("Sentiment Distribution")
        plots['sentiment_dist'] = plot_to_base64(fig)

        for sentiment in df[sentiment_column].unique():
            text = ' '.join(df[df[sentiment_column] == sentiment][text_column].dropna())
            if text:
                wc = WordCloud(width=400, height=200, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"{str(sentiment).capitalize()} Word Cloud")
                plots[f'word_cloud_{str(sentiment)}'] = plot_to_base64(fig)

    df['text_length'] = df[text_column].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['text_length'], bins=30, kde=True, color='#4ecdc4')
    ax.set_title("Text Length Distribution")
    plots['text_length'] = plot_to_base64(fig)
    return plots

def train_model(df, text_column, sentiment_column, model_type, split_ratio):
    df = df.dropna(subset=[text_column, sentiment_column])
    X = df[text_column].apply(clean_text)
    y = df[sentiment_column]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)
    
    # Sentiment counts before prediction
    counts_before = pd.Series(y).value_counts().to_dict()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    else:  # svm
        model = SVC(probability=True, class_weight='balanced')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Sentiment counts after prediction
    counts_after = pd.Series(y_pred).value_counts().to_dict()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return model, vectorizer, {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'confusion_matrix': cm.tolist(), 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'classes': model.classes_.tolist()
    }, counts_before, counts_after

def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    return pred, dict(zip(model.classes_, probs))

def get_explainability(text, model, vectorizer, class_names):
    explainer = LimeTextExplainer(class_names=[str(cls) for cls in class_names])
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text]).toarray()
    pred_fn = lambda x: model.predict_proba(vectorizer.transform(x))
    exp = explainer.explain_instance(cleaned_text, pred_fn, num_features=10)
    return exp.as_html()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if not file.filename:
        return redirect(url_for('index'))
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    file.save(file_path)
    df = load_file(open(file_path, 'rb'))
    if df is None or df.empty:
        return render_template('index.html', error="Invalid or empty file.")
    text_column, sentiment_column = detect_columns(df)
    summary = get_data_summary(df)
    columns = df.columns.tolist()
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                           text_column=text_column, sentiment_column=sentiment_column, summary=summary, columns=columns)

@app.route('/train', methods=['POST'])
def train():
    text_column = request.form['text_column']
    sentiment_column = request.form['sentiment_column']
    model_type = request.form['model_type']
    split_ratio = float(request.form['split_ratio'])
    
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    if not os.path.exists(file_path):
        return render_template('data_preview.html', error="No uploaded data found.")
    
    df = load_file(open(file_path, 'rb'))
    if df is None or text_column not in df.columns or (sentiment_column and sentiment_column not in df.columns):
        return render_template('data_preview.html', error="Invalid columns selected.")
    
    model, vectorizer, metrics, counts_before, counts_after = train_model(df, text_column, sentiment_column, model_type, split_ratio)
    eda_plots = generate_eda_plots(df, text_column, sentiment_column)
    
    df['cleaned_text'] = df[text_column].apply(clean_text)
    X = vectorizer.transform(df['cleaned_text'])
    df['predicted_sentiment'] = model.predict(X)
    
    columns = df.columns.tolist()
    top_5 = df[[text_column, 'predicted_sentiment']].head()
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                           text_column=text_column, sentiment_column=sentiment_column, metrics=metrics,
                           eda_plots=eda_plots, top_5=top_5,
                           counts_before=counts_before, counts_after=counts_after,
                           columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    pred, probs = predict_sentiment(text, model, vectorizer)
    explanation = get_explainability(text, model, vectorizer, model.classes_.tolist())
    
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    df = load_file(open(file_path, 'rb')) if os.path.exists(file_path) else pd.DataFrame()
    text_column, sentiment_column = detect_columns(df) if not df.empty else ("", "")
    summary = get_data_summary(df)
    columns = df.columns.tolist()
    
    return render_template('data_preview.html', data=df.head().to_html(classes='table table-striped table-hover'),
                           text_column=text_column, sentiment_column=sentiment_column, result=f"Predicted: {str(pred)}", probs=probs,
                           explanation=explanation, summary=summary, columns=columns)

@app.route('/download')
def download():
    file_path = os.path.join(os.getcwd(), 'uploaded_data.csv')
    if not os.path.exists(file_path):
        return render_template('data_preview.html', error="No data to download.")
    
    df = load_file(open(file_path, 'rb'))
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    text_column, _ = detect_columns(df)
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['predicted_sentiment'] = model.predict(vectorizer.transform(df['cleaned_text']))
    
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

if __name__ == '__main__':
    app.run(debug=True)