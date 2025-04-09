from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import io
import joblib
import logging
import chardet
import os

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, filename='sentiment_app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Sentiment mapping
SENTIMENT_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
REVERSE_MAP = {v: k for k, v in SENTIMENT_MAP.items()}

# Built-in mini dataset for default model
DEFAULT_DATA = pd.DataFrame({
    'text': [
        'this is great', 'awesome work', 'i love it', 'very good', 'happy day',
        'not good', 'this is bad', 'terrible day', 'i hate it', 'awful experience',
        'not bad', 'pretty decent', 'okay stuff', 'neutral vibe', 'fine today',
        'good effort', 'sad moment', 'strong win', 'weak loss', 'never awesome'
    ],
    'sentiment': [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2, 0, 0]
})

# Utility Functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return ' '.join(stemmer.stem(word) for word in tokens if word not in stop_words)

def clean_text_series(series):
    series = series.fillna('').str.lower()
    series = series.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    series = series.str.replace(r'[^\w\s]', '', regex=True)
    series = series.str.replace(r'\d+', '', regex=True)
    return series.apply(clean_text)

def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

def load_data(file):
    try:
        ext = file.filename.rsplit('.', 1)[-1].lower()
        encoding = detect_encoding(file)
        if ext == 'csv':
            return pd.read_csv(file, encoding=encoding, on_bad_lines='skip')
        elif ext == 'xlsx':
            return pd.read_excel(file)
        elif ext == 'txt':
            return pd.read_csv(file, delimiter='\t', encoding=encoding, on_bad_lines='skip')
        elif ext == 'json':
            return pd.DataFrame(json.load(file))
        return None
    except Exception as e:
        logging.error(f"Failed to load file: {e}")
        return None

def map_sentiments(series):
    def convert(val):
        if pd.isna(val):
            return 1
        if isinstance(val, (int, float)):
            return {0: 1, 1: 2, -1: 0}.get(val, 1)
        return REVERSE_MAP.get(str(val).capitalize(), 1)
    return series.apply(convert)

def get_sentiment_counts(df, sentiment_col):
    sentiment_series = map_sentiments(df[sentiment_col]).map(SENTIMENT_MAP)
    return sentiment_series.value_counts().to_dict()

def simple_sentiment(text):
    """Improved rule-based sentiment analysis with negation handling"""
    positive_words = {'good', 'great', 'awesome', 'happy', 'win', 'strong', 'love', 'decent'}
    negative_words = {'bad', 'terrible', 'awful', 'lose', 'weak', 'sad', 'hate'}
    negation_words = {'not', 'never', 'no', 'aint', 'isnt', 'arent', 'dont'}
    
    tokens = clean_text(text).split()
    pos_score = 0
    neg_score = 0
    
    for i, token in enumerate(tokens):
        # Check for negation in the previous token
        negated = i > 0 and tokens[i-1] in negation_words
        
        if token in positive_words:
            if negated:
                neg_score += 1  # "not good" → negative
            else:
                pos_score += 1
        elif token in negative_words:
            if negated:
                pos_score += 1  # "not bad" → positive
            else:
                neg_score += 1
    
    # More decisive scoring
    if pos_score > neg_score + 0.5:  # Slight bias to break ties
        return 2, {'Positive': 0.80, 'Neutral': 0.15, 'Negative': 0.05}
    elif neg_score > pos_score + 0.5:
        return 0, {'Positive': 0.05, 'Neutral': 0.15, 'Negative': 0.80}
    return 1, {'Positive': 0.35, 'Neutral': 0.40, 'Negative': 0.25}

def train_default_model():
    """Train a default model with built-in data if no user model exists"""
    if not all(os.path.exists(f) for f in ['model.pkl', 'tfidf.pkl', 'label_map.pkl']):
        X = clean_text_series(DEFAULT_DATA['text'])
        y = DEFAULT_DATA['sentiment']
        
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(X)
        
        model = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric='mlogloss')
        model.fit(X_tfidf, y)
        
        label_map = {i: i for i in range(3)}  # Identity map for simplicity
        joblib.dump(model, 'model.pkl')
        joblib.dump(tfidf, 'tfidf.pkl')
        joblib.dump(label_map, 'label_map.pkl')
        logging.info("Default model trained and saved.")

def train_model(df, text_col, sentiment_col, split_ratio=0.8):
    try:
        df = df.dropna(subset=[text_col, sentiment_col])
        if len(df) < 10:
            raise ValueError("Dataset too small (<10 rows).")
        
        X = clean_text_series(df[text_col])
        y = map_sentiments(df[sentiment_col])
        
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        reverse_label_map = {new: old for old, new in label_map.items()}
        y_mapped = y.map(label_map)
        
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
        X_tfidf = tfidf.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_mapped, train_size=split_ratio, random_state=42)
        
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_test_orig = y_test.map(reverse_label_map)
        y_pred_orig = pd.Series(y_pred).map(reverse_label_map)
        report = classification_report(y_test_orig, y_pred_orig, target_names=[SENTIMENT_MAP[i] for i in sorted(unique_labels)], output_dict=True)
        
        joblib.dump(model, 'model.pkl')
        joblib.dump(tfidf, 'tfidf.pkl')
        joblib.dump(label_map, 'label_map.pkl')
        
        df_test = df.iloc[X_test.indices]
        df_test['predicted_sentiment'] = y_pred_orig.map(SENTIMENT_MAP)
        after_counts = df_test['predicted_sentiment'].value_counts().to_dict()
        
        return {
            'accuracy': f"{accuracy:.2%}",
            'report': {k: {m: f'{v:.2f}' for m, v in v.items() if m != 'support'} for k, v in report.items() if k in [SENTIMENT_MAP[i] for i in sorted(unique_labels)]},
            'after_counts': after_counts
        }
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

# Routes
@app.route('/')
def index():
    train_default_model()  # Ensure default model is ready
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file'].filename:
        return redirect(url_for('index'))
    
    df = load_data(request.files['file'])
    if df is None or df.empty:
        return render_template('index.html', error="Invalid or empty file.")
    
    for file in ['uploaded_data.csv', 'model.pkl', 'tfidf.pkl', 'label_map.pkl']:
        if os.path.exists(file):
            os.remove(file)
    
    df.to_csv('uploaded_data.csv', index=False)
    columns = df.columns.tolist()
    text_col = columns[0]
    sentiment_col = columns[1] if len(columns) > 1 else columns[0]
    cleaned_df = df[[text_col]].copy()
    cleaned_df['cleaned_text'] = clean_text_series(df[text_col])
    before_counts = get_sentiment_counts(df, sentiment_col)
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table'),
                          cleaned_data=cleaned_df.head().to_html(classes='data-table'),
                          summary={'rows': len(df), 'columns': columns, 'types': {col: str(df[col].dtype) for col in columns}},
                          columns=columns, text_col=text_col, sentiment_col=sentiment_col,
                          before_counts=before_counts)

@app.route('/train', methods=['POST'])
def train():
    text_col = request.form['text_col']
    sentiment_col = request.form['sentiment_col']
    split_ratio = float(request.form.get('split_ratio', 0.8))
    
    if not os.path.exists('uploaded_data.csv'):
        return render_template('preview.html', error="No data uploaded yet.")
    
    df = pd.read_csv('uploaded_data.csv')
    if text_col not in df.columns or sentiment_col not in df.columns:
        return render_template('preview.html', error="Invalid column selection.")
    
    cleaned_df = df[[text_col]].copy()
    cleaned_df['cleaned_text'] = clean_text_series(df[text_col])
    before_counts = get_sentiment_counts(df, sentiment_col)
    metrics = train_model(df, text_col, sentiment_col, split_ratio)
    columns = df.columns.tolist()
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table'),
                          cleaned_data=cleaned_df.head().to_html(classes='data-table'),
                          summary={'rows': len(df), 'columns': columns, 'types': {col: str(df[col].dtype) for col in columns}},
                          metrics=metrics,
                          text_col=text_col, sentiment_col=sentiment_col,
                          columns=columns, before_counts=before_counts)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('index.html', error="Please enter text to predict.")
    
    train_default_model()  # Ensure a model is available
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    label_map = joblib.load('label_map.pkl')
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    probs_array = model.predict_proba(X)[0]
    probs = {SENTIMENT_MAP[reverse_label_map[i]]: prob for i, prob in enumerate(probs_array)}
    
    result = {
        'prediction': SENTIMENT_MAP[reverse_label_map[pred]],
        'probabilities': {k: f"{v:.2%}" for k, v in probs.items()}
    }
    
    df = pd.read_csv('uploaded_data.csv') if os.path.exists('uploaded_data.csv') else pd.DataFrame()
    columns = df.columns.tolist() if not df.empty else []
    text_col = request.form.get('text_col', columns[0] if columns else "text")
    sentiment_col = request.form.get('sentiment_col', columns[1] if len(columns) > 1 else "sentiment")
    cleaned_df = df[[text_col]].copy() if not df.empty else pd.DataFrame({'text': [text]})
    cleaned_df['cleaned_text'] = clean_text_series(cleaned_df[text_col if not df.empty else 'text'])
    before_counts = get_sentiment_counts(df, sentiment_col) if not df.empty else {}
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table') if not df.empty else None,
                          cleaned_data=cleaned_df.head().to_html(classes='data-table'),
                          summary={'rows': len(df), 'columns': columns, 'types': {col: str(df[col].dtype) for col in columns}} if not df.empty else None,
                          result=result, input_text=text,
                          text_col=text_col, sentiment_col=sentiment_col,
                          columns=columns, before_counts=before_counts)

@app.route('/download')
def download():
    if not os.path.exists('uploaded_data.csv') or not os.path.exists('model.pkl') or not os.path.exists('label_map.pkl'):
        return render_template('preview.html', error="No data or model available.")
    
    df = pd.read_csv('uploaded_data.csv')
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    label_map = joblib.load('label_map.pkl')
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    text_col = request.args.get('text_col', df.columns[0])
    df['cleaned_text'] = clean_text_series(df[text_col])
    preds = model.predict(tfidf.transform(df['cleaned_text']))
    df['prediction'] = pd.Series(preds).map(reverse_label_map).map(SENTIMENT_MAP)
    
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='sentiment_predictions.csv')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)