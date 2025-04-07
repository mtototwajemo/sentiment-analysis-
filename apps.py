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
import base64
import joblib
import logging
import chardet
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

def generate_plots(df, text_col, sentiment_col):
    plots = {}
    sample = df.sample(min(500, len(df)), random_state=42)
    
    if sentiment_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        sample[sentiment_col] = map_sentiments(sample[sentiment_col]).map(SENTIMENT_MAP)
        sns.countplot(x=sentiment_col, hue=sentiment_col, data=sample, palette=['#e74c3c', '#95a5a6', '#2ecc71'], legend=False)
        ax.set_title("Sentiment Distribution")
        plots['sentiment'] = plot_to_base64(fig)
    
    if text_col and sentiment_col:
        sample[sentiment_col] = map_sentiments(sample[sentiment_col]).map(SENTIMENT_MAP)
        for sentiment, colormap in [('Positive', 'Greens'), ('Negative', 'Reds'), ('Neutral', 'Greys')]:
            text = ' '.join(sample[sample[sentiment_col] == sentiment][text_col].dropna())
            if text:
                wc = WordCloud(width=300, height=150, background_color='white', max_words=30, colormap=colormap).generate(text)
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.imshow(wc)
                ax.axis('off')
                ax.set_title(f"{sentiment} Words")
                plots[f'wordcloud_{sentiment.lower()}'] = plot_to_base64(fig)
    
    return plots

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def train_model(df, text_col, sentiment_col, split_ratio=0.8):
    try:
        df = df.dropna(subset=[text_col, sentiment_col])
        if len(df) < 10:
            raise ValueError("Dataset too small (<10 rows).")
        
        X = clean_text_series(df[text_col])
        y = map_sentiments(df[sentiment_col])
        
        # Reindex y to consecutive integers starting from 0
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        reverse_label_map = {new: old for old, new in label_map.items()}
        y_mapped = y.map(label_map)
        
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_mapped, train_size=split_ratio, random_state=42)
        
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_test_orig = y_test.map(reverse_label_map)
        y_pred_orig = pd.Series(y_pred).map(reverse_label_map)
        report = classification_report(y_test_orig, y_pred_orig, target_names=[SENTIMENT_MAP[i] for i in sorted(unique_labels)], output_dict=True)
        
        joblib.dump(model, 'model.pkl')
        joblib.dump(tfidf, 'tfidf.pkl')
        joblib.dump(label_map, 'label_map.pkl')
        
        return {
            'accuracy': f"{accuracy:.2%}",
            'report': {k: {m: f'{v:.2f}' for m, v in v.items() if m != 'support'} for k, v in report.items() if k in [SENTIMENT_MAP[i] for i in sorted(unique_labels)]}
        }
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file'].filename:
        return redirect(url_for('index'))
    
    df = load_data(request.files['file'])
    if df is None or df.empty:
        return render_template('index.html', error="Invalid or empty file.")
    
    df.to_csv('uploaded_data.csv', index=False)
    columns = df.columns.tolist()
    
    summary = {
        'rows': len(df),
        'columns': columns,
        'types': {col: str(df[col].dtype) for col in columns}
    }
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table'),
                          summary=summary, columns=columns)

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
    
    metrics = train_model(df, text_col, sentiment_col, split_ratio)
    plots = generate_plots(df, text_col, sentiment_col)
    columns = df.columns.tolist()
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table'),
                          metrics=metrics, plots=plots,
                          text_col=text_col, sentiment_col=sentiment_col,
                          columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('preview.html', error="Please enter text to predict.")
    
    if not os.path.exists('model.pkl') or not os.path.exists('tfidf.pkl') or not os.path.exists('label_map.pkl'):
        return render_template('preview.html', error="Train a model first!")
    
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    label_map = joblib.load('label_map.pkl')
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    
    pred_orig = reverse_label_map[pred]
    probs_orig = {reverse_label_map[i]: prob for i, prob in enumerate(probs) if i in reverse_label_map}
    
    result = {
        'prediction': SENTIMENT_MAP[pred_orig],
        'probabilities': {SENTIMENT_MAP[k]: f"{v:.2%}" for k, v in probs_orig.items()}
    }
    
    df = pd.read_csv('uploaded_data.csv') if os.path.exists('uploaded_data.csv') else pd.DataFrame()
    columns = df.columns.tolist()
    text_col = request.form.get('text_col', columns[0] if columns else "text")
    sentiment_col = request.form.get('sentiment_col', columns[1] if len(columns) > 1 else "sentiment")
    plots = generate_plots(df, text_col, sentiment_col) if not df.empty else {}
    
    return render_template('preview.html', 
                          data=df.head().to_html(classes='data-table') if not df.empty else None,
                          result=result, input_text=text, plots=plots,
                          text_col=text_col, sentiment_col=sentiment_col,
                          columns=columns)

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