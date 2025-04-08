<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis - Preview</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis Dashboard</h1>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}

        {% if data %}
            <div class="box">
                <h2>Raw Data Preview</h2>
                {{ data | safe }}
            </div>
        {% endif %}

        {% if cleaned_data %}
            <div class="box">
                <h2>Cleaned Data Preview</h2>
                {{ cleaned_data | safe }}
            </div>
        {% endif %}

        {% if summary %}
            <div class="box">
                <h2>Data Summary</h2>
                <p>Rows: {{ summary.rows }}</p>
                <p>Columns: {{ summary.columns | join(', ') }}</p>
                <p>Types: {% for col, typ in summary.types.items() %}{{ col }}: {{ typ }}, {% endfor %}</p>
            </div>
        {% endif %}

        {% if before_counts %}
            <div class="box">
                <h2>Sentiment Counts (Before Training)</h2>
                <p>Positive: {{ before_counts.get('Positive', 0) }}</p>
                <p>Negative: {{ before_counts.get('Negative', 0) }}</p>
                <p>Neutral: {{ before_counts.get('Neutral', 0) }}</p>
            </div>
        {% endif %}

        {% if plots %}
            <div class="box">
                <h2>Visual Insights</h2>
                {% for name, img in plots.items() %}
                    <h3>{{ name.replace('wordcloud_', '') | capitalize }}</h3>
                    <img src="data:image/png;base64,{{ img }}" alt="{{ name }}">
                {% endfor %}
            </div>
        {% endif %}

        <div class="box">
            <h2>Train Model</h2>
            <form method="post" action="/train">
                <label>Text Column: 
                    <select name="text_col" required>
                        <option value="" disabled {% if not text_col %}selected{% endif %}>Select a column</option>
                        {% for col in columns %}
                            <option value="{{ col }}" {% if col == text_col %}selected{% endif %}>{{ col }}</option>
                        {% endfor %}
                    </select>
                </label>
                <label>Sentiment Column: 
                    <select name="sentiment_col" required>
                        <option value="" disabled {% if not sentiment_col %}selected{% endif %}>Select a column</option>
                        {% for col in columns %}
                            <option value="{{ col }}" {% if col == sentiment_col %}selected{% endif %}>{{ col }}</option>
                        {% endfor %}
                    </select>
                </label>
                <label>Split Ratio: <input type="number" step="0.1" min="0.5" max="0.9" name="split_ratio" value="0.8"></label>
                <button type="submit">Train</button>
            </form>
        </div>

        {% if metrics %}
            <div class="box">
                <h2>Training Results</h2>
                <p>Accuracy: {{ metrics.accuracy }}</p>
                <table class="data-table">
                    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
                    {% for cls, scores in metrics.report.items() %}
                        <tr><td>{{ cls }}</td><td>{{ scores.precision }}</td><td>{{ scores.recall }}</td><td>{{ scores['f1-score'] }}</td></tr>
                    {% endfor %}
                </table>
                <h3>Sentiment Counts (After Training - Test Set)</h3>
                <p>Positive: {{ metrics.after_counts.get('Positive', 0) }}</p>
                <p>Negative: {{ metrics.after_counts.get('Negative', 0) }}</p>
                <p>Neutral: {{ metrics.after_counts.get('Neutral', 0) }}</p>
            </div>
        {% endif %}

        <div class="box">
            <h2>Predict Sentiment</h2>
            <form method="post" action="/predict">
                <textarea name="text" rows="4" placeholder="Enter text here"></textarea>
                <input type="hidden" name="text_col" value="{{ text_col }}">
                <input type="hidden" name="sentiment_col" value="{{ sentiment_col }}">
                <button type="submit">Predict</button>
            </form>
        </div>

        {% if result %}
            <div class="box">
                <h2>Prediction Result</h2>
                <p>Text: {{ input_text }}</p>
                <p>Prediction: <span class="sentiment-{{ result.prediction.lower() }}">{{ result.prediction }}</span></p>
                <p>Probabilities: {% for cls, prob in result.probabilities.items() %}{{ cls }}: {{ prob }}, {% endfor %}</p>
            </div>
        {% endif %}

        <div class="box">
            <h2>Download Predictions</h2>
            <a href="/download?text_col={{ text_col }}"><button>Download CSV</button></a>
        </div>
    </div>
</body>
</html>