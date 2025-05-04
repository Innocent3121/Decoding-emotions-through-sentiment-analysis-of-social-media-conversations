from flask import Flask, request, render_template
import joblib
from textblob import TextBlob
import re, string

app = Flask(__name__)
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.translate(str.maketrans('', '', string.punctuation)).strip()

@app.route('/')
def home():
    return '''
    <form method="POST" action="/predict">
        <input name="text" placeholder="Enter tweet..." style="width:300px">
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    clean = clean_text(text)
    polarity = TextBlob(clean).sentiment.polarity
    length = len(clean)

    vec = vectorizer.transform([clean]).toarray()
    import numpy as np
    features = np.append(vec, [[polarity, length]], axis=1)

    prediction = model.predict(features)[0]
    return f"Sentiment: <strong>{prediction}</strong>"

if __name__ == '__main__':
    app.run(debug=True)
