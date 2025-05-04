import joblib
from textblob import TextBlob
import re, string
import numpy as np

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.translate(str.maketrans('', '', string.punctuation)).strip()

def predict_sentiment(text):
    clean = clean_text(text)
    polarity = TextBlob(clean).sentiment.polarity
    length = len(clean)
    vec = vectorizer.transform([clean]).toarray()
    features = np.append(vec, [[polarity, length]], axis=1)
    prediction = model.predict(features)[0]
    return prediction

# Test sample
print(predict_sentiment("I love the new product update!"))
print(predict_sentiment("This service is terrible and slow."))
print(predict_sentiment("It’s okay, not great but not bad either."))
