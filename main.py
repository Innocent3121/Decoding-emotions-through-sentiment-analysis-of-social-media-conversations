import pandas as pd
import re, string, joblib
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.translate(str.maketrans('', '', string.punctuation)).strip()

# Load and clean data
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['target'].isin([0, 2, 4])]
df['sentiment'] = df['target'].map({0: 'Negative', 2: 'Neutral', 4: 'Positive'})
df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['text_length'] = df['clean_text'].apply(len)

# Features and Labels
vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(df['clean_text']).toarray()
X = pd.DataFrame(X_vec)
X['polarity'] = df['polarity']
X['text_length'] = df['text_length']
y = df['sentiment']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
