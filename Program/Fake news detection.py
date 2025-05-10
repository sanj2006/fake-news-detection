import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("fake_news_dataset.csv")  # Adjust file path as needed

# Preprocessing: Clean text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df["clean_text"] = df["text"].apply(clean_text)  # Assuming "text" is the column with news articles

# Vectorization: Convert text into numerical format
vectorizer = TfidfVectorizer(max_features=5000)  # TF-IDF approach
X = vectorizer.fit_transform(df["clean_text"]).toarray()
y = df["label"]  # Assuming "label" column contains Fake (0) vs Real (1) classification

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Using Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predicting on a new sample
new_text = ["Breaking news! Scientists discover AI with human-level sarcasm!"]
new_text_clean = [clean_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(new_text_clean).toarray()
prediction = model.predict(new_text_vectorized)

print("Prediction (0=Fake, 1=Real):", prediction)