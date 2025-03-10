#!/usr/bin/env python3
# SVM classifier to classify emails as spam or non-spam using the Spam SMS Dataset.

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from svm import train_svm, predict_svm

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text) # Remove  special chracters
    text = text.lower() # Lowercase
    text = text.split() # Tokenize
    sanitized_text = []
    # Remove stopwords
    for word in text:
        if word not in stopwords.words('english'):
            sanitized_text.append(word)
    return " ".join(sanitized_text)


if __name__ == "__main__":
    # Load dataset
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', names=['label', 'message'])

    # Convert labels: "spam" -> 1, "ham" -> -1
    df['label'] = df['label'].map({'spam': 1, 'ham': -1})

    # Apply preprocessing
    nltk.download('stopwords')
    df['message'] = df['message'].apply(preprocess_text)

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["message"]).toarray()
    y = df['label'].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM
    w, b = train_svm(X_train, y_train)

    # Evalute model
    y_pred = predict_svm(X_test, w, b)
    accuracy = np.mean(y_pred == y_test)

    print(f"Test Accuracy: {accuracy:.4f}")
