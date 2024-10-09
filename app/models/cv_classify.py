import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

resume_data_path = os.path.join(base_dir, '..', 'dataset', 'Resume.csv')
resume_data = pd.read_csv(resume_data_path)

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import re


def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text


resume_data['Resume_str'] = resume_data['Resume_str'].apply(clean_text)


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    sentences = sent_tokenize(text)
    features = {'feature': ""}  # Initialize the features dictionary
    stop_words = set(stopwords.words("english"))  # Define stop words

    for sent in sentences:
        if any(criteria in sent for criteria in ['skills', 'education']):
            words = word_tokenize(sent)  # Tokenize the sentence
            words = [word for word in words if word not in stop_words]  # Remove stop words
            tagged_words = pos_tag(words)  # Part-of-speech tagging
            # Filter out unnecessary tags
            filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
            # Accumulate the filtered words
            features['feature'] += " ".join(filtered_words) + " "

    return features['feature'].strip()  # Return the processed text as a string


resume_data['Processed_Resume_str'] = resume_data['Resume_str'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(resume_data['Processed_Resume_str'])

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, resume_data['Category'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


def classify_resume(new_resume_text):
    processed_text = preprocess_text(new_resume_text)
    text_features = tfidf.transform([processed_text])
    predicted_category = model.predict(text_features)
    return predicted_category[0]

