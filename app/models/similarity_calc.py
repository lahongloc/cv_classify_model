import nltk


# Download necessary NLTK datasets
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


# download_nltk_data()
from nltk.corpus import wordnet, stopwords
import string
import nltk

# Initialize stop words
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


# Preprocess text
def preprocess_text(text):
    # Tokenize text, remove stop words and punctuation
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words and word not in punctuation]

    # Return processed text
    return ' '.join(words)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Calculate similarity between two texts
def calculate_similarity(cv_text, job_desc_text):
    # Preprocess the texts
    cv_text_processed = preprocess_text(cv_text)
    job_desc_text_processed = preprocess_text(job_desc_text)

    # Create a CountVectorizer to convert text to vectors
    vectorizer = CountVectorizer().fit_transform([cv_text_processed, job_desc_text_processed])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors)

    return cosine_sim[0][1]  # Return similarity score between CV and job description
