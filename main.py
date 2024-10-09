from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import numpy as np
import pandas as pd
import re
import PyPDF2
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('./dataset/nyc-jobs-1.csv')
df = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
df['data'] = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df.drop(['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills'], axis=1, inplace=True)

data = list(df['data'])
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]


# Split the data manually
def split_data(data, test_size=0.2):
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * (1 - test_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    return train_data, val_data


train_data, val_data = split_data(tagged_data, test_size=0.2)


# Define and train the model with early stopping
def train_with_early_stopping(model, train_data, val_data, vector_size=100, min_count=5, alpha=0.01, epochs=20,
                              patience=3):
    best_model = None
    best_score = -np.inf
    no_improvement_count = 0

    model = Doc2Vec(vector_size=vector_size, min_count=min_count, alpha=alpha, epochs=1)  # Initialize model

    model.build_vocab(train_data)  # Build vocabulary before training

    for epoch in range(epochs):
        model.train(train_data, total_examples=model.corpus_count, epochs=1)

        # Evaluate performance on validation set
        val_scores = []
        for doc in val_data:
            inferred_vector = model.infer_vector(doc.words)
            similarity_scores = [
                np.dot(inferred_vector, model.dv[tag]) / (norm(inferred_vector) * norm(model.dv[tag]))
                for tag in model.dv.index_to_key
            ]
            val_scores.append(np.mean(similarity_scores))

        avg_val_score = np.mean(val_scores)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Score: {avg_val_score:.4f}")

        # Check for early stopping
        if avg_val_score > best_score:
            best_score = avg_val_score
            best_model = model
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_model


# Create the Doc2Vec model and train it with early stopping
model = Doc2Vec(vector_size=100, min_count=5, alpha=0.01, epochs=1)
trained_model = train_with_early_stopping(model, train_data, val_data)

# Save the trained model
trained_model.save('cv_job_matching.model')


# Load and preprocess CV and Job Description
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


pdf = PyPDF2.PdfReader('./CV/lahongloc.pdf')
jd = """
Seeking a passionate French Teacher to teach students of various proficiency levels, from beginners to advanced. The role involves planning and delivering engaging lessons, assessing student progress, and fostering a love for French language and culture.

Key Responsibilities:

Teach French to students at different levels.
Create lesson plans and adapt teaching methods.
Assess and track student progress.
Promote French culture and traditions.
Participate in school activities and meetings.
Requirements:

Proficiency in French.
Degree in French, Education, or related field.
Teaching certification and experience.
Strong communication and interpersonal skills.
Preferred:

Experience with online teaching and CEFR.
Familiarity with DELF/DALF exam preparation.
Benefits:

Competitive salary, professional development, and a supportive work environment.
Application: Submit resume, cover letter, and certifications to [Contact Information].
"""
for page in pdf.pages:
    jd += page.extract_text()

input_CV = preprocess_text(jd)
input_JD = preprocess_text(jd)

model = Doc2Vec.load('cv_job_matching.model')
v1 = model.infer_vector(word_tokenize(input_CV))
v2 = model.infer_vector(word_tokenize(input_JD))
similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
print(round(similarity, 2))
