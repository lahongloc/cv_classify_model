from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk

# nltk.download('punkt_tab')

df = pd.read_csv('./dataset/nyc-jobs-1.csv')
df = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
df['data'] = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df.drop(['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills'], axis=1, inplace=True)

data = list(df['data'])
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

model = Doc2Vec(vector_size=50,
                min_count=5,
                epochs=25,
                alpha=0.001
                )
model.build_vocab(tagged_data)
keys = model.wv.key_to_index.keys()
print(len(keys))
for epoch in range(model.epochs):
    print(f"Training epoch {epoch + 1}/{model.epochs}")
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

model.save('cv_job_maching.model')
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


def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)

    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text


pdf = PyPDF2.PdfReader('./CV/lahongloc.pdf')
resume = ""
for i in range(len(pdf.pages)):
    pageObj = pdf.pages[i]
    resume += pageObj.extract_text()

input_CV = preprocess_text(resume)
input_JD = preprocess_text(jd)

model = Doc2Vec.load('cv_job_maching.model')
v1 = model.infer_vector(input_CV.split())
v2 = model.infer_vector(input_JD.split())
similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
print(round(similarity, 2))

print("Model saved")
