import pandas as pd
import re
import numpy as np
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
stopwords_list = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()


def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W', ' ', text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and apply stemming
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_list]
    # Join the tokens back to a string
    processed_text = ' '.join(filtered_tokens)
    return processed_text


def train_model():
    # Load the dataset
    question_data = pd.read_csv("questions.csv", encoding="utf-8")
    answer_data = pd.read_csv("answers.csv", encoding="utf-8")

    # Extract the questions, correct answers, and scores
    questions = question_data['question'].tolist()
    correct_answers = question_data['answer'].tolist()
    scores = answer_data['score'].tolist()

    # Prepare input data
    X_train = []
    y_train = []

    for question_id, correct_answer, score in zip(question_data['question_id'], correct_answers, scores):
        for answer_id in range(1, 11):
            feature_vector = [question_id, answer_id, correct_answer]
            X_train.append(feature_vector)

            if answer_id == question_id:
                y_train.append(float(score))

    # Convert data to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Preprocess the questions and answers
    processed_questions = [preprocess_text(q) for q in questions]
    processed_answers = [preprocess_text(a) for a in correct_answers]
# to reanalyse
    # Prepare the training data
    X_train_processed = processed_answers

    # Define the pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('scaler', MaxAbsScaler()),
        ('regressor', LinearSVR())
    ])

    # Train the model
    pipeline.fit(X_train_processed, y_train)

    return pipeline


def predict_scores(model, student_answers):
    processed_answers = [preprocess_text(a) for a in student_answers]
    predicted_scores = model.predict(processed_answers)
    return predicted_scores


# Train the model
model = train_model()

# Pickle the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
