import csv
import json

from model import preprocess_text
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    # Read questions from the CSV file
    questions = []
    with open('questions.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['question'])

    return render_template('index.html', questions=questions)


# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the answers from the request
    answers = request.json['answers']

    # Preprocess the answers
    preprocessed_answers = [preprocess_text(answer) for answer in answers]

    # Make predictions using the model
    results = []
    for answer, preprocessed_answer in zip(answers, preprocessed_answers):
        prediction = model.predict([preprocessed_answer])  # Pass the processed answer as a list
        prediction_list = prediction.tolist()  # Convert NumPy array to list
        results.append((answer, prediction_list))

    # Return the results as JSON
    return json.dumps(results)


# Define the route for rendering the HTML page

if __name__ == '__main__':
    app.run()
