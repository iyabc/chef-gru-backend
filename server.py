from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from keras import models
import sqlite3

from model_files import tokenizer_vars as tv
from model_files import LSTM_class
from model_files import constants as c

import database

app = Flask(__name__)
CORS(app)

model_emb = models.load_model('model/')
tokenizer = tv.getTokenizer()
max_len = 230

database.delete_table('evaluations_table')
database.create_table()

@app.route('/')
def show_docs():
    return render_template('documentation.html')

@app.route('/api/prediction', methods=['POST'])  
def get_prediction():
    ingredients = request.json.get('ingredients')
    if not ingredients:
        return jsonify({"error": "Ingredients not provided!"}), 400

    print(ingredients)
    prediction = predict(ingredients)
    return jsonify({"prediction": prediction})

def predict(ingredients):
    step = max_len-1
    
    input_prefix = f"{ingredients}"
    tokens_ind_prefix = tokenizer.encode(input_prefix).ids

    pred_emb = LSTM_class.ModelPredict(model_emb, tokens_ind_prefix, input_prefix, max_len, embedding=True)

    return pred_emb.generate_sequence(temperature=1)

def get_database_connection():
    return sqlite3.connect('evaluations.db')


@app.route('/api/eval', methods=['POST'])  
def get_evaluation():
    data = request.json.get('evaluator')
    name = data.get('name')
    email = data.get('email')
    contact_number = data.get('contact_number')
    profession = data.get('profession')
    scores = data.get('scores')

    if not name or not email or not contact_number or not profession or not scores:
        return jsonify({"error": "Invalid data provided!"}), 400
    
    print(data)

    database.insert_row(name, email, contact_number, profession, scores)
    database.view_table()

    return jsonify({"prediction": data})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
