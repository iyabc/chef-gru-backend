from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import tensorflow as tf
import sqlite3

from model_files.classes.Translator import Translator
from model_files.classes.ChefTokenizer import getChefTokenizer

import database

app = Flask(__name__)
CORS(app)

database.create_table()
database.view_table('evaluations_table')

UNITS = 512
FILE_NAME = 'final_model_gru_v100'
MAX_LENGTH = 313
path='model_files/'
file_suffix = '_0_200'


@app.route('/')
def show_docs():
    return render_template('documentation.html')

def get_database_connection():
    return sqlite3.connect('evaluations.db')

@app.route('/api/eval', methods=['POST'])  
def get_evaluation():
    data = request.json.get('evaluator')
    contact_number = data.get('contact_number') or 'none'
    recipe_output = data.get('recipe_output')
    user_input = data.get('user_input')
    years_experience = data.get('years_experience')
    clarity_rating = data.get('clarity_rating') or 0
    creativity_rating = data.get('creativity_rating') or 0
    suitability_rating = data.get('suitability_rating') or 0
    doability_rating = data.get('doability_rating') or 0
    likelihood_to_try_rating = data.get('likelihood_to_try_rating') or 0
    overall_rating = data.get('overall_rating') or 0

    print("sus", suitability_rating)

    if contact_number is None or recipe_output is None or user_input is None or years_experience is None or clarity_rating is None or creativity_rating is None or suitability_rating is None or doability_rating is None or likelihood_to_try_rating is None or overall_rating is None:
        return jsonify({"error": "Invalid data provided!"}), 400

    database.insert_row(contact_number, years_experience, user_input, recipe_output, clarity_rating, creativity_rating, suitability_rating, doability_rating, likelihood_to_try_rating, overall_rating)
    database.view_table('evaluations_table')

    return jsonify({"evaluation": data})

def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

@app.route('/api/prediction', methods=['POST'])  
def get_prediction():
    ingredients = request.json.get('ingredients')
    if not ingredients:
        return jsonify({"error": "Ingredients not provided!"}), 400
        
    encodedTokenizer, decodedTokenizer = getChefTokenizer(
    f"{path}tokenizers/encodedTokenizer{file_suffix}-vocab.json",
    f"{path}tokenizers/encodedTokenizer{file_suffix}-merges.txt",
    f"{path}tokenizers/decodedTokenizer{file_suffix}-vocab.json",
    f"{path}tokenizers/decodedTokenizer{file_suffix}-merges.txt")

    model = Translator(UNITS, encodedTokenizer, decodedTokenizer)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=masked_loss,
                metrics=[masked_acc, masked_loss])

    checkpoint_dir = f'final_model_gru_v100'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    print(ingredients)
    result = model.translate(ingredients, max_length=MAX_LENGTH, temperature=0.07)
    return jsonify({"prediction": result[0].numpy().decode()})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
