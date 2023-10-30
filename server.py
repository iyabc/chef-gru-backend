from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import tensorflow as tf
import sqlite3

from model_files.classes.Translator import Translator
from model_files.classes.ChefTokenizer import getChefTokenizer

import database

app = Flask(__name__)
CORS(app)

# database.delete_table('evaluations_table')
database.create_table()

UNITS = 512
BATCH_SIZE = 64
path='model_files/'

encodedTokenizer, decodedTokenizer = getChefTokenizer(
  f"{path}tokenizers/encodedTokenizer-vocab.json",
  f"{path}tokenizers/encodedTokenizer-merges.txt",
  f"{path}tokenizers/decodedTokenizer-vocab.json",
  f"{path}tokenizers/decodedTokenizer-merges.txt")

model = Translator(UNITS, encodedTokenizer, decodedTokenizer)

@app.route('/')
def show_docs():
    return render_template('documentation.html')

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

model.compile(optimizer='adam',
              loss=masked_loss,
              metrics=[masked_acc, masked_loss])

checkpoint_dir = f'final_model_{BATCH_SIZE}_{UNITS}'
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

@app.route('/api/prediction', methods=['POST'])  
def get_prediction():
    ingredients = request.json.get('ingredients')
    if not ingredients:
        return jsonify({"error": "Ingredients not provided!"}), 400

    print(ingredients)
    result = model.translate(ingredients, max_length=356, temperature=0.1)
    return jsonify({"prediction": result[0].numpy().decode()})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
