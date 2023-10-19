from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from keras import layers, models, optimizers, callbacks

from model_files import tokenizer_vars as tv
from model_files import LSTM_class
from model_files import constants as c

app = Flask(__name__)
CORS(app)

model_emb = models.load_model('model/')
tokenizer = tv.getTokenizer()
max_len = 230

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

if __name__ == "__main__":
    app.run(debug=True, port=8080)
