from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def show_docs():
    return render_template('documentation.html')

# @app.route('/post/prediction', methods=['POST'])  
# def post_prediction():
#     ingredients = request.json.get('ingredients')
#     print(ingredients)
#     prediction = predict(ingredients)

#     return jsonify({"prediction": prediction})

# def predict(ingredients):
#     from model_files import tokenizer_vars as tv
#     from model_files import LSTM_class
#     from model_files import constants as c

#     from keras import layers, models, optimizers, callbacks
    
#     model_emb = models.load_model(f'model_files/blstm_model_50000_emb_chef_tokenizer_30')

#     tokenizer = tv.getTokenizer()
#     max_len = 230
#     step = max_len-1
    
#     input_prefix = f"{ingredients}"
#     tokens_ind_prefix = tokenizer.encode(input_prefix).ids

#     pred_emb = LSTM_class.ModelPredict(model_emb, tokens_ind_prefix, input_prefix, max_len, embedding=True)

#     return pred_emb.generate_sequence(temperature=1)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
