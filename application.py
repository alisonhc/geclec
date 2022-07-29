import os
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import inference
import config

application = app = Flask(__name__)

dirname = os.path.dirname(__file__)

geclec_t5_path = os.path.join(dirname, 'models', config.T5_GEC_LEC)
geclec_t5_tok = AutoTokenizer.from_pretrained(geclec_t5_path)
geclec_t5_model = AutoModelForSeq2SeqLM.from_pretrained(geclec_t5_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST'])
def predict():
    # in the future, have multiple models as choices
    content = request.json
    sent = content['sentence'].strip()
    gen_sent = inference.correct_sent(model=geclec_t5_model, tok=geclec_t5_tok, sent=sent)
    response = {'input_sent': sent, 'output_sent': gen_sent}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
