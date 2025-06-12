from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict
from transformers import RobertaForSequenceClassification, RobertaTokenizer

app = Flask(__name__)
CORS(app)

model = RobertaForSequenceClassification.from_pretrained("mattyh867/fake-news-detector")
tokenizer = RobertaTokenizer.from_pretrained("mattyh867/fake-news-detector")

@app.route('/predict', methods=['POST'])
def classify():
  data = request.get_json()
  text = data.get('text', '')
  result = predict(text)
  return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)