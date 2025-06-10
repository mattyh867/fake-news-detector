from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def classify():
  data = request.get_json()
  text = data.get('text', '')
  result = predict(text)
  return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)