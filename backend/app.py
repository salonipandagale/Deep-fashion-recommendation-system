from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
from fashion_utils import feature_extraction, recommend

app = Flask(__name__)

# Load data
features = pickle.load(open('features.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "API Running 🚀"

@app.route("/recommend", methods=["POST"])
def recommend_api():
    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    input_features = feature_extraction(filepath)
    indices = recommend(input_features, features)

    results = [filenames[i] for i in indices]

    return jsonify({"results": results})

@app.route("/images/<path:filename>")
def get_image(filename):
    return send_from_directory(".", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
