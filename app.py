from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Welcome to Iris Flower Prediction API"

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]

    prediction = model.predict(features)

    flower = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return jsonify({
        "prediction": int(prediction[0]),
        "flower_name": flower[int(prediction[0])]
    })

if __name__ == "__main__":
    app.run(debug=True)