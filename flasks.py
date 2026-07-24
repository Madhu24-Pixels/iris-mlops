from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    experience = data['experience']

    salary = experience * 10000

    return jsonify({
        "Predicted Salary": salary
    })

app.run(debug=True)