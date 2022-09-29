from flask import Flask, message_flashed, render_template, request, jsonify
from main import get_response


app = Flask(__name__)


@app.get("/")
def get_index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")  # set this up in js
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
