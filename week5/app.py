from flask import Flask, render_template, request
import os
import uuid

from emotion_engine import analyze_text, combine_emotion
from response_generator import generate_response
from deepface_predict import predict_emotion
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    filename = str(uuid.uuid4()) + ".jpg"

    image_path = os.path.join(
    app.config["UPLOAD_FOLDER"],
    filename
    )

    image.save(image_path)

    face = predict_emotion(image_path)

    text = request.form["text"]

    sentiment = analyze_text(text)

    final_emotion = combine_emotion(face, sentiment)

    response = generate_response(final_emotion)

    return render_template(
        "index.html",
        image=filename,
        face=face,
        sentiment=sentiment,
        final=final_emotion,
        response=response
    )


if __name__ == "__main__":
    app.run(debug=True)