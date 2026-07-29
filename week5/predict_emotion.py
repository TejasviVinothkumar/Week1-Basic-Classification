from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# -----------------------------
# Load Trained CNN Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "week3" / "emotion_model.keras"

print("Loading model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Emotion Labels
# -----------------------------
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# -----------------------------
# Load Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def predict_emotion(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return "Face Not Detected"

    # Take the biggest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (48, 48))

    face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)

    emotion = emotion_labels[np.argmax(prediction)]

    return emotion