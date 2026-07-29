from deepface import DeepFace

def predict_emotion(image_path):
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        mapping = {
            "happy": "Happy",
            "sad": "Sad",
            "angry": "Angry",
            "fear": "Fear",
            "surprise": "Surprise",
            "neutral": "Neutral",
            "disgust": "Disgust"
        }

        return mapping.get(emotion.lower(), "Neutral")

    except Exception as e:
        print(e)
        return "Neutral"