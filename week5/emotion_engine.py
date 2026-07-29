from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_text(text):

    score = analyzer.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"


def combine_emotion(face_emotion, text_sentiment):

    if face_emotion == "Happy" and text_sentiment == "Positive":
        return "Happy"

    elif face_emotion == "Sad":
        return "Sad"

    elif face_emotion == "Angry":
        return "Angry"

    elif face_emotion == "Neutral" and text_sentiment == "Neutral":
        return "Neutral"

    else:
        return "Mixed"