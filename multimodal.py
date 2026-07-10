from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

print("========== Multimodal Emotion Recognition ==========")

image_emotion = input(
    "Enter facial emotion (Happy/Sad/Angry/Neutral): "
)

text = input("Enter text: ")

result = analyzer.polarity_scores(text)

if result["compound"] >= 0.05:
    text_sentiment = "Positive"
elif result["compound"] <= -0.05:
    text_sentiment = "Negative"
else:
    text_sentiment = "Neutral"

print("\nImage Emotion :", image_emotion)
print("Text Sentiment:", text_sentiment)

if image_emotion.lower() == "happy" and text_sentiment == "Positive":
    final_emotion = "Happy"
elif image_emotion.lower() == "sad" and text_sentiment == "Negative":
    final_emotion = "Sad"
elif image_emotion.lower() == "angry" and text_sentiment == "Negative":
    final_emotion = "Angry"
else:
    final_emotion = "Neutral"

print("\nFinal Multimodal Prediction:", final_emotion)