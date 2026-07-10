from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

text = input("Enter a sentence: ")

result = analyzer.polarity_scores(text)

if result["compound"] >= 0.05:
    sentiment = "Positive"
elif result["compound"] <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print("\nText Sentiment Analysis")
print("-----------------------")
print("Input Text :", text)
print("Sentiment  :", sentiment)
print("Scores     :", result)