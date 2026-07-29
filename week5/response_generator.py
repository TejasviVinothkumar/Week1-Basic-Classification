responses = {
    "Happy": """
😊 That's wonderful to hear! Keep smiling and continue spreading positivity.
Your cheerful mindset will help you achieve great things today.
""",

    "Sad": """
💙 I'm sorry you're feeling down.
Remember that every difficult day eventually comes to an end.
Take a break, talk to someone you trust, and be kind to yourself.
""",

    "Angry": """
😌 Take a deep breath.
Try to stay calm before reacting.
A peaceful mind often finds the best solution.
""",

    "Neutral": """
🙂 You seem calm and balanced.
Keep maintaining this positive mindset and stay focused.
""",

    "Mixed": """
🤝 Your emotions seem mixed.
It's completely normal to experience different feelings at once.
Take things one step at a time.
"""
}

def generate_response(emotion):
    return responses.get(
        emotion,
        "Stay positive and take good care of yourself."
    )