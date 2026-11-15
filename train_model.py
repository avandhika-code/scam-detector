import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Expanded, balanced training dataset
data = {
    "text": [
        "URGENT: Your bank account has been suspended. Click http://bit.ly/login to verify.",
        "Congratulations! You won a $1000 gift card. Claim it now at http://scam-site.com",
        "Immediate action required! Verify your crypto wallet security keys at http://192.168.1.1",
        "Account alert: unauthorized transaction detected. Secure your account immediately.",
        "Free entry in 2 a wkly comp to win FA Cup final tickets 21st May 2005.",
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "Your package shipment update: tracking number #12345 is out for delivery.",
        "Hey Mom, just letting you know I landed safely. Call you later!",
        "Can you pick up some milk and eggs from the grocery store?",
        "Hey! Are we still on for movie night this weekend?",
        "Your doctor appointment is confirmed for Tuesday at 2:30 PM."
    ],
    "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1 = Scam, 0 = Safe
}

df = pd.DataFrame(data)

# 2. Build and Train Model Pipeline
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression())
])

print("Training the scam detection model with expanded data...")
model_pipeline.fit(df["text"], df["label"])
print("Model training complete!\n")

# 3. Interactive Loop
print("=== SCAM DETECTOR READY ===")
print("Type any text message or email to scan it. Type 'exit' to quit.\n")

while True:
    user_input = input("Enter message to test: ")
    if user_input.lower() == 'exit':
        break
    
    probability = model_pipeline.predict_proba([user_input])[0][1]
    prediction = "SCAM / PHISHING" if probability > 0.5 else "SAFE"
    
    print(f"-> Result: {prediction} (Scam Probability: {probability:.2%})\n")