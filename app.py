import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="AI Scam & Phishing Detector", page_icon="🛡️", layout="centered"
)


# 2. Train the Model (Cached so it only trains once when the app loads)
@st.cache_resource
def load_trained_model():
  data = {
      "text": [
          (
              "URGENT: Your bank account has been suspended. Click"
              " http://bit.ly/login to verify."
          ),
          (
              "Congratulations! You won a $1000 gift card. Claim it now at"
              " http://scam-site.com"
          ),
          (
              "Immediate action required! Verify your crypto wallet security"
              " keys at http://192.168.1.1"
          ),
          (
              "Account alert: unauthorized transaction detected. Secure your"
              " account immediately."
          ),
          (
              "Free entry in 2 a wkly comp to win FA Cup final tickets 21st"
              " May 2005."
          ),
          "Hey, are we still meeting for lunch tomorrow at noon?",
          (
              "Your package shipment update: tracking number #12345 is out for"
              " delivery."
          ),
          "Hey Mom, just letting you know I landed safely. Call you later!",
          "Can you pick up some milk and eggs from the grocery store?",
          "Hey! Are we still on for movie night this weekend?",
          "Your doctor appointment is confirmed for Tuesday at 2:30 PM.",
      ],
      "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 1 = Scam, 0 = Safe
  }
  df = pd.DataFrame(data)

  pipeline = Pipeline([
      ("tfidf", TfidfVectorizer(stop_words="english")),
      ("classifier", LogisticRegression()),
  ])
  pipeline.fit(df["text"], df["label"])
  return pipeline


model = load_trained_model()

# 3. Web App UI Design
st.title("🛡️ AI Scam & Phishing Detector")
st.write(
    "Paste any suspicious text message, email, or link below. Our machine"
    " learning model will analyze it for phishing patterns in real time!"
)

# Text input box for user message
user_message = st.text_area(
    "Enter message to analyze:",
    placeholder=(
        "Type or paste a suspicious message here (e.g., URGENT: Click link to"
        " verify bank...)"
    ),
)

if st.button("Scan Message", type="primary"):
  if user_message.strip() == "":
    st.warning("Please enter a message to scan first.")
  else:
    # Run prediction
    probability = model.predict_proba([user_message])[0][1]

    st.subheader("Analysis Result:")

    # Display results visually based on probability
    if probability > 0.5:
      st.error(
          f"🚨 **WARNING: Likely Scam / Phishing** (Risk Score:"
          f" {probability:.2%})"
      )
      st.markdown(
          "*This message contains phrasing commonly associated with phishing"
          " attacks or financial scams. Do not click any links or share"
          " personal information.*"
      )
    else:
      st.success(f"✅ **Looks Safe** (Risk Score: {probability:.2%})")
      st.markdown(
          "*This message matches patterns typical of safe, normal everyday"
          " communication.*"
      )