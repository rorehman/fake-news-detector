import streamlit as st
import requests

# Streamlit UI
st.title("📰 Fake News & Claim Detector")
st.write("Enter text below to check if it's **real or fake**.")

# User input
user_input = st.text_area("Text", height=200)

# Model selection
model_type = st.radio(
    "Choose Model",
    ("news", "claims"),
    index=0,
    help="Use 'news' for news articles, 'claims' for statements/facts."
)

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Call FastAPI backend
        url = "http://127.0.0.1:8000/predict"
        payload = {"text": user_input, "model_type": model_type}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            st.markdown(f"### Model used: {data['model_used']}")
            st.markdown(f"### Prediction: {data['label']}")
            st.write(f"**Confidence:** {data['confidence']}%")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
