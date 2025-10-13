import streamlit as st
import requests

st.title("📰 Fake News & Claim Detector")
st.write("Enter text to check if it's real or fake.")

user_input = st.text_area("News / Claim Text", height=200)
model_type = st.selectbox("Select model type", ["news", "claims"])

API_URL = "http://127.0.0.1:8000/predict"  # Local FastAPI

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        payload = {"text": user_input, "model_type": model_type}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.markdown(f"### 🔹 Prediction: {result['label']}")
            if result['confidence'] is not None:
                st.write(f"**Confidence:** {result['confidence']}%")
        else:
            st.error("Error connecting to the API.")
