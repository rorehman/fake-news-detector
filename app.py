import streamlit as st
import joblib
import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return ' '.join(tokens)

# Load models (cached)
@st.cache_resource
def load_models():
    news_model = joblib.load(os.path.join("models", "final_model.pkl"))
    claims_model = joblib.load(os.path.join("models", "claims_model.pkl"))
    return {
        "News Article": news_model,
        "Claim/Statement": claims_model
    }

models = load_models()

# Streamlit UI
st.title("📰 Fake News & Claim Detector")
st.write("Check if a news article or a short factual claim is **real or fake**.")

# Model choice
model_type = st.radio(
    "What are you checking?",
    ["News Article", "Claim/Statement"]
)

user_input = st.text_area("Enter your text below:", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(user_input)
        model = models[model_type]
        prediction = model.predict([cleaned])[0]
        label = "🔴 Fake" if prediction == 0 else "🟢 Real"

        if hasattr(model, "predict_proba"):
            confidence = round(max(model.predict_proba([cleaned])[0]) * 100, 2)
            st.markdown(f"### {label}")
            st.write(f"**Confidence:** {confidence}%")
        else:
            st.markdown(f"### {label}")
