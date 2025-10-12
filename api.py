# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import spacy
import os

# Initialize FastAPI app
app = FastAPI(title="Fake News & Claim Verification API")

# Load spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# Load models
model_paths = {
    "news": "models/final_model.pkl",
    "claims": "models/claims_model.pkl"
}

models = {}
for key, path in model_paths.items():
    if os.path.exists(path):
        models[key] = joblib.load(path)
    else:
        raise FileNotFoundError(f"Model not found: {path}")

# Define input schema
class TextInput(BaseModel):
    text: str
    model_type: str  # 'news' or 'claims'

@app.post("/predict")
def predict(input_data: TextInput):
    if input_data.model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'news' or 'claims'.")
    
    cleaned_text = preprocess_text(input_data.text)
    model = models[input_data.model_type]
    prediction = model.predict([cleaned_text])[0]

    # Optional: return confidence if available
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned_text])[0]
        confidence = round(max(proba) * 100, 2)

    return {
        "model_used": input_data.model_type,
        "prediction": int(prediction),
        "label": "Real" if prediction == 1 else "Fake",
        "confidence": confidence
    }
