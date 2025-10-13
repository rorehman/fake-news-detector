# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from api.utils import preprocess_text

app = FastAPI(title="Fake News & Claim Detection API")

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
news_model = joblib.load(os.path.join(MODEL_DIR, 'final_model.pkl'))
claims_model = joblib.load(os.path.join(MODEL_DIR, 'claims_model.pkl'))

class NewsRequest(BaseModel):
    text: str
    model_type: str = "news"  # 'news' or 'claims'

@app.post("/predict")
def predict(request: NewsRequest):
    cleaned_text = preprocess_text(request.text)
    
    if request.model_type == "news":
        model = news_model
    elif request.model_type == "claims":
        model = claims_model
    else:
        return {"error": "Invalid model_type. Choose 'news' or 'claims'."}

    prediction = model.predict([cleaned_text])[0]
    label = "Real" if prediction == 1 else "Fake"

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned_text])[0]
        confidence = round(max(proba) * 100, 2)

    return {"label": label, "confidence": confidence}
