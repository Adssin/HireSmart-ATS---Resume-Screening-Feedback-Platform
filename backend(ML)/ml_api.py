from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load REGRESSION model (NOT classifier)
model = joblib.load("general_ats_regressor.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class ResumeRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: ResumeRequest):
    X = vectorizer.transform([req.text])
    score = model.predict(X)[0]

    # Clamp score to 0–100
    score = max(0, min(100, float(score)))

    # Optional label for UI
    if score >= 70:
        label = "Good"
    elif score >= 40:
        label = "Average"
    else:
        label = "Poor"

    return {
        "score": round(score, 2),
        "label": label
    }