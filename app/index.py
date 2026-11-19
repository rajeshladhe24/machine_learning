import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational prediction API using a trained Logistic Regression pipeline.",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([[getattr(data, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(input_data)[0])
    result = "Positive class" if prediction == 1 else "Negative class"
    response = {"prediction": prediction, "result": result}
    if hasattr(model, "predict_proba"):
        response["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(model.classes_, model.predict_proba(input_data)[0])
        }
    response["disclaimer"] = "Educational demonstration only; not a medical diagnosis."
    return response
