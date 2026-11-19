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
    age: int = Field(..., ge=29, le=77)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=94, le=200)
    chol: int = Field(..., ge=126, le=564)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=71, le=202)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=6.2)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("public/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame(
        [[getattr(data, col) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )
    prediction = int(model.predict(input_data)[0])
    response = {
        "prediction": prediction,
        "result": "Positive class" if prediction == 1 else "Negative class",
        "disclaimer": "Educational demonstration only; not a medical diagnosis."
    }
    if hasattr(model, "predict_proba"):
        response["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(model.classes_, model.predict_proba(input_data)[0])
        }
    return response
