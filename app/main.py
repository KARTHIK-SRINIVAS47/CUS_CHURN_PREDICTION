from fastapi import FastAPI
import pandas as pd

from app.model_loader import pipeline
from app.schemas import CustomerData

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts whether a customer will churn"
)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict churn (0/1) and probability
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return {
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 3)
    }
