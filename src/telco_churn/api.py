import sys
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telco_churn.serialization import ModelSerializer
from telco_churn.preprocessing import DataPreprocessor

app = FastAPI(
    title="Telco Customer Churn & Segmentation API",
    description="RESTful microservice for real-time customer churn probability scoring and batch prediction.",
    version="1.0.0",
)

# Global serializer load
serializer = ModelSerializer()
model, preprocessor = serializer.load()


class CustomerPayload(BaseModel):
    Gender: str = Field(..., example="Female")
    Tenure_Months: int = Field(..., alias="Tenure Months", example=12)
    Monthly_Charges: float = Field(..., alias="Monthly Charges", example=65.5)
    Total_Charges: float = Field(..., alias="Total Charges", example=786.0)
    Contract: str = Field(..., example="Month-to-month")


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(payload: CustomerPayload):
    global model, preprocessor
    if model is None or preprocessor is None:
        serializer_inst = ModelSerializer()
        model, preprocessor = serializer_inst.load()
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact not found. Run CLI pipeline to train & save model.")

    # Format input into DataFrame
    df_input = pd.DataFrame([{
        "Gender": payload.Gender,
        "Tenure Months": payload.Tenure_Months,
        "Monthly Charges": payload.Monthly_Charges,
        "Total Charges": payload.Total_Charges,
        "Contract": payload.Contract,
    }])

    # Clean & Encode
    df_clean = preprocessor.clean_data(df_input)
    _, X, _, X_scaled = preprocessor.encode_and_scale(df_clean)

    # Predict
    prob = float(model.predict_proba(X_scaled if hasattr(model, "fit_predict") else X)[:, 1][0]) if hasattr(model, "predict_proba") else 0.5
    pred = int(prob >= 0.5)

    risk_level = "High Risk" if prob >= 0.65 else ("Medium Risk" if prob >= 0.35 else "Low Risk")

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk_level,
    )


@app.post("/predict-batch")
def predict_batch(payloads: List[CustomerPayload]):
    results = []
    for p in payloads:
        res = predict_churn(p)
        results.append(res.dict())
    return {"batch_predictions": results, "total_processed": len(results)}
