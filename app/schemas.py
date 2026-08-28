"""
schemas.py
----------
Pydantic models defining exactly what the API expects as input and
returns as output. Field names/types here must match the trained model's
feature set (models/model_metadata.json) exactly.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """One customer's data, matching the 17 features the model was trained on."""

    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, description="Months as a customer")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
                "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5, "TotalCharges": 477.5,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_category: str
    recommendation: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    feature_names: List[str]
    categorical_columns: List[str]
    target_classes: List[str]
    n_features: int
    decision_threshold: float
    risk_bands: dict