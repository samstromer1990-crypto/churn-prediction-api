"""
main.py
-------
FastAPI backend serving the trained CatBoost churn model.

Run with:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

from fastapi import FastAPI, HTTPException

from app.model_loader import load_default_model
from app.predictor import ChurnPredictor, CHURN_DECISION_THRESHOLD, RISK_BANDS
from app.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serves predictions from a trained CatBoost churn model.",
    version="1.0.0",
)

# Loaded once at startup, not per-request -- reloading the model on every
# request would be needlessly slow and is not how production services work.
model = load_default_model()
predictor = ChurnPredictor(model)


@app.get("/health", response_model=HealthResponse)
def health():
    """Basic liveness check -- confirms the API is up and the model loaded successfully."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    try:
        result = predictor.predict(customer.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(
        churn_prediction=result.churn_prediction,
        churn_probability=result.churn_probability,
        risk_category=result.risk_category,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Predict churn for a list of customers in one request."""
    predictions = []
    for i, customer in enumerate(request.customers):
        try:
            result = predictor.predict(customer.model_dump())
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Customer at index {i}: {e}"
            )
        predictions.append(
            PredictionResponse(
                churn_prediction=result.churn_prediction,
                churn_probability=result.churn_probability,
                risk_category=result.risk_category,
            )
        )

    return BatchPredictionResponse(predictions=predictions)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Metadata about the loaded model -- useful for debugging and for the dashboard."""
    return ModelInfoResponse(
        feature_names=model.feature_names,
        categorical_columns=model.categorical_columns,
        target_classes=model.target_classes,
        n_features=len(model.feature_names),
        decision_threshold=CHURN_DECISION_THRESHOLD,
        risk_bands={label: upper for upper, label in RISK_BANDS},
    )
