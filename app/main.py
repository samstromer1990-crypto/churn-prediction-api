"""
main.py
-------
FastAPI backend serving the trained CatBoost churn model.

Run with:
    python -m uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

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
        recommendation=result.recommendation,
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
                recommendation=result.recommendation,
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


@app.post("/predict/batch/csv")
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Accept a CSV of multiple customers, run predictions on all of them, and
    return a downloadable CSV with churn_prediction, churn_probability,
    risk_category, and recommendation columns appended.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv file")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    missing_columns = set(model.feature_names) - set(df.columns)
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {sorted(missing_columns)}",
        )

    has_customer_id = "customerID" in df.columns

    results = []
    for idx, row in df.iterrows():
        customer_id = row["customerID"] if has_customer_id else f"row_{idx}"

        try:
            customer_data = {col: row[col] for col in model.feature_names}

            for col, value in customer_data.items():
                if pd.isna(value):
                    raise ValueError(f"Missing value in column '{col}'")

            result = predictor.predict(customer_data)
            results.append({
                "customerID": customer_id,
                "churn_prediction": result.churn_prediction,
                "churn_probability": result.churn_probability,
                "risk_category": result.risk_category,
                "recommendation": result.recommendation,
                "error": "",
            })
        except Exception as e:
            results.append({
                "customerID": customer_id,
                "churn_prediction": "",
                "churn_probability": "",
                "risk_category": "",
                "recommendation": "",
                "error": str(e),
            })

    output_df = pd.DataFrame(results)
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )