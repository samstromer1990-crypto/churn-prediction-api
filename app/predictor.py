"""
predictor.py
------------
Turns raw customer data into a usable prediction: churn Yes/No, the raw
probability, and a business-friendly risk category.

Threshold and risk-band values are explicit ASSUMPTIONS, not derived from
business data -- flagged clearly below and meant to be revisited once real
business input (e.g. retention offer cost) is available.
"""

from dataclasses import dataclass

import pandas as pd

from app.model_loader import ChurnModel
from app.recommendations import generate_recommendation


CHURN_DECISION_THRESHOLD = 0.5  # matches the notebook's Phase 11 evaluation

RISK_BANDS = [
    (0.30, "Low"),
    (0.60, "Medium"),
    (1.01, "High"),
]


@dataclass
class PredictionResult:
    churn_prediction: str
    churn_probability: float
    risk_category: str
    recommendation: str


def _categorize_risk(probability: float) -> str:
    for upper_bound, label in RISK_BANDS:
        if probability < upper_bound:
            return label
    return "High"


class ChurnPredictor:
    def __init__(self, model: ChurnModel):
        self.model = model

    def _validate_input(self, customer_data: dict):
        expected = set(self.model.feature_names)
        provided = set(customer_data.keys())

        missing = expected - provided
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        extra = provided - expected
        if extra:
            raise ValueError(
                f"Unexpected fields not used by the model: {sorted(extra)}. "
                f"Check for typos against the expected feature names."
            )

    def predict(self, customer_data: dict) -> PredictionResult:
        self._validate_input(customer_data)

        row = pd.DataFrame([customer_data], columns=self.model.feature_names)

        probability = self.model.model.predict_proba(row)[0][1]
        prediction_label = "Yes" if probability >= CHURN_DECISION_THRESHOLD else "No"
        risk = _categorize_risk(probability)
        recommendation = generate_recommendation(customer_data, risk)

        return PredictionResult(
            churn_prediction=prediction_label,
            churn_probability=round(float(probability), 4),
            risk_category=risk,
            recommendation=recommendation,
        )