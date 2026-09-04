
import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)



VALID_CUSTOMER = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 477.5,
}


VALID_CUSTOMER_2 = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 60.0,
    "TotalCharges": 3600.0,
}


# ---------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------

def test_health_check_returns_200():
    """/health should respond and report the model as loaded.

    This doubles as our "model loading" test: /health can only report
    model_loaded=True if model_loader.py successfully loaded the .cbm
    file and metadata at app startup.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ---------------------------------------------------------------------
# 2. Valid single prediction
# ---------------------------------------------------------------------

def test_predict_valid_customer():
    """A well-formed request should return a complete, sensible prediction."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    data = response.json()

    # All fields from PredictionResponse must be present
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_category" in data
    assert "recommendation" in data

    # Exact value checks now confirmed from predictor.py
    assert data["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_category"] in ("Low", "Medium", "High")
    assert isinstance(data["recommendation"], str) and len(data["recommendation"]) > 0


# ---------------------------------------------------------------------
# 3. Invalid input handling
# ---------------------------------------------------------------------

def test_predict_missing_required_field_returns_422():
    """Omitting a required field (tenure) should fail validation (422),
    not crash the server (500)."""
    bad_customer = copy.deepcopy(VALID_CUSTOMER)
    del bad_customer["tenure"]

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    """Sending a string where a number is expected (MonthlyCharges)
    should also fail validation cleanly."""
    bad_customer = copy.deepcopy(VALID_CUSTOMER)
    bad_customer["MonthlyCharges"] = "not-a-number"

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_out_of_range_senior_citizen_returns_422():
    """SeniorCitizen is constrained to 0 or 1 (ge=0, le=1) in the schema;
    a value outside that range should fail validation."""
    bad_customer = copy.deepcopy(VALID_CUSTOMER)
    bad_customer["SeniorCitizen"] = 5

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_extra_field_ignored_by_schema():
    """Pydantic's default behaviour is to ignore fields not declared on
    the model (CustomerInput), so an extra field like 'customerID' should
    NOT cause a 422 -- it's silently dropped before reaching predictor.py's
    own stricter _validate_input check.
    """
    customer_with_extra = copy.deepcopy(VALID_CUSTOMER)
    customer_with_extra["customerID"] = "cust_999"

    response = client.post("/predict", json=customer_with_extra)
    assert response.status_code == 200


# ---------------------------------------------------------------------
# 4. Batch prediction
# ---------------------------------------------------------------------

def test_predict_batch_valid():
    """Sending two customers should return two predictions, in order."""
    payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER_2]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        assert "churn_prediction" in pred
        assert "churn_probability" in pred
        assert "risk_category" in pred
        assert 0.0 <= pred["churn_probability"] <= 1.0


def test_predict_batch_empty_list():
    """An empty customer list should return an empty prediction list,
    not an error -- this is an edge case worth locking in behaviour for."""
    payload = {"customers": []}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"] == []


# ---------------------------------------------------------------------
# 5. Model info endpoint (bonus -- quick smoke test since it's cheap)
# ---------------------------------------------------------------------

def test_model_info_returns_expected_shape():
    response = client.get("/model-info")
    assert response.status_code == 200

    data = response.json()
    assert data["n_features"] == 17
    assert len(data["feature_names"]) == 17
    assert isinstance(data["decision_threshold"], float)
    assert "risk_bands" in data


# ---------------------------------------------------------------------
# 6. CSV batch prediction (/predict/batch/csv)
# ---------------------------------------------------------------------

def _build_valid_csv_bytes():
    """Builds an in-memory CSV (as bytes) with the 17 required feature
    columns plus customerID, matching what /predict/batch/csv expects."""
    header = list(VALID_CUSTOMER.keys()) + ["customerID"]
    row1 = list(VALID_CUSTOMER.values()) + ["cust_001"]
    row2 = list(VALID_CUSTOMER_2.values()) + ["cust_002"]

    lines = [",".join(str(x) for x in header)]
    lines.append(",".join(str(x) for x in row1))
    lines.append(",".join(str(x) for x in row2))
    csv_text = "\n".join(lines)
    return csv_text.encode("utf-8")


def test_predict_batch_csv_valid():
    """A well-formed CSV upload should return a downloadable CSV with
    prediction columns appended, one row per input customer."""
    csv_bytes = _build_valid_csv_bytes()

    response = client.post(
        "/predict/batch/csv",
        files={"file": ("customers.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    # Parse the returned CSV body back into rows to check shape/content
    body_text = response.content.decode("utf-8")
    lines = [line for line in body_text.strip().splitlines() if line]
    header = lines[0].split(",")

    assert "churn_prediction" in header
    assert "churn_probability" in header
    assert "risk_category" in header
    assert "error" in header
    # header + 2 data rows
    assert len(lines) == 3


def test_predict_batch_csv_rejects_non_csv_file():
    """Uploading a non-.csv file should return 400, not crash."""
    response = client.post(
        "/predict/batch/csv",
        files={"file": ("customers.txt", b"not,a,real,csv", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_batch_csv_missing_columns():
    """A CSV missing required feature columns should return 400 with a
    clear message, per the missing_columns check in main.py."""
    csv_bytes = b"tenure,MonthlyCharges\n5,95.5\n"

    response = client.post(
        "/predict/batch/csv",
        files={"file": ("incomplete.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 400