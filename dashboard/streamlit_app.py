"""
streamlit_app.py
-----------------
Dashboard for the churn prediction system. Calls the FastAPI backend for
actual predictions (single source of truth); loads the model directly
only for generating SHAP explanations, which the API doesn't expose.

Run with (from the project root, FastAPI server must already be running):
    streamlit run dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Let this script import from app/ when run via `streamlit run dashboard/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_loader import load_default_model
from catboost import Pool

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources -- loaded once, not on every interaction/rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_default_model()


@st.cache_data(ttl=60)
def get_model_info():
    response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()


def risk_color(risk_category: str) -> str:
    return {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_category, "gray")


# ---------------------------------------------------------------------------
# Sidebar: model info
# ---------------------------------------------------------------------------
st.sidebar.title("Model Info")
try:
    info = get_model_info()
    st.sidebar.metric("Features used", info["n_features"])
    st.sidebar.metric("Decision threshold", info["decision_threshold"])
    st.sidebar.write("**Risk bands (upper bound):**")
    st.sidebar.json(info["risk_bands"])
    api_online = True
except requests.exceptions.ConnectionError:
    st.sidebar.error("Cannot reach the API. Is it running at " + API_BASE_URL + "?")
    api_online = False

st.title("Customer Churn Prediction Dashboard")

if not api_online:
    st.error(
        f"The FastAPI backend isn't reachable at {API_BASE_URL}. "
        "Start it first with: python -m uvicorn app.main:app --reload"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_individual, tab_batch = st.tabs(["Individual Prediction", "Batch CSV Prediction"])


# ---------------------------------------------------------------------------
# Tab 1: Individual prediction
# ---------------------------------------------------------------------------
with tab_individual:
    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    with col3:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=1.0)

    if st.button("Predict Churn", type="primary"):
        customer_data = {
            "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=customer_data, timeout=5)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction request failed: {e}")
            st.stop()

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Prediction", result["churn_prediction"])
        res_col2.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        res_col3.markdown(
            f"**Risk Category:** :{risk_color(result['risk_category'])}[{result['risk_category']}]"
        )

        # SHAP explanation for this specific customer -- loaded directly,
        # not through the API (see design note above).
        st.subheader("Why this prediction? (SHAP explanation)")
        model = get_model()
        row_df = pd.DataFrame([customer_data], columns=model.feature_names)
        explainer = shap.TreeExplainer(model.model)
        pool = Pool(row_df, cat_features=model.get_cat_feature_indices())
        shap_values = explainer.shap_values(pool)

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=row_df.iloc[0],
            feature_names=model.feature_names,
        )
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Tab 2: Batch CSV prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV of customers")
    st.caption(
        "Extra columns (customerID, gender, PaperlessBilling, Churn) are fine -- "
        "only the columns the model needs are used. Rows with missing values "
        "are flagged individually rather than failing the whole batch."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and st.button("Run Batch Prediction", type="primary"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_BASE_URL}/predict/batch/csv", files=files, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Batch prediction request failed: {e}")
            st.stop()

        # Parse the returned CSV bytes back into a DataFrame for display
        from io import StringIO
        results_df = pd.read_csv(StringIO(response.text))

        st.success(f"Processed {len(results_df)} customers.")

        n_errors = (results_df["error"] != "").sum() if "error" in results_df else 0
        if n_errors:
            st.warning(f"{n_errors} row(s) had errors and were skipped -- see the 'error' column.")

        st.dataframe(results_df, use_container_width=True)

        # Basic business-oriented visualization: risk category distribution
        valid_results = results_df[results_df["risk_category"] != ""]
        if not valid_results.empty:
            st.subheader("Risk Category Distribution")
            risk_counts = valid_results["risk_category"].value_counts()
            st.bar_chart(risk_counts)

        st.download_button(
            "Download predictions as CSV",
            data=response.content,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )