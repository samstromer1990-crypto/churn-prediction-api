from app.model_loader import load_default_model
from app.predictor import ChurnPredictor

model = load_default_model()
predictor = ChurnPredictor(model)

sample_customer = {
    'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No', 'tenure': 5,
    'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.5, 'TotalCharges': 477.5
}

result = predictor.predict(sample_customer)
print(result)