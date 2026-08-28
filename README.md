# Customer Churn Prediction System

> **Status: Work in progress.** This README reflects what has been built and tested so far. Stages not yet complete (automated tests, Docker, live deployment) are listed explicitly at the bottom — they are not overstated as done.

## Business Problem

Customer churn — a customer ending their relationship with the company — is expensive to fix after the fact but often preventable if flagged early. This project predicts which customers are at risk of churning so that retention efforts (offers, outreach, service follow-ups) can be targeted at the customers who actually need them, rather than applied blanket-wide.

## Dataset

**IBM Telco Customer Churn** dataset (via Kaggle) — 7,043 customers, 21 columns covering demographics, account details (tenure, contract type, payment method), subscribed services, monthly/total charges, and whether the customer churned.

This is a well-known, publicly available dataset chosen deliberately as a stable learning dataset rather than for its novelty. The goal of this project was depth of execution — correct methodology, explainability, and a working deployment — rather than a "novel" dataset.

## ML Methodology

**Data cleaning:**
- `customerID` dropped (identifier, no predictive value).
- `TotalCharges` was loading as text due to 11 rows containing a blank string instead of a number. These 11 rows all correspond to customers with `tenure == 0` (brand-new customers not yet billed) — the blank was corrected to `0`, since that is the customer's true total charge so far, not a value to be guessed at.
- `gender` and `PaperlessBilling` were dropped from the feature set early in the project, before EDA confirmed whether they mattered. This is flagged here explicitly as a process gap rather than a validated decision — see **Limitations** below.

**Preprocessing:**
- Train/test split performed *before* any encoding, to avoid any risk of information leaking from the test set into preprocessing.
- Categorical columns were one-hot encoded (`handle_unknown='ignore'`) for models that need it. The final model, CatBoost, instead consumes raw categorical columns directly — see below.

**Model comparison:** Nine classifiers were compared via 5-fold stratified cross-validation (scored on ROC-AUC, F1, Precision, Recall, since churn is imbalanced ~73%/27% and accuracy alone would be misleading):

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **CatBoost** | **0.849** | **0.634** | 0.528 | **0.793** |
| Gradient Boosting | 0.848 | 0.588 | 0.663 | 0.529 |
| Logistic Regression | 0.846 | 0.628 | 0.517 | 0.801 |
| LightGBM | 0.827 | 0.598 | 0.563 | 0.638 |
| SVM (RBF) | 0.827 | 0.624 | 0.520 | 0.781 |
| K-Nearest Neighbors | 0.822 | 0.578 | 0.596 | 0.562 |
| Random Forest | 0.822 | 0.542 | 0.635 | 0.474 |
| Naive Bayes | 0.821 | 0.597 | 0.461 | 0.848 |
| XGBoost | 0.815 | 0.568 | 0.568 | 0.568 |

**Final model: CatBoost**, chosen for:
- Native handling of categorical features (no one-hot encoding needed, avoiding the associated dimensionality blowup and letting CatBoost's own ordered target-statistics encoding do the work).
- Best combination of ROC-AUC and, more importantly, **Recall** — for this business problem, missing an actual churner (false negative) is typically more costly than a false alarm, making Recall a meaningful metric to weight alongside ROC-AUC and F1.

**Hyperparameter tuning:** `RandomizedSearchCV` (15 combinations × 3-fold CV) over `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, optimizing for F1. Result: **tuning barely improved on CatBoost's defaults** (F1 0.634 → 0.637, ROC-AUC 0.849 → 0.848 — within noise). This is reported here as a legitimate finding, not a failed experiment: it indicates CatBoost's defaults were already close to optimal for this dataset's size and structure.

**Final test-set evaluation** (held out, touched only once):
- ROC-AUC: 0.843
- Recall (Churn class): 0.79 — catches ~294 of 374 actual churners
- Precision (Churn class): 0.53 — roughly half of flagged customers are false alarms
- These numbers closely match the cross-validation estimates, indicating the model is not overfit.

**SHAP explainability:** Global feature importance ranking: `Contract` > `InternetService` > `tenure` > `OnlineSecurity` > `TotalCharges` > `PaymentMethod` > `TechSupport` > `MonthlyCharges`. Longer tenure and longer-term contracts push predictions away from churn; Fiber optic internet and high monthly charges push toward churn — consistent with general churn intuition.

**Error analysis:** the model's main blind spot is **churners who "look loyal"** — the false negatives (missed churners) have on average 2.5x longer tenure (30.7 vs 12.6 months) and a much more mixed contract-type distribution (51% on annual/two-year contracts, vs. 1.4% for correctly-caught churners) than the churners it catches easily. This suggests the model has learned "long tenure + committed contract = safe" as a strong heuristic, and lacks the features (e.g. recent support tickets, satisfaction data, competitor offers) that would explain "surprise" churn among otherwise stable-looking customers.

## System Architecture

```
Raw customer data
       │
       ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  FastAPI backend │◄─────┤ Streamlit         │      │  Saved model         │
│  (app/main.py)   │      │ dashboard         │      │  (models/*.cbm,      │
│                  │─────►│ (dashboard/       │      │   *.json)            │
│  loads model once│      │  streamlit_app.py)│      │  loaded at startup    │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
```

- The **FastAPI backend** is the single source of truth for predictions — it loads the trained CatBoost model once at startup and serves all prediction logic.
- The **Streamlit dashboard** calls the FastAPI backend over HTTP for actual predictions (rather than duplicating prediction logic), so the two can never drift out of sync. It loads the model directly only to compute live SHAP explanations for the dashboard, which the API does not currently expose.
- **CatBoost consumes raw categorical columns directly** — there is no separate encoder object to keep synchronized between training and serving, which removes a common class of train/serve inconsistency bugs.

## Project Structure

```
goal intern project_1/
├── app/
│   ├── model_loader.py      # Loads model + metadata, validates consistency
│   ├── predictor.py          # Prediction logic: probability, Yes/No, risk band
│   ├── recommendations.py    # Rule-based business recommendations (see note below)
│   ├── schemas.py             # Pydantic request/response models
│   └── main.py                 # FastAPI app and all endpoints
├── dashboard/
│   └── streamlit_app.py       # Individual + batch prediction UI, SHAP explanations
├── models/
│   ├── catboost_churn_model.cbm
│   └── model_metadata.json    # feature names, categorical columns, target classes
├── churn_notebook_fixed.ipynb # Full ML workflow: cleaning through error analysis
├── model_selection.py         # Standalone model comparison script
├── model_catboost.py          # Standalone CatBoost-only comparison script
├── tune_catboost.py            # Standalone hyperparameter tuning script
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## API Documentation

Interactive docs available at `http://127.0.0.1:8000/docs` once the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Confirms the API is running and the model loaded successfully. |
| `/predict` | POST | Predicts churn for a single customer. Returns prediction, probability, risk category, and a recommendation. |
| `/predict/batch` | POST | Predicts churn for a JSON list of customers in one request. |
| `/predict/batch/csv` | POST | Accepts an uploaded CSV of customers, returns a downloadable CSV with predictions appended. Rows with missing/invalid values are flagged individually rather than failing the whole batch. |
| `/model-info` | GET | Returns the model's feature names, categorical columns, target classes, decision threshold, and risk band definitions. |

## Business Recommendations — Explicit Assumption

The system attaches a short recommendation to each prediction (e.g. "recommend a retention offer"). **These are heuristics, not validated business rules.** No real cost/ROI data on retention offers or outreach campaigns was available for this project. The rules are based on:
- Risk category (Low/Medium/High) as a baseline
- Whether the customer is on a month-to-month contract or has Fiber optic internet — the two strongest churn drivers identified via SHAP

Similarly, the **0.5 decision threshold** and the **Low (<30%) / Medium (30–60%) / High (>60%) risk bands** are reasonable defaults, not derived from an actual cost-of-false-negative vs. cost-of-false-positive analysis. Both should be revisited if real business figures become available.

## How to Run Locally

Requires two terminals running at the same time.

**Terminal 1 — start the API:**
```bash
cd "goal intern project_1"
python -m uvicorn app.main:app --reload
```
Wait for `Application startup complete.`

**Terminal 2 — start the dashboard:**
```bash
cd "goal intern project_1"
streamlit run dashboard/streamlit_app.py
```

Then visit:
- API docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://localhost:8501`

**Note (Windows):** always use `python -m uvicorn ...` rather than a bare `uvicorn` command — on machines with multiple Python installations, the bare command can silently resolve to the wrong Python environment.

## Limitations

- `gender` and `PaperlessBilling` were dropped from the feature set before EDA confirmed whether they had predictive value — a process gap, not a validated decision. `gender` is very likely safe to drop; `PaperlessBilling` was not checked before being dropped and may have mattered.
- The model's recall (0.79) means roughly 1 in 5 actual churners are missed, concentrated specifically among customers who "look loyal" (long tenure, longer-term contracts) — see **Error Analysis** above.
- Precision (0.53) means close to half of customers flagged as at-risk are false alarms; the current 0.5 decision threshold has not been tuned against real retention-offer cost data.
- The dataset lacks features that might explain "surprise" churn (support ticket history, satisfaction surveys, competitor pricing), which likely explains the model's specific blind spot.
- Business recommendation rules and risk-band thresholds are heuristic placeholders, not derived from real business cost data (see above).

## Future Improvements / Remaining Work

The following stages of the project roadmap are **not yet complete**:
- **Automated tests** (API health, valid/invalid prediction, batch prediction, model loading)
- **Dockerization** (Dockerfile, requirements.txt, containerized run instructions)
- **Deployment** to a live, affordable hosting platform, with an actual deployment link
- Revisiting the `gender` / `PaperlessBilling` drop decision with proper EDA
- Investigating additional features that might address the "surprise churn" blind spot identified in error analysis
