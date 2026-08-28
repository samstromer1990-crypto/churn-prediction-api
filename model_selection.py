"""
model_selection.py
-------------------
Compares several classification models on the Telco Customer Churn dataset
using stratified k-fold cross-validation, and reports which one performs best.

Run with:
    python model_selection.py

Requires: pandas, numpy, scikit-learn
Optional: xgboost (falls back to GradientBoostingClassifier if not installed)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Load and clean data (same fixes we established in the notebook)
# ---------------------------------------------------------------------------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)
df.drop(columns=["customerID"], inplace=True)

# TotalCharges: blank strings for tenure==0 customers -> true value is 0, not missing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

X = df.drop("Churn", axis=1)
y = df["Churn"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)  # 'No' -> 0, 'Yes' -> 1
print(f"Target classes: {list(le_target.classes_)} -> {list(le_target.transform(le_target.classes_))}")

# ---------------------------------------------------------------------------
# 2. Split (held out for a final sanity check; CV happens on training data)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------------------------
# 3. Preprocessing: one-hot encode categoricals, scale numerics
#    (scaling matters for Logistic Regression, KNN, SVM; tree models ignore it,
#    but a shared pipeline keeps this script simple and still correct)
# ---------------------------------------------------------------------------
categorical_columns = X.select_dtypes(include="object").columns.tolist()
numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        # sparse_output=False: GaussianNB needs a dense array, and the dataset
        # is small enough (7k rows) that dense one-hot columns are no issue.
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ("num", StandardScaler(), numeric_columns),
    ]
)

# ---------------------------------------------------------------------------
# 4. Candidate models
#    class_weight='balanced' / scale_pos_weight handle the class imbalance
#    (~73% No / ~27% Yes) without needing manual resampling.
# ---------------------------------------------------------------------------
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos  # for XGBoost imbalance handling

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(probability=True, class_weight="balanced", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    # Naive Bayes assumes features are conditionally independent given the
    # class -- clearly violated here (e.g. InternetService/OnlineSecurity/
    # StreamingTV are all correlated). Expect it to underperform, but it's
    # still a useful, fast baseline reference point.
    "Naive Bayes": GaussianNB(),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
else:
    print("Note: xgboost not installed, skipping it. Install with: pip install xgboost")

if LIGHTGBM_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
else:
    print("Note: lightgbm not installed, skipping it. Install with: pip install lightgbm")

# ---------------------------------------------------------------------------
# 5. Cross-validate each model on TRAINING data only
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["roc_auc", "f1", "precision", "recall"]

results = []

print("\nRunning 5-fold stratified cross-validation on each model...\n")

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    row = {
        "Model": name,
        "ROC-AUC": cv_results["test_roc_auc"].mean(),
        "F1": cv_results["test_f1"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall": cv_results["test_recall"].mean(),
    }
    results.append(row)

    print(
        f"{name:22s} | ROC-AUC: {row['ROC-AUC']:.4f} | "
        f"F1: {row['F1']:.4f} | Precision: {row['Precision']:.4f} | "
        f"Recall: {row['Recall']:.4f}"
    )

# ---------------------------------------------------------------------------
# 6. Rank and report the best model
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON (ranked by ROC-AUC, 5-fold CV on training data)")
print("=" * 70)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by ROC-AUC: {best_model_name}")
print(
    "\nNote: ROC-AUC is used as the primary ranking metric because churn is "
    "imbalanced (~73% No / ~27% Yes) — accuracy alone would be misleading. "
    "Check Precision/Recall too: which one matters more depends on the "
    "business cost of missing a churner (false negative) vs. wrongly "
    "flagging a loyal customer (false positive)."
)
