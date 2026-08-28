"""
model_catboost.py
------------------
Evaluates CatBoost on the Telco Customer Churn dataset using its native
categorical feature handling -- NOT one-hot encoding. This is the whole
point of using CatBoost: it takes raw categorical columns (as strings)
directly and handles them internally far more effectively than one-hot
encoding does, especially as category count grows.

Run with:
    python model_catboost.py

Requires: pandas, numpy, scikit-learn, catboost
    pip install catboost
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from catboost import CatBoostClassifier, Pool


# ---------------------------------------------------------------------------
# 1. Load and clean data (same fixes as the rest of the project)
# ---------------------------------------------------------------------------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)
df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

X = df.drop("Churn", axis=1)
y = df["Churn"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)  # 'No' -> 0, 'Yes' -> 1
print(f"Target classes: {list(le_target.classes_)} -> {list(le_target.transform(le_target.classes_))}")

# ---------------------------------------------------------------------------
# 2. Identify categorical columns -- CatBoost needs these named explicitly.
#    NOTE: no OneHotEncoder, no LabelEncoder on X. CatBoost wants the raw
#    strings; it builds its own internal encoding (ordered target statistics)
#    which generally outperforms one-hot, especially as cardinality grows.
# ---------------------------------------------------------------------------
categorical_columns = X.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns passed natively to CatBoost: {categorical_columns}")

# ---------------------------------------------------------------------------
# 3. Split (held out for a final sanity check, same as model_selection.py)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------------------------
# 4. Stratified 5-fold CV, matching model_selection.py so results are
#    directly comparable to the Logistic Regression / Gradient Boosting /
#    XGBoost / LightGBM / Naive Bayes numbers you already have.
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], []

print("\nRunning 5-fold stratified cross-validation for CatBoost...\n")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_columns)
    val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_columns)

    model = CatBoostClassifier(
        iterations=500,
        random_seed=42,
        auto_class_weights="Balanced",  # handles the ~73/27 imbalance
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_probs = model.predict_proba(val_pool)[:, 1]
    val_preds = model.predict(val_pool)

    roc_auc_scores.append(roc_auc_score(y_fold_val, val_probs))
    f1_scores.append(f1_score(y_fold_val, val_preds))
    precision_scores.append(precision_score(y_fold_val, val_preds))
    recall_scores.append(recall_score(y_fold_val, val_preds))

    print(
        f"Fold {fold_idx} | ROC-AUC: {roc_auc_scores[-1]:.4f} | "
        f"F1: {f1_scores[-1]:.4f} | Precision: {precision_scores[-1]:.4f} | "
        f"Recall: {recall_scores[-1]:.4f}"
    )

# ---------------------------------------------------------------------------
# 5. Summary -- directly comparable to model_selection.py's table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATBOOST -- 5-FOLD CV SUMMARY (native categorical handling)")
print("=" * 70)
print(f"ROC-AUC:   {np.mean(roc_auc_scores):.4f}  (std: {np.std(roc_auc_scores):.4f})")
print(f"F1:        {np.mean(f1_scores):.4f}  (std: {np.std(f1_scores):.4f})")
print(f"Precision: {np.mean(precision_scores):.4f}  (std: {np.std(precision_scores):.4f})")
print(f"Recall:    {np.mean(recall_scores):.4f}  (std: {np.std(recall_scores):.4f})")
print(
    "\nCompare this ROC-AUC directly against the model_selection.py table. "
    "If CatBoost's ROC-AUC here beats Gradient Boosting's 0.848, it becomes "
    "your new leading candidate for hyperparameter tuning."
)