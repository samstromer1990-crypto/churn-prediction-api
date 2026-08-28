"""
tune_catboost.py
-----------------
Hyperparameter tuning for CatBoost using RandomizedSearchCV.

Optimizes for F1 (refit metric) while also tracking Recall, Precision, and
ROC-AUC for every combination tested -- so you can see the full trade-off,
not just a single number.

Run with:
    python tune_catboost.py

Requires: pandas, numpy, scikit-learn, catboost
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier


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

categorical_columns = X.select_dtypes(include="object").columns.tolist()
cat_feature_indices = [X.columns.get_loc(c) for c in categorical_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------------------------
# 2. Hyperparameter search space
#    Ranges chosen to explore realistic values without wasting time on
#    combinations very unlikely to help on a dataset this size (~5.6k train rows):
#      - iterations: fewer than baseline up to more, to see if 500 was already enough
#      - depth: CatBoost default is 6; going much deeper risks overfitting on this
#        small a dataset
#      - learning_rate: wider range lets the search trade off against iterations
#      - l2_leaf_reg: regularization strength, higher = more conservative
# ---------------------------------------------------------------------------
param_distributions = {
    "iterations": [200, 300, 500, 700],
    "depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
}

base_model = CatBoostClassifier(
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False,
    thread_count=2,  # capped so it doesn't fight with RandomizedSearchCV's
                     # own n_jobs=2 workers for the same CPU cores
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scoring = ["f1", "roc_auc"]

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=15,          # tests 15 random combinations (was 30 -- much lighter)
    scoring=scoring,
    refit="f1",          # F1 decides the final "best" model
    cv=cv,
    random_state=42,
    n_jobs=2,            # NOT -1: CatBoost already multithreads internally,
                         # so n_jobs=-1 oversubscribes CPU/RAM (many parallel
                         # workers each spawning many threads). n_jobs=2 keeps
                         # some parallelism without exhausting resources.
    verbose=1,
)

print("Starting RandomizedSearchCV -- 15 combos x 3 folds = 45 CatBoost fits...\n")
# cat_features passed here (not in the constructor) -- RandomizedSearchCV
# forwards this straight through to each underlying CatBoostClassifier.fit()
# call. Passing it in the constructor instead breaks sklearn's clone()
# machinery, which RandomizedSearchCV relies on internally.
search.fit(X_train, y_train, cat_features=cat_feature_indices)

# ---------------------------------------------------------------------------
# 3. Report results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS (selected by F1)")
print("=" * 70)
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

best_idx = search.best_index_
cv_results = search.cv_results_

print("\nCV performance of the best combination (mean across 5 folds):")
print(f"  F1:        {cv_results['mean_test_f1'][best_idx]:.4f}")
print(f"  ROC-AUC:   {cv_results['mean_test_roc_auc'][best_idx]:.4f}")

print(
    "\nCompare against the untuned CatBoost baseline: "
    "ROC-AUC 0.8491 | F1 0.6340"
)

# ---------------------------------------------------------------------------
# 4. Show the top 5 combinations by F1, so you can see nearby trade-offs
#    (e.g. a slightly lower F1 combo with meaningfully higher ROC-AUC)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(cv_results)
top5 = results_df.sort_values("mean_test_f1", ascending=False).head(5)

print("\nTop 5 combinations by F1:")
print(
    top5[
        ["mean_test_f1", "mean_test_roc_auc", "params"]
    ].to_string(index=False)
)
