"""
model_loader.py
----------------
Loads the trained CatBoost model and its companion metadata, and exposes
them through a single object so the rest of the application (API, batch
processing, dashboard) never has to worry about column order, which
columns are categorical, or how the target was encoded.
"""

import json
from pathlib import Path

from catboost import CatBoostClassifier


class ChurnModel:
    """
    Wraps the trained CatBoost churn model together with the metadata
    needed to use it correctly: which columns are categorical, the
    expected feature order, and the target class mapping.
    """

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self._validate_paths()

        self.model = self._load_model()
        self.metadata = self._load_metadata()

        self.feature_names = self.metadata["feature_names"]
        self.categorical_columns = self.metadata["categorical_columns"]
        self.target_classes = self.metadata["target_classes"]  # e.g. ['No', 'Yes']

        self._validate_metadata_consistency()

    def _validate_paths(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Copy your saved .cbm file into the models/ directory."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at '{self.metadata_path}'. "
                "Copy your saved model_metadata.json into the models/ directory."
            )

    def _load_model(self) -> CatBoostClassifier:
        model = CatBoostClassifier()
        model.load_model(str(self.model_path))
        return model

    def _load_metadata(self) -> dict:
        with open(self.metadata_path) as f:
            return json.load(f)

    def _validate_metadata_consistency(self):
        """
        Catches the most common real-world deployment bug: metadata that
        doesn't actually match the model it's shipped alongside (e.g. an
        outdated metadata.json left over from a previous training run).
        """
        required_keys = {"feature_names", "categorical_columns", "target_classes"}
        missing = required_keys - self.metadata.keys()
        if missing:
            raise ValueError(f"model_metadata.json is missing required keys: {missing}")

        if len(self.target_classes) != 2:
            raise ValueError(
                f"Expected exactly 2 target classes for binary classification, "
                f"got {self.target_classes}"
            )

        unknown_cats = set(self.categorical_columns) - set(self.feature_names)
        if unknown_cats:
            raise ValueError(
                f"categorical_columns contains names not present in feature_names: {unknown_cats}"
            )

    def get_cat_feature_indices(self) -> list:
        """CatBoost's Pool/predict methods need categorical column *positions*, not names."""
        return [self.feature_names.index(col) for col in self.categorical_columns]

    def __repr__(self):
        return (
            f"ChurnModel(features={len(self.feature_names)}, "
            f"categorical={len(self.categorical_columns)}, "
            f"classes={self.target_classes})"
        )


def load_default_model() -> "ChurnModel":
    """Convenience loader using the standard project paths."""
    base_dir = Path(__file__).resolve().parent.parent  # goal intern project_1/
    return ChurnModel(
        model_path=base_dir / "models" / "catboost_churn_model.cbm",
        metadata_path=base_dir / "models" / "model_metadata.json",
    )