"""Model registry and dataset-driven model selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None


@dataclass(frozen=True)
class ModelSpec:
    """Configuration describing one candidate model family."""

    name: str
    estimator: object
    supports_tuning: bool = True


def build_model_registry(task_type: str) -> dict[str, object]:
    """Return the production AutoML model pool for the requested task type."""
    if task_type == "classification":
        models: dict[str, object] = {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
        if XGBClassifier is not None:
            models["XGBoostClassifier"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
        if LGBMClassifier is not None:
            models["LightGBMClassifier"] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        return models

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if XGBRegressor is not None:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    if LGBMRegressor is not None:
        models["LightGBMRegressor"] = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    return models


def recommend_model_names(
    *,
    row_count: int,
    column_count: int,
    numeric_feature_count: int,
    categorical_feature_count: int,
    missing_ratio: float,
    high_cardinality_columns: list[str],
    task_type: str,
    shortlist_limit: int,
) -> list[str]:
    """Recommend the most suitable model families for a dataset profile."""
    if task_type == "classification":
        recommended = [
            "LogisticRegression",
            "RandomForestClassifier",
        ]
        if categorical_feature_count > 0 or high_cardinality_columns:
            recommended.append("LightGBMClassifier")
        if row_count >= 2000 or numeric_feature_count >= 10:
            recommended.append("XGBoostClassifier")
    else:
        recommended = [
            "LinearRegression",
            "RandomForestRegressor",
        ]
        if categorical_feature_count > 0 or high_cardinality_columns:
            recommended.append("LightGBMRegressor")
        if row_count >= 2000 or numeric_feature_count >= 10:
            recommended.append("XGBoostRegressor")

    if missing_ratio > 0.05:
        preferred = "LightGBMClassifier" if task_type == "classification" else "LightGBMRegressor"
        recommended.insert(0, preferred)

    return list(dict.fromkeys(recommended))[:shortlist_limit]
