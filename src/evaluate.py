from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def build_test_metrics(
    task_type: str,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        }
        unique_classes = pd.Series(y_test).dropna().unique()
        if len(unique_classes) == 2 and probabilities is not None:
            try:
                positive_scores = probabilities[:, 1] if getattr(probabilities, "ndim", 1) == 2 and probabilities.shape[1] >= 2 else probabilities
                metrics["roc_auc"] = float(roc_auc_score(y_test, positive_scores))
            except Exception:
                pass
        return metrics

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": rmse,
    }


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)

    if importances is None or len(importances) != len(feature_names):
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
