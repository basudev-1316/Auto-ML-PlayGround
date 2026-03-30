"""Evaluation utilities and report artifact generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.feature_engineering import get_feature_names_after_engineering


def build_test_metrics(
    task_type: str,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute task-specific test metrics."""
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
    """Extract feature importance from supported estimators."""
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
    except Exception:
        return None

    importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)

    if importances is None:
        return None

    try:
        preprocessed_feature_names = preprocessor.get_feature_names_out()
        feature_names = get_feature_names_after_engineering(
            pipeline.named_steps.get("feature_engineering"),
            preprocessed_feature_names,
        )
    except Exception:
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


def save_model_comparison(results_df: pd.DataFrame, reports_dir: Path, metric_label: str) -> None:
    """Save comparison CSV and bar chart into the reports directory."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    ascending = metric_label.upper() == "RMSE"
    order_df = results_df.sort_values("cv_score", ascending=ascending)
    sns.barplot(data=order_df, x="cv_score", y="model", hue="best", dodge=False, palette="Blues_r")
    plt.title(f"Model Comparison ({metric_label})")
    plt.xlabel(metric_label)
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison.png")
    plt.close()


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, reports_dir: Path) -> None:
    """Save a confusion matrix heatmap for classification tasks."""
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "confusion_matrix.png")
    plt.close()
