"""SHAP-based explainability utilities for the AutoML system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation import ensure_reports_dir
from src.logger import get_logger

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


LOGGER = get_logger(__name__)


TREE_MODEL_KEYWORDS = (
    "randomforest",
    "xgb",
    "lgbm",
    "lightgbm",
    "decisiontree",
    "extratrees",
    "gradientboosting",
)


@dataclass
class ExplanationArtifacts:
    """Container for SHAP explanation outputs and saved report artifacts."""

    shap_values: Any
    transformed_features: np.ndarray
    feature_names: list[str]
    importance_frame: pd.DataFrame
    feature_importance_path: Path | None = None
    summary_plot_path: Path | None = None


def _extract_model_and_features(model: Any, features: Any) -> tuple[Any, np.ndarray, list[str]]:
    """Extract the fitted estimator, transformed feature matrix, and feature names."""
    if isinstance(features, pd.DataFrame):
        base_feature_names = features.columns.tolist()
    else:
        base_feature_names = [f"feature_{index}" for index in range(np.asarray(features).shape[1])]

    if isinstance(model, Pipeline) and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed_features = preprocessor.transform(features)
        if hasattr(transformed_features, "toarray"):
            transformed_features = transformed_features.toarray()

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = base_feature_names
        return estimator, np.asarray(transformed_features), feature_names

    return model, np.asarray(features), base_feature_names


def _build_explainer(estimator: Any, transformed_features: np.ndarray) -> Any:
    """Create the most suitable SHAP explainer for the fitted estimator."""
    estimator_name = estimator.__class__.__name__.lower()
    if any(keyword in estimator_name for keyword in TREE_MODEL_KEYWORDS):
        return shap.TreeExplainer(estimator)
    return shap.Explainer(estimator, transformed_features)


def _normalize_shap_values(raw_shap_values: Any) -> np.ndarray:
    """Normalize SHAP outputs into a 2D array for plotting and aggregation."""
    values = raw_shap_values.values if hasattr(raw_shap_values, "values") else raw_shap_values
    values = np.asarray(values)

    if values.ndim == 3:
        values = np.mean(np.abs(values), axis=2)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)

    return values


def build_shap_summary(model: Any, features: Any) -> ExplanationArtifacts:
    """Generate SHAP values and a feature-importance summary for the best model."""
    if shap is None:
        raise ImportError("The 'shap' package is required to build explainability artifacts.")

    estimator, transformed_features, feature_names = _extract_model_and_features(model, features)
    explainer = _build_explainer(estimator, transformed_features)
    raw_shap_values = explainer(transformed_features)
    normalized_values = _normalize_shap_values(raw_shap_values)

    if normalized_values.shape[1] != len(feature_names):
        feature_names = [f"feature_{index}" for index in range(normalized_values.shape[1])]

    importance_scores = np.mean(np.abs(normalized_values), axis=0)
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance_scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    LOGGER.info("SHAP values generated for %d features.", len(feature_names))

    return ExplanationArtifacts(
        shap_values=raw_shap_values,
        transformed_features=transformed_features,
        feature_names=feature_names,
        importance_frame=importance_frame,
    )


def save_feature_importance_report(
    explanation: ExplanationArtifacts,
    reports_dir: str | Path,
) -> Path:
    """Save a SHAP feature-importance bar chart into the reports directory."""
    output_dir = ensure_reports_dir(reports_dir)
    output_path = output_dir / "feature_importance.png"

    top_rows = explanation.importance_frame.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_rows["feature"], top_rows["importance"], color="#1f77b4")
    plt.title("SHAP Feature Importance")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    explanation.feature_importance_path = output_path
    explanation.importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)
    LOGGER.info("Saved SHAP feature importance report to %s", output_path)
    return output_path


def save_shap_summary_report(explanation: ExplanationArtifacts, reports_dir: str | Path) -> Path:
    """Save a SHAP summary plot into the reports directory."""
    if shap is None:
        raise ImportError("The 'shap' package is required to save SHAP reports.")

    output_dir = ensure_reports_dir(reports_dir)
    output_path = output_dir / "shap_summary.png"

    plt.figure()
    shap.summary_plot(
        explanation.shap_values,
        features=explanation.transformed_features,
        feature_names=explanation.feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    explanation.summary_plot_path = output_path
    LOGGER.info("Saved SHAP summary report to %s", output_path)
    return output_path


def explain_best_model(model: Any, features: Any, reports_dir: str | Path) -> ExplanationArtifacts:
    """Generate SHAP explanations for the selected best model and save report artifacts."""
    explanation = build_shap_summary(model, features)
    save_feature_importance_report(explanation, reports_dir)
    save_shap_summary_report(explanation, reports_dir)
    return explanation
