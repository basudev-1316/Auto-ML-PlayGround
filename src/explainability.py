"""Explainability helpers based on SHAP and report export utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.feature_engineering import get_feature_names_after_engineering


def build_shap_summary(model_pipeline, x_sample: pd.DataFrame) -> pd.DataFrame | None:
    """Compute a lightweight SHAP importance summary for the trained pipeline."""
    try:
        import shap
    except Exception:
        return None

    try:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        feature_engineering = model_pipeline.named_steps.get("feature_engineering")
        estimator = model_pipeline.named_steps["model"]
        transformed = preprocessor.transform(x_sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if feature_engineering is not None:
            transformed = feature_engineering.transform(transformed)

        try:
            preprocessed_feature_names = preprocessor.get_feature_names_out()
            feature_names = get_feature_names_after_engineering(
                feature_engineering,
                preprocessed_feature_names,
            )
        except Exception:
            feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]

        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)
        shap_array = np.asarray(shap_values.values)
        if shap_array.ndim == 3:
            scores = np.mean(np.abs(shap_array), axis=(0, 2))
        elif shap_array.ndim == 2:
            scores = np.mean(np.abs(shap_array), axis=0)
        else:
            return None

        if len(feature_names) != len(scores):
            feature_names = [f"feature_{idx}" for idx in range(len(scores))]
        return pd.DataFrame({"feature": feature_names, "mean_abs_shap": scores}).sort_values(
            "mean_abs_shap", ascending=False
        )
    except Exception:
        return None


def save_shap_bar_plot(shap_summary: pd.DataFrame, output_path: Path) -> None:
    """Save a SHAP bar plot to the reports directory."""
    if shap_summary.empty:
        return

    top_rows = shap_summary.head(15).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_rows["feature"], top_rows["mean_abs_shap"], color="#1f77b4")
    plt.title("SHAP Feature Importance")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
