"""Tests for SHAP-based model explainability."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import build_preprocessing_pipeline
from src.explainability import (
    build_shap_summary,
    explain_best_model,
    save_feature_importance_report,
    save_shap_summary_report,
)
from src.model_training import build_training_pipeline


def build_regression_pipeline():
    """Build a small fitted tree-based pipeline for explainability tests."""
    features = pd.DataFrame(
        {
            "sqft": [800, 900, 950, 1100, 1200, 1300, 1500, 1600],
            "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C"],
        }
    )
    target = pd.Series([120000, 140000, 155000, 180000, 210000, 230000, 260000, 280000], name="price")
    preprocessor = build_preprocessing_pipeline(features)
    pipeline = build_training_pipeline(
        preprocessor,
        RandomForestRegressor(n_estimators=20, random_state=42),
    )
    pipeline.fit(features, target)
    return pipeline, features


def test_build_shap_summary_returns_explanation_object() -> None:
    pipeline, features = build_regression_pipeline()

    explanation = build_shap_summary(pipeline, features)

    assert explanation.importance_frame is not None
    assert not explanation.importance_frame.empty
    assert explanation.feature_names


def test_save_feature_importance_report_creates_outputs(tmp_path) -> None:
    pipeline, features = build_regression_pipeline()
    explanation = build_shap_summary(pipeline, features)

    output_path = save_feature_importance_report(explanation, tmp_path)

    assert output_path.exists()
    assert (tmp_path / "feature_importance.csv").exists()


def test_save_shap_summary_report_creates_png(tmp_path) -> None:
    pipeline, features = build_regression_pipeline()
    explanation = build_shap_summary(pipeline, features)

    output_path = save_shap_summary_report(explanation, tmp_path)

    assert output_path.exists()


def test_explain_best_model_saves_all_reports(tmp_path) -> None:
    pipeline, features = build_regression_pipeline()

    explanation = explain_best_model(pipeline, features, tmp_path)

    assert explanation.feature_importance_path is not None
    assert explanation.summary_plot_path is not None
    assert explanation.feature_importance_path.exists()
    assert explanation.summary_plot_path.exists()
