"""Unit tests for the model registry and recommendation engine."""

from __future__ import annotations

from src.model_selection import build_model_registry, recommend_model_names


def test_classification_registry_contains_required_models() -> None:
    registry = build_model_registry("classification")

    assert "LogisticRegression" in registry
    assert "RandomForestClassifier" in registry


def test_regression_registry_contains_required_models() -> None:
    registry = build_model_registry("regression")

    assert "LinearRegression" in registry
    assert "RandomForestRegressor" in registry


def test_recommend_model_names_respects_shortlist_limit() -> None:
    recommendations = recommend_model_names(
        row_count=5000,
        column_count=20,
        numeric_feature_count=12,
        categorical_feature_count=3,
        missing_ratio=0.08,
        high_cardinality_columns=["city"],
        task_type="classification",
        shortlist_limit=3,
    )

    assert len(recommendations) <= 3
