"""Tests for the multi-model training system."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.model_training import build_training_pipeline, train_all_models


def test_build_training_pipeline_contains_preprocessor_and_model() -> None:
    features = pd.DataFrame(
        {
            "age": [21, 25, 29],
            "city": ["Delhi", "Pune", "Mumbai"],
        }
    )

    from src.data_preprocessing import build_preprocessing_pipeline

    preprocessing_pipeline = build_preprocessing_pipeline(features)
    training_pipeline = build_training_pipeline(preprocessing_pipeline, LogisticRegression())

    assert list(training_pipeline.named_steps.keys()) == ["preprocessor", "model"]


def test_train_all_models_returns_results_for_classification(monkeypatch) -> None:
    features = pd.DataFrame(
        {
            "age": [21, 25, 29, 31, 37, 42, 48, 52, 23, 27],
            "income": [35_000, 42_000, 51_000, 58_000, 61_000, 73_000, 82_000, 90_000, 39_000, 47_000],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
        }
    )
    target = pd.Series(["no", "no", "no", "yes", "yes", "yes", "yes", "yes", "no", "no"], name="buy")

    monkeypatch.setattr(
        "src.model_training.build_model_registry",
        lambda problem_type: {"LogisticRegression": LogisticRegression(max_iter=500)},
    )

    results, trained_models = train_all_models(features, target, "classification")

    assert "LogisticRegression" in results
    assert set(results["LogisticRegression"].keys()) == {"mean_score", "std_score", "train_time"}
    assert "LogisticRegression" in trained_models


def test_train_all_models_returns_results_for_regression(monkeypatch) -> None:
    features = pd.DataFrame(
        {
            "sqft": [800, 900, 950, 1100, 1200, 1300, 1500, 1600, 1750, 1900],
            "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
        }
    )
    target = pd.Series([120000, 140000, 155000, 180000, 210000, 230000, 260000, 280000, 310000, 340000], name="price")

    monkeypatch.setattr(
        "src.model_training.build_model_registry",
        lambda problem_type: {"LinearRegression": LinearRegression()},
    )

    results, trained_models = train_all_models(features, target, "regression")

    assert "LinearRegression" in results
    assert set(results["LinearRegression"].keys()) == {"mean_score", "std_score", "train_time"}
    assert results["LinearRegression"]["mean_score"] >= 0.0
    assert "LinearRegression" in trained_models
