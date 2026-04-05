"""Tests for Optuna-based hyperparameter tuning."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.hyperparameter_tuning import tune_model


def test_tune_model_returns_best_pipeline_for_classification(monkeypatch) -> None:
    features = pd.DataFrame(
        {
            "age": [21, 25, 29, 31, 37, 42, 48, 52, 23, 27],
            "income": [35_000, 42_000, 51_000, 58_000, 61_000, 73_000, 82_000, 90_000, 39_000, 47_000],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
        }
    )
    target = pd.Series(["no", "no", "no", "yes", "yes", "yes", "yes", "yes", "no", "no"], name="buy")

    monkeypatch.setattr(
        "src.hyperparameter_tuning.build_model_registry",
        lambda problem_type: {"RandomForestClassifier": RandomForestClassifier(random_state=42)},
    )

    best_model, best_params, best_score = tune_model(
        "RandomForestClassifier",
        features,
        target,
        "classification",
        n_trials=2,
    )

    assert hasattr(best_model, "predict")
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)


def test_tune_model_returns_best_pipeline_for_regression(monkeypatch) -> None:
    features = pd.DataFrame(
        {
            "sqft": [800, 900, 950, 1100, 1200, 1300, 1500, 1600, 1750, 1900],
            "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
        }
    )
    target = pd.Series([120000, 140000, 155000, 180000, 210000, 230000, 260000, 280000, 310000, 340000], name="price")

    monkeypatch.setattr(
        "src.hyperparameter_tuning.build_model_registry",
        lambda problem_type: {"RandomForestRegressor": RandomForestRegressor(random_state=42)},
    )

    best_model, best_params, best_score = tune_model(
        "RandomForestRegressor",
        features,
        target,
        "regression",
        n_trials=2,
    )

    assert hasattr(best_model, "predict")
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score >= 0.0


def test_tune_model_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="not supported for advanced tuning"):
        tune_model(
            "LogisticRegression",
            pd.DataFrame({"feature": [1, 2, 3, 4]}),
            pd.Series([0, 1, 0, 1]),
            "classification",
            n_trials=1,
        )
