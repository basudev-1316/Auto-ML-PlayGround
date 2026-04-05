"""Tests for intelligent AutoML model selection."""

from __future__ import annotations

from src.model_selection import compare_model_scores, select_best_model


def test_compare_model_scores_penalizes_high_variance() -> None:
    results = {
        "RandomForestClassifier": {
            "mean_score": 0.92,
            "std_score": 0.08,
            "train_time": 2.1,
            "metric_name": "Accuracy",
            "model_object": "rf-model",
        },
        "LogisticRegression": {
            "mean_score": 0.91,
            "std_score": 0.01,
            "train_time": 0.4,
            "metric_name": "Accuracy",
            "model_object": "log-model",
        },
    }

    ranked = compare_model_scores(results)

    assert ranked[0]["model_name"] == "LogisticRegression"


def test_select_best_model_prefers_simpler_model_when_scores_are_within_one_percent() -> None:
    results = {
        "RandomForestClassifier": {
            "mean_score": 0.9200,
            "std_score": 0.0100,
            "train_time": 2.1,
            "metric_name": "Accuracy",
            "model_object": "rf-model",
        },
        "LogisticRegression": {
            "mean_score": 0.9150,
            "std_score": 0.0050,
            "train_time": 0.4,
            "metric_name": "Accuracy",
            "model_object": "log-model",
        },
    }

    best_model_name, best_model_object, reasoning = select_best_model(results)

    assert best_model_name == "LogisticRegression"
    assert best_model_object == "log-model"
    assert "below 1%" in reasoning


def test_select_best_model_handles_regression_rmse_as_lower_is_better() -> None:
    results = {
        "RandomForestRegressor": {
            "mean_score": 25.0,
            "std_score": 4.0,
            "train_time": 2.4,
            "metric_name": "RMSE",
            "model_object": "rf-reg",
        },
        "LinearRegression": {
            "mean_score": 25.2,
            "std_score": 0.5,
            "train_time": 0.2,
            "metric_name": "RMSE",
            "model_object": "lin-reg",
        },
    }

    best_model_name, best_model_object, reasoning = select_best_model(results)

    assert best_model_name == "LinearRegression"
    assert best_model_object == "lin-reg"
    assert "RMSE" in reasoning or "lower penalized RMSE" in reasoning


def test_select_best_model_raises_for_empty_results() -> None:
    try:
        select_best_model({})
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty results.")
