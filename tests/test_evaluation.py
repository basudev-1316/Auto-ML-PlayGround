"""Tests for evaluation metrics and report artifact generation."""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.evaluation import (
    build_model_comparison_dataframe,
    evaluate_model,
    save_confusion_matrix_report,
    save_model_comparison_report,
    save_residual_plot_report,
)


def test_build_model_comparison_dataframe_adds_rank_and_best_highlight() -> None:
    results = {
        "LogisticRegression": {
            "mean_score": 0.91,
            "std_score": 0.01,
            "train_time": 0.4,
            "metric_name": "Accuracy",
        },
        "RandomForestClassifier": {
            "mean_score": 0.89,
            "std_score": 0.03,
            "train_time": 2.0,
            "metric_name": "Accuracy",
        },
    }

    comparison_df = build_model_comparison_dataframe(results, best_model_name="LogisticRegression")

    assert list(comparison_df["rank"]) == [1, 2]
    assert comparison_df.loc[0, "highlight"] == "Best Model"


def test_save_model_comparison_report_creates_csv_and_chart(tmp_path) -> None:
    results = {
        "LinearRegression": {
            "mean_score": 12.5,
            "std_score": 0.8,
            "train_time": 0.2,
            "metric_name": "RMSE",
        },
        "RandomForestRegressor": {
            "mean_score": 11.8,
            "std_score": 1.1,
            "train_time": 1.5,
            "metric_name": "RMSE",
        },
    }

    csv_path = save_model_comparison_report(results, tmp_path, best_model_name="RandomForestRegressor")

    assert csv_path.exists()
    assert (tmp_path / "model_scores.png").exists()


def test_save_confusion_matrix_report_creates_png(tmp_path) -> None:
    output_path = save_confusion_matrix_report(
        actual_values=pd.Series(["yes", "no", "yes", "no"]),
        predicted_values=pd.Series(["yes", "no", "no", "no"]),
        reports_dir=tmp_path,
    )

    assert output_path.exists()


def test_save_residual_plot_report_creates_png(tmp_path) -> None:
    output_path = save_residual_plot_report(
        actual_values=pd.Series([10.0, 12.0, 15.0, 18.0]),
        predicted_values=pd.Series([9.5, 11.5, 15.5, 17.0]),
        reports_dir=tmp_path,
    )

    assert output_path.exists()


def test_evaluate_model_returns_classification_metrics() -> None:
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    target = pd.Series(["yes", "yes", "yes", "yes"])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(features, target)

    metrics = evaluate_model("classification", model, features, target)

    assert "accuracy" in metrics


def test_evaluate_model_returns_regression_metrics() -> None:
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    target = pd.Series([10.0, 12.0, 14.0, 16.0])
    model = DummyRegressor(strategy="mean")
    model.fit(features, target)

    metrics = evaluate_model("regression", model, features, target)

    assert set(metrics.keys()) == {"r2", "mae", "rmse"}
