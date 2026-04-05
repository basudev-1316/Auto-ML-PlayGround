"""Evaluation and reporting utilities for the AutoML system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def ensure_reports_dir(reports_dir: str | Path) -> Path:
    """Create the reports directory if needed and return it as a Path."""
    resolved_path = Path(reports_dir)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def evaluate_model(
    problem_type: str,
    model: Any,
    features: Any,
    target: Any,
) -> dict[str, float]:
    """Compute task-appropriate evaluation metrics for a trained model."""
    predictions = model.predict(features)

    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(target, predictions)),
        }

    rmse = float(mean_squared_error(target, predictions) ** 0.5)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": rmse,
    }


def build_model_comparison_dataframe(
    comparison_results: dict[str, dict[str, object]],
    best_model_name: str | None = None,
) -> pd.DataFrame:
    """Convert raw comparison results into a ranked reporting dataframe."""
    if not comparison_results:
        raise ValueError("Comparison results cannot be empty.")

    rows = []
    metric_name = str(next(iter(comparison_results.values())).get("metric_name", "Score")).upper()
    lower_is_better = "RMSE" in metric_name

    for model_name, metrics in comparison_results.items():
        rows.append(
            {
                "model_name": model_name,
                "mean_score": float(metrics.get("mean_score", 0.0)),
                "std_score": float(metrics.get("std_score", 0.0)),
                "train_time": float(metrics.get("train_time", 0.0)),
                "metric_name": metric_name,
                "is_best": model_name == best_model_name,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("mean_score", ascending=lower_is_better).reset_index(drop=True)
    comparison_df["rank"] = comparison_df.index + 1
    comparison_df["highlight"] = comparison_df["is_best"].map({True: "Best Model", False: "Other Models"})
    return comparison_df


def save_model_comparison_report(
    comparison_results: dict[str, dict[str, object]],
    reports_dir: str | Path,
    best_model_name: str | None = None,
) -> Path:
    """Save the model comparison table and score bar chart into the reports directory."""
    output_dir = ensure_reports_dir(reports_dir)
    comparison_df = build_model_comparison_dataframe(comparison_results, best_model_name=best_model_name)

    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    metric_name = str(comparison_df["metric_name"].iloc[0])
    lower_is_better = "RMSE" in metric_name
    plot_df = comparison_df.sort_values("mean_score", ascending=lower_is_better)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="mean_score",
        y="model_name",
        hue="highlight",
        dodge=False,
        palette={"Best Model": "#ff7a18", "Other Models": "#7fa7ff"},
    )
    plt.title(f"Model Comparison ({metric_name})")
    plt.xlabel(metric_name)
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_dir / "model_scores.png")
    plt.close()

    return csv_path


def save_confusion_matrix_report(
    actual_values: Any,
    predicted_values: Any,
    reports_dir: str | Path,
) -> Path:
    """Save a confusion-matrix heatmap for classification workloads."""
    output_dir = ensure_reports_dir(reports_dir)
    labels = sorted(pd.Series(actual_values).dropna().unique().tolist())
    cm = confusion_matrix(actual_values, predicted_values, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_residual_plot_report(
    actual_values: Any,
    predicted_values: Any,
    reports_dir: str | Path,
) -> Path:
    """Save a residual plot for regression workloads."""
    output_dir = ensure_reports_dir(reports_dir)
    actual_series = pd.Series(actual_values, name="actual")
    predicted_series = pd.Series(predicted_values, name="predicted")
    residual_df = pd.DataFrame(
        {
            "actual": actual_series,
            "predicted": predicted_series,
            "residual": actual_series - predicted_series,
        }
    )

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=residual_df, x="predicted", y="residual", color="#1f77b4")
    plt.axhline(0.0, color="#ff7a18", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.tight_layout()

    output_path = output_dir / "residual_plot.png"
    plt.savefig(output_path)
    plt.close()
    return output_path
