"""Explainable model-selection utilities for AutoML candidate comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.logger import get_logger


LOGGER = get_logger(__name__)

@dataclass(frozen=True)
class ModelCandidate:
    """Metadata describing one model family available to the AutoML engine."""

    name: str
    family: str
    estimator: object | None = None


SIMPLICITY_RANKING: dict[str, int] = {
    "LogisticRegression": 1,
    "LinearRegression": 1,
    "NaiveBayes": 1,
    "KNN": 2,
    "KNNRegressor": 2,
    "SVC": 3,
    "SVR": 3,
    "RandomForestClassifier": 4,
    "RandomForestRegressor": 4,
    "LightGBMClassifier": 5,
    "LightGBMRegressor": 5,
    "XGBoostClassifier": 6,
    "XGBoostRegressor": 6,
}


def build_model_registry(problem_type: str) -> dict[str, ModelCandidate]:
    """Create a lightweight model registry for the requested problem type."""
    if problem_type == "classification":
        names = [
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBoostClassifier",
            "LightGBMClassifier",
            "SVC",
            "KNN",
            "NaiveBayes",
        ]
    else:
        names = [
            "LinearRegression",
            "RandomForestRegressor",
            "XGBoostRegressor",
            "LightGBMRegressor",
            "SVR",
            "KNNRegressor",
        ]

    return {
        model_name: ModelCandidate(
            name=model_name,
            family=model_name,
            estimator=None,
        )
        for model_name in names
    }


def _extract_metric(entry: dict[str, object], key: str) -> float:
    """Return a numeric metric from a result entry, defaulting safely to zero."""
    return float(entry.get(key, 0.0))


def _get_simplicity_rank(model_name: str) -> int:
    """Return the configured simplicity rank for a model family."""
    return SIMPLICITY_RANKING.get(model_name, 999)


def _get_score_direction(results: dict[str, dict[str, object]]) -> str:
    """Infer whether higher or lower scores are better based on the metric label."""
    if not results:
        raise ValueError("Results dictionary cannot be empty.")

    first_entry = next(iter(results.values()))
    metric_name = str(first_entry.get("metric_name", "")).upper()
    if "RMSE" in metric_name:
        return "min"
    return "max"


def compare_model_scores(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Compare models after applying score, variance, and simplicity adjustments."""
    if not results:
        raise ValueError("Results dictionary cannot be empty.")

    score_direction = _get_score_direction(results)
    comparisons: list[dict[str, object]] = []

    for model_name, metrics in results.items():
        mean_score = _extract_metric(metrics, "mean_score")
        std_score = _extract_metric(metrics, "std_score")
        train_time = _extract_metric(metrics, "train_time")
        simplicity_rank = _get_simplicity_rank(model_name)
        adjusted_score = mean_score - std_score if score_direction == "max" else mean_score + std_score

        comparisons.append(
            {
                "model_name": model_name,
                "mean_score": mean_score,
                "std_score": std_score,
                "train_time": train_time,
                "simplicity_rank": simplicity_rank,
                "adjusted_score": adjusted_score,
                "model_object": metrics.get("model_object"),
            }
        )

    reverse = score_direction == "max"
    return sorted(comparisons, key=lambda item: item["adjusted_score"], reverse=reverse)


def select_best_model(results: dict[str, dict[str, object]]) -> tuple[str, Any, str]:
    """Select the best model with explainable score, variance, and simplicity logic."""
    ranked_models = compare_model_scores(results)
    best_candidate = ranked_models[0]
    runner_up = ranked_models[1] if len(ranked_models) > 1 else None
    score_direction = _get_score_direction(results)

    chosen = best_candidate
    reasoning_parts = [
        f"Started with the strongest adjusted score after penalizing variance: {best_candidate['model_name']}.",
        f"Adjusted score = {best_candidate['adjusted_score']:.4f}, mean score = {best_candidate['mean_score']:.4f}, std = {best_candidate['std_score']:.4f}.",
    ]

    if runner_up is not None:
        score_gap = abs(best_candidate["mean_score"] - runner_up["mean_score"])
        reference_score = max(abs(best_candidate["mean_score"]), 1e-12)
        gap_ratio = score_gap / reference_score

        if gap_ratio < 0.01 and runner_up["simplicity_rank"] < best_candidate["simplicity_rank"]:
            chosen = runner_up
            reasoning_parts.append(
                f"The score gap versus {runner_up['model_name']} is below 1%, so the simpler model was preferred."
            )
        else:
            if score_direction == "max":
                reasoning_parts.append(
                    f"{best_candidate['model_name']} kept the lead after variance penalty over {runner_up['model_name']}."
                )
            else:
                reasoning_parts.append(
                    f"{best_candidate['model_name']} kept the lower penalized RMSE over {runner_up['model_name']}."
                )

    if chosen["simplicity_rank"] == best_candidate["simplicity_rank"]:
        reasoning_parts.append(
            f"Simplicity rank for the selected model is {chosen['simplicity_rank']}."
        )

    reasoning = " ".join(reasoning_parts)
    LOGGER.info("Best model selected: %s. %s", chosen["model_name"], reasoning)
    return chosen["model_name"], chosen["model_object"], reasoning
