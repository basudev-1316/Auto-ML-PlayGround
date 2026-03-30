"""Feature engineering utilities for the AutoML training pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def build_feature_engineering_pipeline(
    task_type: str,
    *,
    enable_polynomial: bool = False,
    use_feature_selection: bool = True,
    k_best: int | str = "all",
    variance_threshold: float = 0.0,
) -> Pipeline:
    """Build a post-preprocessing feature engineering pipeline.

    The pipeline supports:
    - low-variance feature removal
    - univariate feature selection
    - optional polynomial features
    """

    score_func = f_classif if task_type == "classification" else f_regression
    steps: list[tuple[str, object]] = [
        ("variance_threshold", VarianceThreshold(threshold=variance_threshold)),
    ]

    if use_feature_selection:
        steps.append(("select_k_best", SelectKBest(score_func=score_func, k=k_best)))

    if enable_polynomial:
        steps.append(
            (
                "polynomial_features",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            )
        )

    return Pipeline(steps)


def get_feature_names_after_engineering(
    feature_engineering: Pipeline | None,
    input_feature_names: list[str] | np.ndarray,
) -> list[str]:
    """Project input feature names through the feature-engineering pipeline."""
    feature_names = np.asarray(input_feature_names, dtype=object)
    if feature_engineering is None:
        return feature_names.astype(str).tolist()

    variance_step = feature_engineering.named_steps.get("variance_threshold")
    if variance_step is not None and hasattr(variance_step, "get_support"):
        feature_names = feature_names[variance_step.get_support()]

    select_step = feature_engineering.named_steps.get("select_k_best")
    if select_step is not None and hasattr(select_step, "get_support"):
        feature_names = feature_names[select_step.get_support()]

    polynomial_step = feature_engineering.named_steps.get("polynomial_features")
    if polynomial_step is not None and hasattr(polynomial_step, "get_feature_names_out"):
        feature_names = polynomial_step.get_feature_names_out(feature_names.astype(str))

    return np.asarray(feature_names, dtype=str).tolist()
