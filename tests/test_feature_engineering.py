"""Unit tests for feature engineering helpers."""

from __future__ import annotations

import numpy as np

from src.feature_engineering import (
    build_feature_engineering_pipeline,
    get_feature_names_after_engineering,
)


def test_feature_engineering_pipeline_builds_expected_steps() -> None:
    pipeline = build_feature_engineering_pipeline(
        "classification",
        enable_polynomial=True,
        use_feature_selection=True,
        k_best=2,
    )

    assert "variance_threshold" in pipeline.named_steps
    assert "select_k_best" in pipeline.named_steps
    assert "polynomial_features" in pipeline.named_steps


def test_feature_names_follow_selection_masks() -> None:
    pipeline = build_feature_engineering_pipeline(
        "regression",
        enable_polynomial=False,
        use_feature_selection=True,
        k_best=2,
    )
    matrix = np.array(
        [
            [0.0, 1.0, 10.0],
            [0.0, 2.0, 11.0],
            [0.0, 3.0, 12.0],
            [0.0, 4.0, 13.0],
        ]
    )
    target = np.array([1.0, 2.0, 3.0, 4.0])
    pipeline.fit(matrix, target)

    feature_names = get_feature_names_after_engineering(
        pipeline,
        ["constant_feature", "signal_a", "signal_b"],
    )

    assert feature_names == ["signal_a", "signal_b"]
