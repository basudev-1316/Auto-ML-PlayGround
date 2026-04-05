"""Tests for the reusable data preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preprocessing import (
    build_full_preprocessing_workflow,
    build_preprocessing_pipeline,
    detect_feature_types,
    fit_transform_features,
    split_features_and_target,
    transform_features,
)


def build_sample_training_frame() -> pd.DataFrame:
    """Create a mixed-type training dataframe with missing values."""
    return pd.DataFrame(
        {
            "age": [25.0, np.nan, 40.0, 31.0],
            "income": [50_000.0, 62_000.0, np.nan, 71_000.0],
            "city": ["Delhi", "Mumbai", None, "Delhi"],
            "segment": ["A", "B", "A", None],
            "target": [0, 1, 0, 1],
        }
    )


def test_detect_feature_types_splits_numeric_and_categorical_columns() -> None:
    features, _ = split_features_and_target(build_sample_training_frame(), "target")

    schema = detect_feature_types(features)

    assert schema.numeric_columns == ["age", "income"]
    assert schema.categorical_columns == ["city", "segment"]


def test_preprocessing_pipeline_imputes_and_encodes_without_missing_values() -> None:
    features, _ = split_features_and_target(build_sample_training_frame(), "target")
    pipeline = build_preprocessing_pipeline(features)

    transformed = fit_transform_features(pipeline, features)

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_preprocessing_pipeline_handles_unseen_categories_safely() -> None:
    training_features, _ = split_features_and_target(build_sample_training_frame(), "target")
    pipeline = build_preprocessing_pipeline(training_features)
    fit_transform_features(pipeline, training_features)

    inference_features = pd.DataFrame(
        {
            "age": [29.0],
            "income": [58_000.0],
            "city": ["Chennai"],
            "segment": ["C"],
        }
    )
    transformed = transform_features(pipeline, inference_features)

    assert transformed.shape[0] == 1


def test_full_preprocessing_workflow_wraps_preprocessor() -> None:
    features, _ = split_features_and_target(build_sample_training_frame(), "target")

    workflow = build_full_preprocessing_workflow(features)

    assert "preprocessor" in workflow.named_steps
