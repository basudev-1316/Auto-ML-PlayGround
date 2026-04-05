"""Problem-detection utilities for the AutoML system."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_preprocessing import FeatureSchema, detect_feature_types


@dataclass(frozen=True)
class ProblemDetectionResult:
    """Container describing the inferred ML problem type and feature groups."""

    problem_type: str
    feature_types: FeatureSchema


def validate_target_column(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Validate the requested target column and return it as a Series."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    target = df[target_column]
    unique_values = target.dropna().nunique()
    if unique_values < 2:
        raise ValueError(
            "Target column must contain at least two unique non-null values to detect a problem type."
        )
    return target


def infer_problem_type(target: pd.Series) -> str:
    """Infer whether the target represents regression or classification."""
    if pd.api.types.is_bool_dtype(target):
        return "classification"

    if pd.api.types.is_object_dtype(target) or isinstance(target.dtype, pd.CategoricalDtype):
        return "classification"

    if pd.api.types.is_numeric_dtype(target):
        return "regression"

    return "classification"


def summarize_target_distribution(target: pd.Series) -> dict[str, object]:
    """Build a lightweight summary of the target column for diagnostics."""
    non_null_target = target.dropna()
    return {
        "dtype": str(target.dtype),
        "non_null_count": int(non_null_target.shape[0]),
        "unique_values": int(non_null_target.nunique()),
        "missing_values": int(target.isna().sum()),
    }


def detect_problem_type(df: pd.DataFrame, target_column: str) -> ProblemDetectionResult:
    """Detect the ML problem type and feature groups for a dataset."""
    target = validate_target_column(df, target_column)
    features = df.drop(columns=[target_column])

    if features.empty:
        raise ValueError("Dataset must contain at least one feature column besides the target.")

    problem_type = infer_problem_type(target)
    feature_types = detect_feature_types(features)

    return ProblemDetectionResult(
        problem_type=problem_type,
        feature_types=feature_types,
    )
