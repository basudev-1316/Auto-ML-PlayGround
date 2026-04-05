"""Reusable data preprocessing utilities for the AutoML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSchema:
    """Container describing detected numeric and categorical feature groups."""

    numeric_columns: list[str]
    categorical_columns: list[str]


def validate_dataset(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the input dataset contains the requested target column."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into feature matrix and target vector."""
    validate_dataset(df, target_column)
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def detect_feature_types(features: pd.DataFrame) -> FeatureSchema:
    """Automatically detect numeric and categorical feature columns."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return FeatureSchema(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def build_numeric_pipeline(imputation_strategy: str = "median") -> Pipeline:
    """Build the numeric preprocessing pipeline with imputation and scaling."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputation_strategy)),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline(imputation_strategy: str = "most_frequent") -> Pipeline:
    """Build the categorical preprocessing pipeline with mode imputation and one-hot encoding."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputation_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessing_pipeline(
    features: pd.DataFrame,
    *,
    numeric_imputation_strategy: str = "median",
    categorical_imputation_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Build a preprocessing pipeline with automatic feature-type detection."""
    schema = detect_feature_types(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if schema.numeric_columns:
        transformers.append(
            (
                "numeric",
                build_numeric_pipeline(imputation_strategy=numeric_imputation_strategy),
                schema.numeric_columns,
            )
        )

    if schema.categorical_columns:
        transformers.append(
            (
                "categorical",
                build_categorical_pipeline(imputation_strategy=categorical_imputation_strategy),
                schema.categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_features(
    preprocessing_pipeline: ColumnTransformer,
    features: pd.DataFrame,
) -> Any:
    """Apply a fitted preprocessing pipeline to the feature matrix."""
    return preprocessing_pipeline.transform(features)


def fit_transform_features(
    preprocessing_pipeline: ColumnTransformer,
    features: pd.DataFrame,
) -> Any:
    """Fit a preprocessing pipeline and return the transformed features."""
    return preprocessing_pipeline.fit_transform(features)


def build_full_preprocessing_workflow(features: pd.DataFrame) -> Pipeline:
    """Create a reusable workflow wrapper around the preprocessing stage."""
    preprocessing_pipeline = build_preprocessing_pipeline(features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing_pipeline),
        ]
    )
