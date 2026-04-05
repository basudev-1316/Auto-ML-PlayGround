"""Tests for automatic ML problem detection."""

from __future__ import annotations

import pandas as pd
import pytest

from src.problem_detection import detect_problem_type


def test_detect_problem_type_returns_regression_for_numeric_target() -> None:
    df = pd.DataFrame(
        {
            "age": [22, 30, 41, 36],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune"],
            "price": [100000.0, 150000.0, 175000.0, 160000.0],
        }
    )

    result = detect_problem_type(df, "price")

    assert result.problem_type == "regression"
    assert result.feature_types.numeric_columns == ["age"]
    assert result.feature_types.categorical_columns == ["city"]


def test_detect_problem_type_returns_classification_for_categorical_target() -> None:
    df = pd.DataFrame(
        {
            "age": [22, 30, 41, 36],
            "income": [30000, 50000, 70000, 45000],
            "segment": ["low", "mid", "high", "mid"],
        }
    )

    result = detect_problem_type(df, "segment")

    assert result.problem_type == "classification"
    assert result.feature_types.numeric_columns == ["age", "income"]
    assert result.feature_types.categorical_columns == []


def test_detect_problem_type_raises_for_missing_target_column() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="was not found"):
        detect_problem_type(df, "target")


def test_detect_problem_type_raises_for_target_with_single_unique_value() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": ["same", "same", "same"],
        }
    )

    with pytest.raises(ValueError, match="at least two unique"):
        detect_problem_type(df, "target")


def test_detect_problem_type_raises_when_no_feature_columns_exist() -> None:
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="at least one feature column"):
        detect_problem_type(df, "target")
