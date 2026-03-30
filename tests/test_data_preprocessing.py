"""Unit tests for dataset preparation and task detection."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing import detect_task_type, prepare_dataframe


def test_prepare_dataframe_expands_date_columns() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-02-20"],
            "feature": [1, 2],
            "target": [10.0, 12.0],
        }
    )

    prepared = prepare_dataframe(df, "target")

    assert "date" not in prepared.columns
    assert {"sale_year", "sale_month", "sale_day"}.issubset(prepared.columns)


def test_detect_task_type_identifies_string_target_as_classification() -> None:
    target = pd.Series(["yes", "no", "yes", "no"])

    detected = detect_task_type(target)

    assert detected == "classification"


def test_detect_task_type_identifies_continuous_target_as_regression() -> None:
    target = pd.Series([101.2, 110.5, 120.1, 118.9, 140.4])

    detected = detect_task_type(target)

    assert detected == "regression"
