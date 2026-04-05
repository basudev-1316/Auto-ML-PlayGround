"""Tests for lightweight monitoring metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring import load_metrics, record_failure, record_prediction_event


def test_record_prediction_event_updates_totals_and_last_model(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    metrics = record_prediction_event("RandomForestClassifier", 12, metrics_path)

    assert metrics["total_predictions"] == 12
    assert metrics["last_model_used"] == "RandomForestClassifier"
    assert metrics["error_count"] == 0


def test_record_failure_increments_error_count(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    record_prediction_event("LogisticRegression", 3, metrics_path)

    metrics = record_failure("training_failure: sample", metrics_path)

    assert metrics["total_predictions"] == 3
    assert metrics["error_count"] == 1
    assert metrics["last_model_used"] == "LogisticRegression"

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        persisted_metrics = json.load(metrics_file)

    assert persisted_metrics["error_count"] == 1


def test_load_metrics_returns_defaults_when_file_is_missing(tmp_path: Path) -> None:
    metrics = load_metrics(tmp_path / "missing_metrics.json")

    assert metrics == {
        "total_predictions": 0,
        "error_count": 0,
        "last_model_used": None,
    }
