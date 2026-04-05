"""Tests for experiment tracking output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.experiment_tracking import generate_run_id, track_experiments


class DummyEstimator:
    """Minimal estimator stub for parameter extraction tests."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def get_params(self, deep: bool = False) -> dict[str, float]:
        return {"alpha": self.alpha}


def test_generate_run_id_has_expected_prefix() -> None:
    run_id = generate_run_id()

    assert run_id.startswith("run_")
    assert len(run_id) > 4


def test_track_experiments_writes_expected_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "experiments.csv"
    results = {
        "LinearRegression": {"mean_score": 0.91, "std_score": 0.03, "train_time": 0.5},
        "RandomForestRegressor": {"mean_score": 0.95, "std_score": 0.02, "train_time": 1.1},
    }
    trained_models = {
        "LinearRegression": Pipeline([("model", DummyEstimator(alpha=0.5))]),
        "RandomForestRegressor": Pipeline([("model", DummyEstimator(alpha=2.0))]),
    }

    saved_path = track_experiments("run_test123", results, trained_models, output_path)

    assert saved_path == output_path
    experiments_df = pd.read_csv(output_path)
    assert list(experiments_df.columns) == ["run_id", "model_name", "parameters", "score"]
    assert len(experiments_df) == 2
    assert set(experiments_df["model_name"]) == {"LinearRegression", "RandomForestRegressor"}
    parsed_parameters = json.loads(experiments_df.iloc[0]["parameters"])
    assert "alpha" in parsed_parameters
