"""Smoke tests for the end-to-end training service."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.train import run_training


def test_run_training_saves_model_and_returns_results(tmp_path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "age": [21, 25, 29, 31, 37, 42, 48, 52, 23, 27],
            "income": [35_000, 42_000, 51_000, 58_000, 61_000, 73_000, 82_000, 90_000, 39_000, 47_000],
            "segment": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
            "buy": ["no", "no", "no", "yes", "yes", "yes", "yes", "yes", "no", "no"],
        }
    )

    monkeypatch.setattr("src.train.build_model_registry", lambda task_type: {"LogisticRegression": LogisticRegression(max_iter=500)})
    monkeypatch.setattr("src.train.recommend_model_names", lambda **_: ["LogisticRegression"])

    model_path = tmp_path / "models" / "best_model.pkl"
    results_df, evaluation_artifacts = run_training(
        df=df,
        target_column="buy",
        model_path=model_path,
        task_type_override="classification",
        training_mode="Fast",
    )

    assert not results_df.empty
    assert evaluation_artifacts["task_type"] == "classification"
    assert evaluation_artifacts["best_model_name"] == "LogisticRegression"
    assert model_path.exists()
    assert (tmp_path / "reports" / "model_comparison.csv").exists()
