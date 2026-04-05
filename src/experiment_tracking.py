"""Experiment tracking helpers for AutoML training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from sklearn.pipeline import Pipeline

from src.logger import get_logger


LOGGER = get_logger(__name__)
EXPERIMENTS_PATH = Path(__file__).resolve().parent.parent / "experiments.csv"


def generate_run_id() -> str:
    """Generate a compact unique identifier for a training run."""
    return f"run_{uuid4().hex[:12]}"


def extract_model_parameters(model_pipeline: Pipeline | None) -> str:
    """Serialize estimator parameters from a fitted training pipeline."""
    if model_pipeline is None:
        return json.dumps({})

    estimator = model_pipeline.named_steps.get("model")
    if estimator is None:
        return json.dumps({})

    return json.dumps(estimator.get_params(deep=False), default=str, sort_keys=True)


def track_experiments(
    run_id: str,
    results: dict[str, dict[str, Any]],
    trained_models: dict[str, Pipeline],
    output_path: str | Path = EXPERIMENTS_PATH,
) -> Path:
    """Append model experiment records for a training run to the tracking CSV."""
    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "parameters": extract_model_parameters(trained_models.get(model_name)),
                "score": metrics["mean_score"],
            }
        )

    tracking_df = pd.DataFrame(rows)
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_path.exists():
        tracking_df.to_csv(resolved_path, mode="a", header=False, index=False)
    else:
        tracking_df.to_csv(resolved_path, index=False)

    LOGGER.info(
        "Tracked %d experiment rows for run_id=%s into %s",
        len(tracking_df),
        run_id,
        resolved_path,
    )
    return resolved_path
