from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.automl_pipeline import AutoMLPipeline


def run_training(
    df: pd.DataFrame,
    target_column: str,
    model_path: str | Path,
    task_type_override: str | None = None,
    training_mode: str = "Balanced",
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    automl = AutoMLPipeline(
        df,
        target=target_column,
        model_path=model_path,
        task_type_override=task_type_override,
        training_mode=training_mode,
        progress_callback=progress_callback,
    )
    results = automl.run()
    evaluation_artifacts = {
        "task_type": automl.task_type,
        "model": automl.best_model,
        "x_test": automl.x_test,
        "y_test": automl.y_test,
        "predictions": automl.best_test_predictions,
        "probabilities": automl.best_test_probabilities,
        "metrics": automl.best_test_metrics,
        "best_model_name": automl.best_model_name,
        "feature_importance": automl.best_feature_importance,
        "dataset_profile": automl.dataset_profile,
    }
    return results, evaluation_artifacts
