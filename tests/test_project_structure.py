"""Basic structural tests for the phase-1 AutoML scaffold."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec


def test_required_modules_can_be_imported() -> None:
    required_modules = [
        "src.data_preprocessing",
        "src.benchmarking",
        "src.feature_engineering",
        "src.problem_detection",
        "src.model_training",
        "src.hyperparameter_tuning",
        "src.model_selection",
        "src.evaluation",
        "src.explainability",
        "src.experiment_tracking",
        "src.monitoring",
        "src.utils",
        "app.streamlit_app",
    ]

    if find_spec("fastapi") is not None:
        required_modules.append("api.main")

    for module_name in required_modules:
        assert import_module(module_name) is not None
