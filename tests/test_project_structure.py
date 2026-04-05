"""Basic structural tests for the phase-1 AutoML scaffold."""

from __future__ import annotations

from importlib import import_module


def test_required_modules_can_be_imported() -> None:
    required_modules = [
        "src.data_preprocessing",
        "src.feature_engineering",
        "src.problem_detection",
        "src.model_training",
        "src.hyperparameter_tuning",
        "src.model_selection",
        "src.evaluation",
        "src.explainability",
        "src.utils",
        "app.streamlit_app",
    ]

    for module_name in required_modules:
        assert import_module(module_name) is not None
