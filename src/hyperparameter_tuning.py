"""Optuna-based hyperparameter tuning for top-performing AutoML models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from src.data_preprocessing import build_preprocessing_pipeline
from src.model_training import (
    build_cv_strategy,
    build_model_registry,
    build_training_pipeline,
    prepare_target_for_training,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Container describing the best outcome from one Optuna tuning run."""

    model_name: str
    best_model: Any
    best_params: dict[str, Any]
    best_score: float


def build_optuna_study(problem_type: str) -> optuna.Study:
    """Create an Optuna study with pruning enabled for the given problem type."""
    direction = "maximize" if problem_type == "classification" else "minimize"
    return optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Define the Optuna search space for a supported model family."""
    if model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 20),
            "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "model__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name in {"XGBoostClassifier", "XGBoostRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 10),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if model_name in {"LightGBMClassifier", "LightGBMRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    raise ValueError(f"Unsupported tunable model: {model_name}")


def score_fold(problem_type: str, y_true: pd.Series, predictions: np.ndarray) -> float:
    """Score one cross-validation fold for the requested problem type."""
    if problem_type == "classification":
        return float(accuracy_score(y_true, predictions))
    return float(np.sqrt(mean_squared_error(y_true, predictions)))


def _log_trial(study: optuna.Study, trial: optuna.Trial) -> None:
    """Log completed Optuna trials and the current best configuration."""
    LOGGER.info(
        "Trial %s completed for study '%s' with value=%s params=%s",
        trial.number,
        study.study_name,
        trial.value,
        trial.params,
    )


def _build_supported_tuning_registry(problem_type: str) -> dict[str, Any]:
    """Return only the model families supported for advanced tuning."""
    base_registry = build_model_registry(problem_type)
    supported_names = {
        "classification": {
            "RandomForestClassifier",
            "XGBoostClassifier",
            "LightGBMClassifier",
        },
        "regression": {
            "RandomForestRegressor",
            "XGBoostRegressor",
            "LightGBMRegressor",
        },
    }
    return {
        model_name: estimator
        for model_name, estimator in base_registry.items()
        if model_name in supported_names[problem_type]
    }


def _apply_fallback_defaults(model_name: str, estimator: Any) -> Any:
    """Keep estimators stable across training runs by enforcing safe defaults."""
    if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
        estimator.set_params(random_state=42, n_jobs=-1)
    return estimator


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    *,
    n_trials: int = 25,
    cv_folds: int = 5,
) -> tuple[Any, dict[str, Any], float]:
    """Tune a supported model family and return the best fitted pipeline."""
    supported_registry = _build_supported_tuning_registry(problem_type)
    if model_name not in supported_registry:
        raise ValueError(
            f"Model '{model_name}' is not supported for advanced tuning. "
            f"Supported models: {sorted(supported_registry)}"
        )

    estimator = _apply_fallback_defaults(model_name, clone(supported_registry[model_name]))
    preprocessing_pipeline = build_preprocessing_pipeline(X)
    training_target, _ = prepare_target_for_training(y, problem_type)
    cv_strategy = build_cv_strategy(problem_type, training_target, cv_folds=cv_folds)
    cv_splits = list(cv_strategy.split(X, training_target))

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_training_pipeline(preprocessing_pipeline, estimator)
        pipeline.set_params(**suggest_hyperparameters(trial, model_name))

        fold_scores: list[float] = []
        for fold_index, (train_index, valid_index) in enumerate(cv_splits):
            x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train = training_target.iloc[train_index]
            y_valid = training_target.iloc[valid_index]

            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_valid)
            fold_score = score_fold(problem_type, y_valid, predictions)
            fold_scores.append(fold_score)

            intermediate_value = float(np.mean(fold_scores))
            trial.report(intermediate_value, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_scores", fold_scores)
        return float(np.mean(fold_scores))

    study = build_optuna_study(problem_type)
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_trial], show_progress_bar=False)

    LOGGER.info("Best parameters for %s: %s", model_name, study.best_params)

    best_model = build_training_pipeline(preprocessing_pipeline, estimator)
    best_model.set_params(**{f"model__{key}" if not key.startswith("model__") else key: value for key, value in study.best_params.items()})
    best_model.fit(X, training_target)

    best_score = float(study.best_value)
    tuning_result = TuningResult(
        model_name=model_name,
        best_model=best_model,
        best_params=study.best_params,
        best_score=best_score,
    )
    return tuning_result.best_model, tuning_result.best_params, tuning_result.best_score
