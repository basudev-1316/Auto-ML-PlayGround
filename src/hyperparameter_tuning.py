"""Optuna-based hyperparameter tuning for production AutoML training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone


@dataclass
class TuningResult:
    """Result of tuning one model family."""

    model_name: str
    pipeline: Pipeline
    cv_score: float
    best_params: dict[str, object]
    metric_name: str


def _get_cv(task_type: str, y_train: pd.Series, n_splits: int = 5):
    if task_type == "classification":
        min_class_count = int(y_train.value_counts().min())
        effective_splits = max(2, min(n_splits, min_class_count))
        return StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    effective_splits = max(2, min(n_splits, len(y_train)))
    return KFold(n_splits=effective_splits, shuffle=True, random_state=42)


def get_objective_scoring(task_type: str) -> tuple[str, Callable[[float], float], str]:
    """Return sklearn scoring, study optimization direction, and UI metric label."""
    if task_type == "classification":
        return "accuracy", lambda value: value, "Accuracy"
    return "neg_root_mean_squared_error", lambda value: -value, "RMSE"


def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    """Suggest Optuna parameters for a supported model family."""
    if model_name == "LogisticRegression":
        return {
            "model__C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        }
    if model_name == "LinearRegression":
        return {}
    if model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 20),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
    if model_name in {"XGBoostClassifier", "XGBoostRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 10),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if model_name in {"LightGBMClassifier", "LightGBMRegressor"}:
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    return {}


def tune_model(
    *,
    model_name: str,
    estimator: object,
    preprocessor: ColumnTransformer,
    feature_engineering: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    n_trials: int = 20,
) -> TuningResult:
    """Tune one model family with Optuna and return the best-fitted pipeline."""
    scoring, score_transform, metric_name = get_objective_scoring(task_type)
    cv = _get_cv(task_type, y_train, n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        pipeline = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("feature_engineering", clone(feature_engineering)),
                ("model", clone(estimator)),
            ]
        )
        params = suggest_model_params(trial, model_name)
        if params:
            pipeline.set_params(**params)
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        return score_transform(float(np.mean(scores)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("feature_engineering", clone(feature_engineering)),
            ("model", clone(estimator)),
        ]
    )
    best_params = suggest_model_params(optuna.trial.FixedTrial(study.best_params), model_name) if study.best_params else {}
    if best_params:
        best_pipeline.set_params(**best_params)
    best_pipeline.fit(x_train, y_train)

    return TuningResult(
        model_name=model_name,
        pipeline=best_pipeline,
        cv_score=float(study.best_value),
        best_params=study.best_params,
        metric_name=metric_name,
    )
