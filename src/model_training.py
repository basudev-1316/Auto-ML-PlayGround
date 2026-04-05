"""Model training utilities for the AutoML engine."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR

from src.data_preprocessing import build_preprocessing_pipeline
from src.logger import get_logger

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Container returned by the training service."""

    results: dict[str, dict[str, float]]
    trained_models: dict[str, Pipeline]


def build_model_registry(problem_type: str) -> dict[str, Any]:
    """Build the production model registry for the requested problem type."""
    if problem_type == "classification":
        models: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
        }
        if XGBClassifier is not None:
            models["XGBoostClassifier"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
        if LGBMClassifier is not None:
            models["LightGBMClassifier"] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        return models

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "SVR": SVR(),
        "KNNRegressor": KNeighborsRegressor(),
    }
    if XGBRegressor is not None:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    if LGBMRegressor is not None:
        models["LightGBMRegressor"] = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    return models


def build_cv_strategy(problem_type: str, target: pd.Series, cv_folds: int = 5) -> Any:
    """Build a safe cross-validation strategy for the current problem type."""
    if problem_type == "classification":
        min_class_count = int(target.value_counts().min())
        effective_folds = max(2, min(cv_folds, min_class_count))
        return StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)

    effective_folds = max(2, min(cv_folds, len(target)))
    return KFold(n_splits=effective_folds, shuffle=True, random_state=42)


def get_scoring_name(problem_type: str) -> str:
    """Return the sklearn scoring string for the requested problem type."""
    if problem_type == "classification":
        return "accuracy"
    return "neg_root_mean_squared_error"


def prepare_target_for_training(
    target: pd.Series,
    problem_type: str,
) -> tuple[pd.Series, LabelEncoder | None]:
    """Encode classification targets for model families that require numeric labels."""
    if problem_type != "classification":
        return target, None

    encoder = LabelEncoder()
    encoded_target = pd.Series(
        encoder.fit_transform(target),
        index=target.index,
        name=target.name,
    )
    return encoded_target, encoder


def build_training_pipeline(
    preprocessing_pipeline: ColumnTransformer,
    estimator: Any,
) -> Pipeline:
    """Compose preprocessing and estimator steps into one sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessing_pipeline)),
            ("model", clone(estimator)),
        ]
    )


def train_model_candidate(
    model_name: str,
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    problem_type: str,
    *,
    cv_folds: int = 5,
) -> tuple[dict[str, float], Pipeline]:
    """Train and evaluate one candidate model family."""
    LOGGER.info("Training started for model '%s' (%s).", model_name, problem_type)
    preprocessing_pipeline = build_preprocessing_pipeline(features)
    training_pipeline = build_training_pipeline(preprocessing_pipeline, estimator)
    scoring = get_scoring_name(problem_type)
    cv_strategy = build_cv_strategy(problem_type, target, cv_folds=cv_folds)

    try:
        start_time = perf_counter()
        cv_scores = cross_val_score(
            training_pipeline,
            features,
            target,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=1,
        )
        training_pipeline.fit(features, target)
        train_time = perf_counter() - start_time
    except Exception:
        LOGGER.exception("Training failed for model '%s'.", model_name)
        raise

    if problem_type == "regression":
        cv_scores = -cv_scores

    result = {
        "mean_score": float(np.mean(cv_scores)),
        "std_score": float(np.std(cv_scores)),
        "train_time": float(train_time),
    }
    LOGGER.info(
        "Model '%s' completed. mean_score=%.4f std_score=%.4f train_time=%.4fs",
        model_name,
        result["mean_score"],
        result["std_score"],
        result["train_time"],
    )
    return result, training_pipeline


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    *,
    cv_folds: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, Pipeline]]:
    """Train every model in the registry and return metrics plus fitted pipelines."""
    LOGGER.info("Training run started for problem type '%s' with %d rows.", problem_type, len(X))
    model_registry = build_model_registry(problem_type)
    training_target, _ = prepare_target_for_training(y, problem_type)

    results: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Pipeline] = {}

    for model_name, estimator in model_registry.items():
        model_result, trained_pipeline = train_model_candidate(
            model_name=model_name,
            estimator=estimator,
            features=X,
            target=training_target,
            problem_type=problem_type,
            cv_folds=cv_folds,
        )
        results[model_name] = model_result
        trained_models[model_name] = trained_pipeline

    LOGGER.info("Training run finished for problem type '%s'. %d models trained.", problem_type, len(results))
    return results, trained_models
