"""End-to-end training orchestration for the production AutoML system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import (
    build_preprocessor,
    decode_target_values,
    detect_task_type,
    encode_target,
    prepare_dataframe,
    split_dataset,
)
from src.evaluate import (
    build_test_metrics,
    extract_feature_importance,
    save_confusion_matrix,
    save_model_comparison,
)
from src.explainability import build_shap_summary, save_shap_bar_plot
from src.feature_engineering import build_feature_engineering_pipeline
from src.hyperparameter_tuning import get_objective_scoring, tune_model
from src.model_selection import build_model_registry, recommend_model_names


@dataclass
class DatasetProfile:
    """Summary of the active dataset used to guide model selection."""

    row_count: int
    column_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    missing_ratio: float
    target_unique_values: int
    task_type: str
    high_cardinality_columns: list[str]
    recommended_models: list[str]
    reasoning: list[str]


def _report_progress(progress_callback, phase: str, message: str, current: int | None = None, total: int | None = None) -> None:
    if progress_callback is None:
        return
    progress_callback({"phase": phase, "message": message, "current": current, "total": total})


def _build_dataset_profile(df: pd.DataFrame, target_column: str, task_type: str, shortlist_limit: int) -> DatasetProfile:
    features = df.drop(columns=[target_column])
    target = df[target_column]
    numeric_features = features.select_dtypes(include=["number"])
    categorical_features = features.select_dtypes(exclude=["number"])
    missing_ratio = float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1))
    high_cardinality_columns = [
        column
        for column in categorical_features.columns
        if categorical_features[column].nunique(dropna=True) > min(50, max(int(len(df) * 0.05), 10))
    ]
    recommended_models = recommend_model_names(
        row_count=len(df),
        column_count=features.shape[1],
        numeric_feature_count=numeric_features.shape[1],
        categorical_feature_count=categorical_features.shape[1],
        missing_ratio=missing_ratio,
        high_cardinality_columns=high_cardinality_columns,
        task_type=task_type,
        shortlist_limit=shortlist_limit,
    )
    reasoning = [
        f"Detected a {task_type} problem with {len(df):,} rows and {features.shape[1]} feature columns.",
        f"Categorical features: {categorical_features.shape[1]}, numeric features: {numeric_features.shape[1]}.",
        f"Missing ratio: {missing_ratio:.2%}.",
    ]
    return DatasetProfile(
        row_count=len(df),
        column_count=features.shape[1],
        numeric_feature_count=numeric_features.shape[1],
        categorical_feature_count=categorical_features.shape[1],
        missing_ratio=missing_ratio,
        target_unique_values=int(target.nunique(dropna=True)),
        task_type=task_type,
        high_cardinality_columns=high_cardinality_columns,
        recommended_models=recommended_models,
        reasoning=reasoning,
    )


def _get_mode_settings(training_mode: str) -> dict[str, int | bool]:
    normalized = training_mode.lower()
    if normalized == "fast":
        return {"shortlist_limit": 3, "trials": 10, "enable_polynomial": False}
    if normalized == "full":
        return {"shortlist_limit": 4, "trials": 30, "enable_polynomial": True}
    return {"shortlist_limit": 4, "trials": 20, "enable_polynomial": False}


def run_training(
    df: pd.DataFrame,
    target_column: str,
    model_path: str | Path,
    task_type_override: str | None = None,
    training_mode: str = "Balanced",
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train all candidate models, tune them, compare them, and save the winner."""

    model_path = Path(model_path)
    reports_dir = model_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    prepared_df = prepare_dataframe(df, target_column)

    _report_progress(progress_callback, "profiling", "Analyzing dataset")
    task_type = detect_task_type(prepared_df[target_column], task_type_override)
    if task_type == "classification":
        prepared_df[target_column], target_encoder = encode_target(prepared_df[target_column], task_type)
    else:
        target_encoder = None

    mode_settings = _get_mode_settings(training_mode)
    dataset_profile = _build_dataset_profile(
        prepared_df,
        target_column,
        task_type,
        shortlist_limit=int(mode_settings["shortlist_limit"]),
    )

    x_train, x_test, y_train, y_test = split_dataset(prepared_df, target_column, task_type)
    preprocessor = build_preprocessor(x_train)
    feature_engineering = build_feature_engineering_pipeline(
        task_type,
        enable_polynomial=bool(mode_settings["enable_polynomial"]),
        use_feature_selection=True,
        k_best="all",
    )

    model_registry = build_model_registry(task_type)
    shortlisted_models = {
        name: model_registry[name]
        for name in dataset_profile.recommended_models
        if name in model_registry
    }
    if not shortlisted_models:
        shortlisted_models = model_registry

    _report_progress(progress_callback, "shortlist", "Tuning candidate models", current=0, total=len(shortlisted_models))

    scoring_name, _, metric_label = get_objective_scoring(task_type)
    results: list[dict[str, object]] = []
    best_bundle: dict[str, object] | None = None

    for index, (model_name, estimator) in enumerate(shortlisted_models.items(), start=1):
        _report_progress(progress_callback, "shortlist", f"Tuning {model_name}", current=index, total=len(shortlisted_models))
        tuning_result = tune_model(
            model_name=model_name,
            estimator=estimator,
            preprocessor=preprocessor,
            feature_engineering=feature_engineering,
            x_train=x_train,
            y_train=y_train,
            task_type=task_type,
            n_trials=int(mode_settings["trials"]),
        )

        raw_predictions = tuning_result.pipeline.predict(x_test)
        decoded_predictions = decode_target_values(raw_predictions, task_type, target_encoder)
        probabilities = None
        if hasattr(tuning_result.pipeline, "predict_proba"):
            try:
                probabilities = tuning_result.pipeline.predict_proba(x_test)
            except Exception:
                probabilities = None
        metrics = build_test_metrics(task_type, y_test, raw_predictions, probabilities)
        test_score = metrics["accuracy"] if task_type == "classification" else metrics["rmse"]

        result_row = {
            "stage": "shortlist",
            "model": model_name,
            "cv_score": round(tuning_result.cv_score, 4),
            "test_score": round(float(test_score), 4),
            "best": "",
            "metric_name": metric_label,
        }
        results.append(result_row)

        if best_bundle is None:
            best_bundle = {
                "model_name": model_name,
                "pipeline": tuning_result.pipeline,
                "cv_score": tuning_result.cv_score,
                "metrics": metrics,
                "predictions": decoded_predictions,
                "raw_predictions": raw_predictions,
                "probabilities": probabilities,
            }
        else:
            better = tuning_result.cv_score > best_bundle["cv_score"] if task_type == "classification" else tuning_result.cv_score < best_bundle["cv_score"]
            if better:
                best_bundle = {
                    "model_name": model_name,
                    "pipeline": tuning_result.pipeline,
                    "cv_score": tuning_result.cv_score,
                    "metrics": metrics,
                    "predictions": decoded_predictions,
                    "raw_predictions": raw_predictions,
                    "probabilities": probabilities,
                }

    if best_bundle is None:
        raise RuntimeError("No candidate models could be trained successfully on this dataset.")

    for row in results:
        if row["model"] == best_bundle["model_name"]:
            row["best"] = "This is the best fit model"

    results_df = pd.DataFrame(results)
    save_model_comparison(results_df, reports_dir, metric_label)
    if task_type == "classification":
        decoded_y_test = decode_target_values(y_test.copy(), task_type, target_encoder)
        save_confusion_matrix(decoded_y_test, best_bundle["predictions"], reports_dir)
    else:
        decoded_y_test = y_test.copy()

    feature_importance = extract_feature_importance(best_bundle["pipeline"])
    shap_summary = build_shap_summary(best_bundle["pipeline"], x_test.head(min(150, len(x_test))))
    if shap_summary is not None and not shap_summary.empty:
        shap_summary.to_csv(reports_dir / "shap_summary.csv", index=False)
        save_shap_bar_plot(shap_summary, reports_dir / "shap_summary.png")

    _report_progress(progress_callback, "finalizing", f"Selecting final winner: {best_bundle['model_name']}")
    model_bundle = {
        "model": best_bundle["pipeline"],
        "target": target_column,
        "task_type": task_type,
        "best_model_name": best_bundle["model_name"],
        "target_encoder": target_encoder,
        "feature_columns": x_train.columns.tolist(),
        "test_metrics": best_bundle["metrics"],
        "test_probabilities": best_bundle["probabilities"],
        "feature_importance": feature_importance,
        "dataset_profile": dataset_profile,
        "dataset_signature": {
            "columns": prepared_df.columns.tolist(),
            "feature_columns": x_train.columns.tolist(),
            "target": target_column,
            "row_count": int(len(prepared_df)),
        },
    }
    _report_progress(progress_callback, "saving", "Saving trained model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    training_summary = {
        "best_model_name": best_bundle["model_name"],
        "task_type": task_type,
        "metric_name": metric_label,
        "cv_score": best_bundle["cv_score"],
        "test_metrics": best_bundle["metrics"],
        "recommended_models": dataset_profile.recommended_models,
        "row_count": dataset_profile.row_count,
        "column_count": dataset_profile.column_count,
        "training_mode": training_mode,
    }
    with (reports_dir / "training_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(training_summary, summary_file, indent=2)
    _report_progress(progress_callback, "done", "Training complete")

    evaluation_artifacts = {
        "task_type": task_type,
        "model": best_bundle["pipeline"],
        "x_test": x_test,
        "y_test": decoded_y_test,
        "predictions": best_bundle["predictions"],
        "probabilities": best_bundle["probabilities"],
        "metrics": best_bundle["metrics"],
        "best_model_name": best_bundle["model_name"],
        "feature_importance": feature_importance,
        "dataset_profile": dataset_profile,
        "metric_name": metric_label,
        "shap_summary": shap_summary,
    }
    return results_df, evaluation_artifacts
