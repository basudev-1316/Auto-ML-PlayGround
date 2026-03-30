from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data_preprocessing import (
    build_preprocessor as build_feature_preprocessor,
    decode_target_values,
    detect_task_type,
    encode_target,
    prepare_dataframe,
    split_dataset,
)
from src.evaluate import build_test_metrics as build_metrics, extract_feature_importance
from src.model_selection import build_model_registry, recommend_model_names


@dataclass
class TrainingSummary:
    task_type: str
    best_model_name: str
    best_score: float
    model_path: Path


@dataclass
class DatasetProfile:
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


class AutoMLPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        model_path: str | Path = "models/best_model.pkl",
        task_type_override: str | None = None,
        training_mode: str = "balanced",
        progress_callback=None,
    ):
        self.df = df.copy()
        self.target = target
        self.model_path = Path(model_path)
        self.task_type_override = task_type_override
        self.training_mode = training_mode.lower()
        self.progress_callback = progress_callback
        self.task_type: str | None = None
        self.best_model: Pipeline | None = None
        self.best_model_name: str | None = None
        self.best_score = -np.inf
        self.results: list[dict[str, object]] = []
        self.feature_columns: list[str] = []
        self.available_model_names: list[str] = []
        self.x_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.target_encoder: LabelEncoder | None = None
        self.best_test_predictions: np.ndarray | None = None
        self.best_test_probabilities: np.ndarray | None = None
        self.best_test_metrics: dict[str, float] = {}
        self.best_feature_importance: pd.DataFrame | None = None
        self.dataset_profile: DatasetProfile | None = None
        self.first_round_results: list[dict[str, object]] = []
        self.dataset_signature: dict[str, object] | None = None
        self.model_errors: list[str] = []

    def report_progress(self, phase: str, message: str, current: int | None = None, total: int | None = None) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                "phase": phase,
                "message": message,
                "current": current,
                "total": total,
            }
        )

    def prepare_data(self) -> pd.DataFrame:
        return prepare_dataframe(self.df, self.target)

    def detect_task(self, y: pd.Series) -> None:
        self.task_type = detect_task_type(y, self.task_type_override)
        print(f"Detected Task: {self.task_type}")

    def profile_dataset(self, df: pd.DataFrame) -> DatasetProfile:
        if self.task_type is None:
            raise RuntimeError("Task type must be detected before profiling the dataset.")

        features = df.drop(columns=[self.target])
        target = df[self.target]
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
            task_type=self.task_type,
            shortlist_limit=self.get_shortlist_limit(),
        )
        reasoning = self.build_profile_reasoning(
            row_count=len(df),
            column_count=features.shape[1],
            categorical_feature_count=categorical_features.shape[1],
            missing_ratio=missing_ratio,
            high_cardinality_columns=high_cardinality_columns,
            task_type=self.task_type,
        )

        profile = DatasetProfile(
            row_count=len(df),
            column_count=features.shape[1],
            numeric_feature_count=numeric_features.shape[1],
            categorical_feature_count=categorical_features.shape[1],
            missing_ratio=missing_ratio,
            target_unique_values=int(target.nunique(dropna=True)),
            task_type=self.task_type,
            high_cardinality_columns=high_cardinality_columns,
            recommended_models=recommended_models,
            reasoning=reasoning,
        )
        self.dataset_profile = profile
        self.dataset_signature = {
            "columns": df.columns.tolist(),
            "feature_columns": features.columns.tolist(),
            "target": self.target,
            "row_count": int(len(df)),
        }
        return profile

    def build_profile_reasoning(
        self,
        row_count: int,
        column_count: int,
        categorical_feature_count: int,
        missing_ratio: float,
        high_cardinality_columns: list[str],
        task_type: str,
    ) -> list[str]:
        reasoning = [f"Detected a {task_type} problem with {row_count:,} rows and {column_count} feature columns."]
        if row_count <= 5000:
            reasoning.append("Dataset size is moderate, so kernel and distance-based models can still be considered.")
        else:
            reasoning.append("Dataset size is larger, so boosting and tree ensembles are prioritized over slower local models.")
        if categorical_feature_count > 0:
            reasoning.append(f"Found {categorical_feature_count} categorical feature(s), so tree/boosting models that handle mixed data are prioritized.")
        if high_cardinality_columns:
            reasoning.append(f"High-cardinality categorical columns detected: {', '.join(high_cardinality_columns[:3])}.")
        if missing_ratio > 0:
            reasoning.append(f"Missing-data ratio is {missing_ratio:.2%}, so models robust to preprocessing noise are favored.")
        if row_count >= 15000:
            reasoning.append("Large datasets use a faster benchmarking path with fewer folds and a tighter shortlist.")
        return reasoning

    def sample_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        row_count = len(df)
        if self.training_mode == "full":
            return df

        max_rows = 25000
        if self.training_mode == "fast":
            max_rows = 12000
        elif self.training_mode == "balanced":
            max_rows = 25000

        if row_count <= max_rows:
            return df

        sample_size = max_rows if row_count <= 100000 else min(max_rows + 10000, row_count)
        sample_size = min(sample_size, row_count)

        if self.task_type == "classification":
            sampled_parts: list[pd.DataFrame] = []
            for _, part in df.groupby(self.target, group_keys=False):
                class_sample_size = max(1, round(len(part) / row_count * sample_size))
                sampled_parts.append(part.sample(n=min(len(part), class_sample_size), random_state=42))
            sampled_df = pd.concat(sampled_parts, ignore_index=True)
            if len(sampled_df) > sample_size:
                sampled_df = sampled_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            return sampled_df

        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    def get_shortlist_limit(self) -> int:
        if self.training_mode == "fast":
            return 5
        if self.training_mode == "full":
            return 10
        if self.dataset_profile is None:
            return 8
        if self.dataset_profile.row_count >= 100000:
            return 5
        if self.dataset_profile.row_count >= 25000:
            return 6
        if self.dataset_profile.row_count >= 10000:
            return 7
        return 8

    def split_data(self, df: pd.DataFrame):
        x, x_test, y, y_test = split_dataset(df, self.target, self.task_type or "regression")
        self.feature_columns = x.columns.tolist()
        return x, x_test, y, y_test

    def encode_target(self, y: pd.Series) -> pd.Series:
        encoded_target, encoder = encode_target(y, self.task_type or "regression")
        self.target_encoder = encoder
        return encoded_target

    def decode_target_values(self, values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        return decode_target_values(values, self.task_type or "regression", self.target_encoder)

    def build_preprocessor(self, x: pd.DataFrame) -> ColumnTransformer:
        return build_feature_preprocessor(x)

    def get_models(self) -> dict[str, object]:
        models = build_model_registry(self.task_type or "regression")
        self.available_model_names = list(models.keys())
        return models

    def shortlist_models(self, models: dict[str, object]) -> dict[str, object]:
        if self.dataset_profile is None:
            return models

        shortlisted = {
            model_name: model
            for model_name, model in models.items()
            if model_name in self.dataset_profile.recommended_models
        }
        if shortlisted:
            return shortlisted
        return models

    def get_cv_splits(self, y_train: pd.Series) -> int:
        if self.training_mode == "fast":
            max_cv = 2
        elif self.training_mode == "full":
            max_cv = 5
        else:
            max_cv = 5
        if self.training_mode != "full" and self.dataset_profile is not None:
            if self.dataset_profile.row_count >= 100000:
                max_cv = 2
            elif self.dataset_profile.row_count >= 15000:
                max_cv = 3
            elif self.dataset_profile.row_count >= 5000:
                max_cv = 4
        if self.task_type == "classification":
            min_class_count = int(y_train.value_counts().min())
            if min_class_count < 2:
                return 0
            return max(2, min(max_cv, min_class_count))
        if len(y_train) < 2:
            return 0
        return max(2, min(max_cv, len(y_train)))

    def get_scoring(self) -> str:
        return "accuracy" if self.task_type == "classification" else "r2"

    def evaluate(self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        predictions = model.predict(x_test)
        if self.task_type == "classification":
            return accuracy_score(y_test, predictions)
        return r2_score(y_test, predictions)

    def get_prediction_probabilities(self, model: Pipeline, x_test: pd.DataFrame) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(x_test)
                return np.asarray(probabilities)
            except Exception:
                return None

        if hasattr(model, "decision_function"):
            try:
                decision_scores = model.decision_function(x_test)
                return np.asarray(decision_scores)
            except Exception:
                return None

        return None

    def build_pipeline(self, preprocessor: ColumnTransformer, model: object) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def evaluate_candidate_group(
        self,
        candidate_models: dict[str, object],
        preprocessor: ColumnTransformer,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int,
        scoring: str,
        stage_name: str,
    ) -> list[dict[str, object]]:
        stage_results: list[dict[str, object]] = []
        total_candidates = len(candidate_models)

        for index, (name, model) in enumerate(candidate_models.items(), start=1):
            self.report_progress(
                stage_name,
                f"Training {name} ({index}/{total_candidates})",
                current=index,
                total=total_candidates,
            )
            pipeline = self.build_pipeline(preprocessor, model)
            try:
                if cv_splits >= 2:
                    try:
                        scores = cross_val_score(pipeline, x_train, y_train, cv=cv_splits, scoring=scoring)
                        avg_score = float(np.mean(scores))
                    except Exception:
                        avg_score = None
                else:
                    avg_score = None

                pipeline.fit(x_train, y_train)
                test_score = self.evaluate(pipeline, x_test, y_test)
                if avg_score is None:
                    avg_score = float(test_score)
            except Exception as exc:
                print(f"Skipping {name}: {exc}")
                self.model_errors.append(f"{name}: {exc}")
                continue

            print(f"{name} | CV: {avg_score:.4f} | Test: {test_score:.4f}")

            result_row = {
                "stage": stage_name,
                "model": name,
                "cv_score": round(avg_score, 4),
                "test_score": round(float(test_score), 4),
                "_raw_cv_score": avg_score,
                "_pipeline": pipeline,
            }
            stage_results.append(result_row)

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_model = pipeline
                self.best_model_name = name
                encoded_predictions = pipeline.predict(x_test)
                self.best_test_predictions = self.decode_target_values(encoded_predictions)
                self.best_test_probabilities = self.get_prediction_probabilities(pipeline, x_test)
                self.best_test_metrics = self.build_test_metrics(
                    y_test,
                    encoded_predictions,
                    self.best_test_probabilities,
                )
                self.best_feature_importance = self.extract_feature_importance(pipeline)

        return stage_results

    def build_test_metrics(
        self,
        y_test: pd.Series,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> dict[str, float]:
        return build_metrics(self.task_type or "regression", y_test, predictions, probabilities)

    def extract_feature_importance(self, pipeline: Pipeline) -> pd.DataFrame | None:
        return extract_feature_importance(pipeline)

    def train(self) -> TrainingSummary:
        prepared_df = self.prepare_data()
        y = prepared_df[self.target]
        self.report_progress("profiling", "Analyzing dataset and detecting task")
        self.detect_task(y)
        if self.task_type == "classification":
            prepared_df[self.target] = self.encode_target(y)
        self.profile_dataset(prepared_df)
        training_df = self.sample_training_frame(prepared_df)
        if len(training_df) < len(prepared_df):
            self.report_progress(
                "profiling",
                f"Using a sampled training set of {len(training_df):,} rows for faster benchmarking",
            )
        x_train, x_test, y_train, y_test = self.split_data(training_df)

        preprocessor = self.build_preprocessor(x_train)
        models = self.shortlist_models(self.get_models())
        cv_splits = self.get_cv_splits(y_train)
        scoring = self.get_scoring()

        self.results = []
        self.first_round_results = []
        self.model_errors = []
        self.best_score = -np.inf
        self.best_model = None
        self.best_model_name = None
        self.x_test = x_test.copy()
        self.y_test = y_test.copy()
        self.best_test_predictions = None
        self.best_test_probabilities = None
        self.best_test_metrics = {}
        self.best_feature_importance = None

        if self.task_type == "classification":
            self.y_test = self.decode_target_values(y_test.copy())
        else:
            self.y_test = y_test.copy()

        print("\nStage 1: shortlist round")
        self.report_progress("shortlist", "Building shortlist from dataset profile")
        self.first_round_results = self.evaluate_candidate_group(
            candidate_models=models,
            preprocessor=preprocessor,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv_splits=cv_splits,
            scoring=scoring,
            stage_name="shortlist",
        )
        if not self.first_round_results:
            error_summary = "\n".join(self.model_errors[:5])
            raise RuntimeError(
                "No candidate models could be trained successfully on this dataset."
                + (f"\n\nModel errors:\n{error_summary}" if error_summary else "")
            )

        if self.best_model is None or self.best_model_name is None:
            raise RuntimeError("No model was trained.")

        self.report_progress("finalizing", f"Selecting final winner: {self.best_model_name}")
        print("\nBest model selected!")
        return TrainingSummary(
            task_type=self.task_type or "unknown",
            best_model_name=self.best_model_name,
            best_score=self.best_score,
            model_path=self.model_path,
        )

    def save_model(self) -> Path:
        if self.best_model is None:
            raise RuntimeError("Train the pipeline before saving the model.")

        self.report_progress("saving", "Saving trained model")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_bundle = {
            "model": self.best_model,
            "target": self.target,
            "task_type": self.task_type,
            "best_model_name": self.best_model_name,
            "target_encoder": self.target_encoder,
            "feature_columns": self.feature_columns,
            "test_metrics": self.best_test_metrics,
            "test_probabilities": self.best_test_probabilities,
            "feature_importance": self.best_feature_importance,
            "dataset_profile": self.dataset_profile,
            "dataset_signature": self.dataset_signature,
        }
        joblib.dump(model_bundle, self.model_path)
        print(f"Model saved at {self.model_path}")
        return self.model_path

    def run(self) -> pd.DataFrame:
        self.train()
        self.save_model()
        self.report_progress("done", "Training complete")
        public_first_round = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in self.first_round_results
        ]
        self.results = public_first_round
        results_df = pd.DataFrame(self.results)
        if self.best_model_name is None:
            results_df["best"] = ""
        else:
            results_df["best"] = results_df["model"].apply(
                lambda model_name: "This is the best fit model" if model_name == self.best_model_name else ""
            )
        return results_df
