from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


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
        df = self.df.copy()
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' was not found in the dataset.")

        if "date" in df.columns:
            parsed_dates = pd.to_datetime(df["date"], errors="coerce")
            df["sale_year"] = parsed_dates.dt.year
            df["sale_month"] = parsed_dates.dt.month
            df["sale_day"] = parsed_dates.dt.day
            df = df.drop(columns=["date"])

        return df

    def detect_task(self, y: pd.Series) -> None:
        unique_values = y.nunique(dropna=True)
        if unique_values < 2:
            raise ValueError(
                f"Target column '{self.target}' must have at least 2 unique values to train a model. "
                f"Found only {unique_values} unique value(s)."
            )

        if self.task_type_override in {"classification", "regression"}:
            self.task_type = self.task_type_override
            print(f"Detected Task: {self.task_type} (manual override)")
            return

        sample_size = max(len(y), 1)
        unique_ratio = unique_values / sample_size
        is_object_like = y.dtype == "object" or str(y.dtype).startswith("category") or str(y.dtype) == "bool"
        numeric_target = pd.api.types.is_numeric_dtype(y)
        non_null_target = pd.Series(y.dropna())
        integer_like_target = bool(
            numeric_target and not non_null_target.empty and np.allclose(non_null_target % 1, 0)
        )

        if is_object_like:
            self.task_type = "classification"
        elif unique_values <= 20 and unique_ratio <= 0.2:
            self.task_type = "classification"
        elif integer_like_target and unique_values <= 50 and unique_ratio <= 0.1:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

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

        recommended_models = self.recommend_model_names(
            row_count=len(df),
            column_count=features.shape[1],
            numeric_feature_count=numeric_features.shape[1],
            categorical_feature_count=categorical_features.shape[1],
            missing_ratio=missing_ratio,
            high_cardinality_columns=high_cardinality_columns,
            task_type=self.task_type,
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

    def recommend_model_names(
        self,
        row_count: int,
        column_count: int,
        numeric_feature_count: int,
        categorical_feature_count: int,
        missing_ratio: float,
        high_cardinality_columns: list[str],
        task_type: str,
    ) -> list[str]:
        if task_type == "classification":
            recommended = [
                "LogisticRegression",
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "HistGradientBoostingClassifier",
            ]

            if row_count <= 5000 and column_count <= 40:
                recommended.extend(["SVM", "KNeighborsClassifier"])
            if categorical_feature_count > 0:
                recommended.append("CatBoostClassifier")
            if row_count >= 2000:
                recommended.extend(["LightGBMClassifier", "XGBoostClassifier"])
            if missing_ratio > 0.02:
                recommended.append("GradientBoostingClassifier")
        else:
            recommended = [
                "Ridge",
                "RandomForestRegressor",
                "ExtraTreesRegressor",
                "HistGradientBoostingRegressor",
            ]

            if row_count <= 5000 and column_count <= 40:
                recommended.extend(["SVR", "KNeighborsRegressor", "ElasticNet"])
            else:
                recommended.append("ElasticNet")
            if categorical_feature_count > 0:
                recommended.append("CatBoostRegressor")
            if row_count >= 2000:
                recommended.extend(["LightGBMRegressor", "XGBoostRegressor"])
            if missing_ratio > 0.02:
                recommended.append("GradientBoostingRegressor")

        if categorical_feature_count == 0 and numeric_feature_count >= 10:
            if task_type == "classification":
                recommended.append("GradientBoostingClassifier")
            else:
                recommended.append("GradientBoostingRegressor")

        if high_cardinality_columns:
            if task_type == "classification":
                recommended.append("CatBoostClassifier")
            else:
                recommended.append("CatBoostRegressor")

        deduped_recommendations = list(dict.fromkeys(recommended))
        return deduped_recommendations[: self.get_shortlist_limit()]

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
        x = df.drop(columns=[self.target])
        y = df[self.target]
        self.feature_columns = x.columns.tolist()
        stratify = y if self.task_type == "classification" else None
        return train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)

    def encode_target(self, y: pd.Series) -> pd.Series:
        if self.task_type != "classification":
            return y

        self.target_encoder = LabelEncoder()
        encoded_values = self.target_encoder.fit_transform(y)
        return pd.Series(encoded_values, index=y.index, name=y.name)

    def decode_target_values(self, values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        if self.task_type != "classification" or self.target_encoder is None:
            return values

        values_array = np.asarray(values)
        decoded_values = self.target_encoder.inverse_transform(values_array.astype(int))
        if isinstance(values, pd.Series):
            return pd.Series(decoded_values, index=values.index, name=values.name)
        return decoded_values

    def build_preprocessor(self, x: pd.DataFrame) -> ColumnTransformer:
        numeric_features = x.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            [
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

    def get_models(self) -> dict[str, object]:
        models: dict[str, object]
        if self.task_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=120, random_state=42),
                "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=120, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }
            if XGBClassifier is not None:
                models["XGBoostClassifier"] = XGBClassifier(
                    n_estimators=120,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric="logloss",
                )
            if LGBMClassifier is not None:
                models["LightGBMClassifier"] = LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1,
                )
            if CatBoostClassifier is not None:
                models["CatBoostClassifier"] = CatBoostClassifier(
                    iterations=120,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                )
            self.available_model_names = list(models.keys())
            return models

        models = {
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
        }
        if XGBRegressor is not None:
            models["XGBoostRegressor"] = XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        if LGBMRegressor is not None:
            models["LightGBMRegressor"] = LGBMRegressor(
                n_estimators=120,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        if CatBoostRegressor is not None:
            models["CatBoostRegressor"] = CatBoostRegressor(
                iterations=120,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=False,
            )
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
            return max(2, min(max_cv, min_class_count))
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
                scores = cross_val_score(pipeline, x_train, y_train, cv=cv_splits, scoring=scoring)
                pipeline.fit(x_train, y_train)
                test_score = self.evaluate(pipeline, x_test, y_test)
                avg_score = float(np.mean(scores))
            except Exception as exc:
                print(f"Skipping {name}: {exc}")
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
        if self.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            }
            unique_classes = pd.Series(y_test).dropna().unique()
            if len(unique_classes) == 2 and probabilities is not None:
                try:
                    if getattr(probabilities, "ndim", 1) == 2 and probabilities.shape[1] >= 2:
                        positive_scores = probabilities[:, 1]
                    else:
                        positive_scores = probabilities
                    metrics["roc_auc"] = float(roc_auc_score(y_test, positive_scores))
                except Exception:
                    pass
            return metrics

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        return {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": rmse,
        }

    def extract_feature_importance(self, pipeline: Pipeline) -> pd.DataFrame | None:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return None

        importances: np.ndarray | None = None
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_, dtype=float)
            if coefficients.ndim > 1:
                importances = np.mean(np.abs(coefficients), axis=0)
            else:
                importances = np.abs(coefficients)

        if importances is None or len(importances) != len(feature_names):
            return None

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        importance_df = importance_df.sort_values("importance", ascending=False).head(20).reset_index(drop=True)
        return importance_df

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
            raise RuntimeError("No candidate models could be trained successfully on this dataset.")

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
