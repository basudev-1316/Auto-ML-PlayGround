from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from math import prod

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
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
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
        progress_callback=None,
    ):
        self.df = df.copy()
        self.target = target
        self.model_path = Path(model_path)
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
        self.second_round_results: list[dict[str, object]] = []
        self.tuning_round_results: list[dict[str, object]] = []
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

        if y.dtype == "object" or unique_values < 10:
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
        if row_count <= 25000:
            return df

        sample_size = 25000 if row_count <= 100000 else 40000
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
        if self.dataset_profile is None:
            return 8
        if self.dataset_profile.row_count >= 100000:
            return 5
        if self.dataset_profile.row_count >= 25000:
            return 6
        if self.dataset_profile.row_count >= 10000:
            return 7
        return 8

    def get_champion_candidate_limit(self) -> int:
        if self.dataset_profile is None:
            return 3
        return 2 if self.dataset_profile.row_count >= 25000 else 3

    def get_tuning_candidate_limit(self) -> int:
        if self.dataset_profile is None:
            return 2
        return 1 if self.dataset_profile.row_count >= 25000 else 2

    def get_tuning_iterations(self) -> int:
        if self.dataset_profile is None:
            return 4
        if self.dataset_profile.row_count >= 100000:
            return 2
        if self.dataset_profile.row_count >= 25000:
            return 3
        return 4

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
        max_cv = 5
        if self.dataset_profile is not None:
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

    def build_tuned_candidates(self, top_candidates: list[dict[str, object]]) -> dict[str, object]:
        tuned_models: dict[str, object] = {}

        for candidate in top_candidates:
            base_name = str(candidate["model"])
            pipeline = candidate["_pipeline"]
            base_model = deepcopy(pipeline.named_steps["model"])

            tuned_models[f"{base_name}__base"] = base_model

            if base_name == "Ridge":
                tuned_models["Ridge__alpha_0.1"] = Ridge(alpha=0.1)
                tuned_models["Ridge__alpha_10"] = Ridge(alpha=10.0)
            elif base_name == "ElasticNet":
                tuned_models["ElasticNet__balanced"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                tuned_models["ElasticNet__lasso_like"] = ElasticNet(alpha=0.05, l1_ratio=0.8, random_state=42)
            elif base_name == "RandomForestRegressor":
                tuned_models["RandomForestRegressor__deep"] = RandomForestRegressor(
                    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
                )
                tuned_models["RandomForestRegressor__regularized"] = RandomForestRegressor(
                    n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
                )
            elif base_name == "ExtraTreesRegressor":
                tuned_models["ExtraTreesRegressor__deep"] = ExtraTreesRegressor(
                    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
                )
                tuned_models["ExtraTreesRegressor__regularized"] = ExtraTreesRegressor(
                    n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
                )
            elif base_name == "GradientBoostingRegressor":
                tuned_models["GradientBoostingRegressor__fast"] = GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.08, random_state=42
                )
                tuned_models["GradientBoostingRegressor__strong"] = GradientBoostingRegressor(
                    n_estimators=250, learning_rate=0.04, max_depth=3, random_state=42
                )
            elif base_name == "HistGradientBoostingRegressor":
                tuned_models["HistGradientBoostingRegressor__fast"] = HistGradientBoostingRegressor(
                    learning_rate=0.08, max_depth=8, random_state=42
                )
                tuned_models["HistGradientBoostingRegressor__regularized"] = HistGradientBoostingRegressor(
                    learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=42
                )
            elif base_name == "SVR":
                tuned_models["SVR__rbf_tight"] = SVR(C=10.0, epsilon=0.1, kernel="rbf")
                tuned_models["SVR__rbf_wide"] = SVR(C=3.0, epsilon=0.2, kernel="rbf")
            elif base_name == "KNeighborsRegressor":
                tuned_models["KNeighborsRegressor__k3"] = KNeighborsRegressor(n_neighbors=3)
                tuned_models["KNeighborsRegressor__distance"] = KNeighborsRegressor(n_neighbors=7, weights="distance")
            elif base_name == "CatBoostRegressor":
                tuned_models["CatBoostRegressor__fast"] = CatBoostRegressor(
                    iterations=150, learning_rate=0.08, depth=6, random_seed=42, verbose=False
                )
                tuned_models["CatBoostRegressor__strong"] = CatBoostRegressor(
                    iterations=300, learning_rate=0.04, depth=8, random_seed=42, verbose=False
                )
            elif base_name == "XGBoostRegressor" and XGBRegressor is not None:
                tuned_models["XGBoostRegressor__fast"] = XGBRegressor(
                    n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42
                )
                tuned_models["XGBoostRegressor__strong"] = XGBRegressor(
                    n_estimators=250, max_depth=7, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=42
                )
            elif base_name == "LightGBMRegressor" and LGBMRegressor is not None:
                tuned_models["LightGBMRegressor__fast"] = LGBMRegressor(
                    n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1
                )
                tuned_models["LightGBMRegressor__strong"] = LGBMRegressor(
                    n_estimators=250, learning_rate=0.04, num_leaves=31, random_state=42, verbose=-1
                )
            elif base_name == "LogisticRegression":
                tuned_models["LogisticRegression__c0.5"] = LogisticRegression(max_iter=1000, C=0.5)
                tuned_models["LogisticRegression__c2"] = LogisticRegression(max_iter=1000, C=2.0)
            elif base_name == "RandomForestClassifier":
                tuned_models["RandomForestClassifier__deep"] = RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                )
                tuned_models["RandomForestClassifier__regularized"] = RandomForestClassifier(
                    n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
                )
            elif base_name == "ExtraTreesClassifier":
                tuned_models["ExtraTreesClassifier__deep"] = ExtraTreesClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                )
                tuned_models["ExtraTreesClassifier__regularized"] = ExtraTreesClassifier(
                    n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
                )
            elif base_name == "GradientBoostingClassifier":
                tuned_models["GradientBoostingClassifier__fast"] = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.08, random_state=42
                )
                tuned_models["GradientBoostingClassifier__strong"] = GradientBoostingClassifier(
                    n_estimators=250, learning_rate=0.04, random_state=42
                )
            elif base_name == "HistGradientBoostingClassifier":
                tuned_models["HistGradientBoostingClassifier__fast"] = HistGradientBoostingClassifier(
                    learning_rate=0.08, max_depth=8, random_state=42
                )
                tuned_models["HistGradientBoostingClassifier__regularized"] = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=42
                )
            elif base_name == "SVM":
                tuned_models["SVM__c0.5"] = SVC(C=0.5, probability=True)
                tuned_models["SVM__c2"] = SVC(C=2.0, probability=True)
            elif base_name == "KNeighborsClassifier":
                tuned_models["KNeighborsClassifier__k3"] = KNeighborsClassifier(n_neighbors=3)
                tuned_models["KNeighborsClassifier__distance"] = KNeighborsClassifier(n_neighbors=7, weights="distance")
            elif base_name == "CatBoostClassifier" and CatBoostClassifier is not None:
                tuned_models["CatBoostClassifier__fast"] = CatBoostClassifier(
                    iterations=150, learning_rate=0.08, depth=6, random_seed=42, verbose=False
                )
                tuned_models["CatBoostClassifier__strong"] = CatBoostClassifier(
                    iterations=300, learning_rate=0.04, depth=8, random_seed=42, verbose=False
                )
            elif base_name == "XGBoostClassifier" and XGBClassifier is not None:
                tuned_models["XGBoostClassifier__fast"] = XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"
                )
                tuned_models["XGBoostClassifier__strong"] = XGBClassifier(
                    n_estimators=250, max_depth=7, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"
                )
            elif base_name == "LightGBMClassifier" and LGBMClassifier is not None:
                tuned_models["LightGBMClassifier__fast"] = LGBMClassifier(
                    n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1
                )
                tuned_models["LightGBMClassifier__strong"] = LGBMClassifier(
                    n_estimators=250, learning_rate=0.04, num_leaves=31, random_state=42, verbose=-1
                )

        return tuned_models

    def get_tuning_space(self, model_name: str) -> dict[str, list[object]]:
        tuning_spaces: dict[str, dict[str, list[object]]] = {
            "Ridge": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 25.0],
            },
            "ElasticNet": {
                "model__alpha": [0.01, 0.05, 0.1, 0.5],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
            "RandomForestRegressor": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "ExtraTreesRegressor": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "GradientBoostingRegressor": {
                "model__n_estimators": [100, 150, 250],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [2, 3, 4],
            },
            "HistGradientBoostingRegressor": {
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [4, 6, 8],
                "model__l2_regularization": [0.0, 0.05, 0.1],
            },
            "SVR": {
                "model__C": [1.0, 3.0, 10.0],
                "model__epsilon": [0.05, 0.1, 0.2],
                "model__kernel": ["rbf"],
            },
            "KNeighborsRegressor": {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
            "CatBoostRegressor": {
                "model__iterations": [120, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__depth": [4, 6, 8],
            },
            "XGBoostRegressor": {
                "model__n_estimators": [120, 200, 300],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.08],
            },
            "LightGBMRegressor": {
                "model__n_estimators": [120, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__num_leaves": [15, 31, 63],
            },
            "LogisticRegression": {
                "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
            "RandomForestClassifier": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "ExtraTreesClassifier": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 10, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "GradientBoostingClassifier": {
                "model__n_estimators": [100, 150, 250],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [2, 3, 4],
            },
            "HistGradientBoostingClassifier": {
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [4, 6, 8],
                "model__l2_regularization": [0.0, 0.05, 0.1],
            },
            "SVM": {
                "model__C": [0.5, 1.0, 2.0, 5.0],
                "model__kernel": ["rbf"],
            },
            "KNeighborsClassifier": {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
            "CatBoostClassifier": {
                "model__iterations": [120, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__depth": [4, 6, 8],
            },
            "XGBoostClassifier": {
                "model__n_estimators": [120, 200, 300],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.08],
            },
            "LightGBMClassifier": {
                "model__n_estimators": [120, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__num_leaves": [15, 31, 63],
            },
        }
        return tuning_spaces.get(model_name, {})

    def evaluate_tuning_round(
        self,
        top_candidates: list[dict[str, object]],
        preprocessor: ColumnTransformer,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int,
        scoring: str,
    ) -> list[dict[str, object]]:
        tuning_results: list[dict[str, object]] = []
        total_candidates = len(top_candidates)

        for index, candidate in enumerate(top_candidates, start=1):
            base_label = str(candidate["model"])
            base_pipeline: Pipeline = candidate["_pipeline"]
            base_model = deepcopy(base_pipeline.named_steps["model"])
            base_name = base_label.split("__")[0]
            tuning_space = self.get_tuning_space(base_name)
            if not tuning_space:
                continue

            total_combinations = prod(len(values) for values in tuning_space.values())
            n_iter = min(self.get_tuning_iterations(), total_combinations)
            self.report_progress(
                "tuning",
                f"Tuning {base_label} ({index}/{total_candidates})",
                current=index,
                total=total_candidates,
            )

            search = RandomizedSearchCV(
                estimator=self.build_pipeline(preprocessor, base_model),
                param_distributions=tuning_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_splits,
                random_state=42,
                n_jobs=1,
            )
            try:
                search.fit(x_train, y_train)
                tuned_pipeline = search.best_estimator_
                test_score = self.evaluate(tuned_pipeline, x_test, y_test)
                avg_score = float(search.best_score_)
            except Exception as exc:
                print(f"Skipping tuning for {base_label}: {exc}")
                continue
            tuned_name = f"{base_label}__tuned"

            print(f"{tuned_name} | CV: {avg_score:.4f} | Test: {test_score:.4f}")

            result_row = {
                "stage": "tuning",
                "model": tuned_name,
                "cv_score": round(avg_score, 4),
                "test_score": round(float(test_score), 4),
                "_raw_cv_score": avg_score,
                "_pipeline": tuned_pipeline,
            }
            tuning_results.append(result_row)

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_model = tuned_pipeline
                self.best_model_name = tuned_name
                encoded_predictions = tuned_pipeline.predict(x_test)
                self.best_test_predictions = self.decode_target_values(encoded_predictions)
                self.best_test_probabilities = self.get_prediction_probabilities(tuned_pipeline, x_test)
                self.best_test_metrics = self.build_test_metrics(
                    y_test,
                    encoded_predictions,
                    self.best_test_probabilities,
                )
                self.best_feature_importance = self.extract_feature_importance(tuned_pipeline)

        return tuning_results

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
        self.second_round_results = []
        self.tuning_round_results = []
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

        top_candidates = sorted(
            self.first_round_results,
            key=lambda row: row["_raw_cv_score"],
            reverse=True,
        )[: self.get_champion_candidate_limit()]
        tuned_candidates = self.build_tuned_candidates(top_candidates)
        if tuned_candidates:
            print("\nStage 2: champion round")
            self.report_progress("champion", "Preparing champion round from top models")
            champion_cv = max(2, min(3, cv_splits))
            self.second_round_results = self.evaluate_candidate_group(
                candidate_models=tuned_candidates,
                preprocessor=preprocessor,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                cv_splits=champion_cv,
                scoring=scoring,
                stage_name="champion",
            )

            top_tuning_candidates = sorted(
                self.second_round_results,
                key=lambda row: row["_raw_cv_score"],
                reverse=True,
            )[: self.get_tuning_candidate_limit()]
            if top_tuning_candidates:
                print("\nStage 3: hyperparameter tuning")
                self.report_progress("tuning", "Running hyperparameter tuning on top champion models")
                tuning_cv = max(2, min(3, cv_splits))
                self.tuning_round_results = self.evaluate_tuning_round(
                    top_candidates=top_tuning_candidates,
                    preprocessor=preprocessor,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    cv_splits=tuning_cv,
                    scoring=scoring,
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
        public_second_round = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in self.second_round_results
        ]
        public_tuning_round = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in self.tuning_round_results
        ]
        self.results = public_first_round + public_second_round + public_tuning_round
        results_df = pd.DataFrame(self.results)
        if self.best_model_name is None:
            results_df["best"] = ""
        else:
            results_df["best"] = results_df["model"].apply(
                lambda model_name: "This is the best fit model" if model_name == self.best_model_name else ""
            )
        return results_df
