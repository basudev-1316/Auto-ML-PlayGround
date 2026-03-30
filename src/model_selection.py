from __future__ import annotations

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
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
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


def recommend_model_names(
    *,
    row_count: int,
    column_count: int,
    numeric_feature_count: int,
    categorical_feature_count: int,
    missing_ratio: float,
    high_cardinality_columns: list[str],
    task_type: str,
    shortlist_limit: int,
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
        recommended.append("GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor")

    if high_cardinality_columns:
        recommended.append("CatBoostClassifier" if task_type == "classification" else "CatBoostRegressor")

    return list(dict.fromkeys(recommended))[:shortlist_limit]


def build_model_registry(task_type: str) -> dict[str, object]:
    if task_type == "classification":
        models: dict[str, object] = {
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
    return models
