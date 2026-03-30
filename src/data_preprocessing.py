from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def prepare_dataframe(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Validate the dataset and expand supported raw columns before training."""
    prepared_df = df.copy()
    if target not in prepared_df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    if "date" in prepared_df.columns:
        parsed_dates = pd.to_datetime(prepared_df["date"], errors="coerce")
        prepared_df["sale_year"] = parsed_dates.dt.year
        prepared_df["sale_month"] = parsed_dates.dt.month
        prepared_df["sale_day"] = parsed_dates.dt.day
        prepared_df = prepared_df.drop(columns=["date"])

    return prepared_df


def detect_task_type(y: pd.Series, task_type_override: str | None = None) -> str:
    """Infer whether the target represents a classification or regression task."""
    unique_values = y.nunique(dropna=True)
    if unique_values < 2:
        raise ValueError(
            "Target column must have at least 2 unique values to train a model. "
            f"Found only {unique_values} unique value(s)."
        )

    if task_type_override in {"classification", "regression"}:
        return task_type_override

    sample_size = max(len(y), 1)
    unique_ratio = unique_values / sample_size
    is_object_like = y.dtype == "object" or str(y.dtype).startswith("category") or str(y.dtype) == "bool"
    numeric_target = pd.api.types.is_numeric_dtype(y)
    non_null_target = pd.Series(y.dropna())
    integer_like_target = bool(
        numeric_target and not non_null_target.empty and np.allclose(non_null_target % 1, 0)
    )

    if is_object_like:
        return "classification"
    if unique_values <= 20 and unique_ratio <= 0.2:
        return "classification"
    if integer_like_target and unique_values <= 50 and unique_ratio <= 0.1:
        return "classification"
    return "regression"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical preprocessing."""
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


def split_dataset(df: pd.DataFrame, target: str, task_type: str):
    """Split the dataset into train and test partitions with safe stratification."""
    x = df.drop(columns=[target])
    y = df[target]
    stratify = y if task_type == "classification" else None
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)


def encode_target(y: pd.Series, task_type: str) -> tuple[pd.Series, LabelEncoder | None]:
    """Encode classification targets while leaving regression targets unchanged."""
    if task_type != "classification":
        return y, None

    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(y)
    return pd.Series(encoded_values, index=y.index, name=y.name), encoder


def decode_target_values(
    values: pd.Series | np.ndarray,
    task_type: str,
    target_encoder: LabelEncoder | None,
) -> pd.Series | np.ndarray:
    """Decode encoded class labels back to their original representation."""
    if task_type != "classification" or target_encoder is None:
        return values

    values_array = np.asarray(values)
    decoded_values = target_encoder.inverse_transform(values_array.astype(int))
    if isinstance(values, pd.Series):
        return pd.Series(decoded_values, index=values.index, name=values.name)
    return decoded_values
