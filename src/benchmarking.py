"""Benchmark execution utilities for running AutoML on multiple datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from src.evaluation import ensure_reports_dir
from src.logger import get_logger
from src.model_selection import select_best_model
from src.model_training import train_all_models
from src.problem_detection import detect_problem_type


LOGGER = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class BenchmarkDataset:
    """Container describing one benchmark dataset and its target column."""

    name: str
    dataframe: pd.DataFrame
    target_column: str
    cv_folds: int = 3


def load_benchmark_datasets() -> list[BenchmarkDataset]:
    """Load a small benchmark pack spanning classification and regression."""
    local_classification = pd.read_csv(PROJECT_ROOT / "data" / "sample_dataset.csv")
    local_regression = (
        pd.read_csv(PROJECT_ROOT / "data" / "data.csv")
        .sample(n=1200, random_state=42)
        .reset_index(drop=True)
    )

    iris_bundle = load_iris(as_frame=True)
    iris_df = iris_bundle.frame.copy()
    iris_df["target"] = iris_df["target"].map(
        {index: name for index, name in enumerate(iris_bundle.target_names)}
    )

    breast_cancer_bundle = load_breast_cancer(as_frame=True)
    breast_cancer_df = breast_cancer_bundle.frame.copy()
    breast_cancer_df["target"] = breast_cancer_df["target"].map(
        {index: name for index, name in enumerate(breast_cancer_bundle.target_names)}
    )

    diabetes_bundle = load_diabetes(as_frame=True)
    diabetes_df = diabetes_bundle.frame.copy()

    return [
        BenchmarkDataset("Sample Purchase", local_classification, "buy", cv_folds=3),
        BenchmarkDataset("Seattle Housing", local_regression, "price", cv_folds=3),
        BenchmarkDataset("Iris", iris_df, "target", cv_folds=3),
        BenchmarkDataset("Breast Cancer", breast_cancer_df, "target", cv_folds=3),
        BenchmarkDataset("Diabetes", diabetes_df, "target", cv_folds=3),
    ]


def run_benchmark_dataset(dataset: BenchmarkDataset) -> dict[str, object]:
    """Train the AutoML system on one dataset and return the winning model result."""
    detection = detect_problem_type(dataset.dataframe, dataset.target_column)
    features = dataset.dataframe.drop(columns=[dataset.target_column])
    target = dataset.dataframe[dataset.target_column]

    results, trained_models = train_all_models(
        features,
        target,
        detection.problem_type,
        cv_folds=dataset.cv_folds,
    )
    selection_ready_results = {
        model_name: {
            **metrics,
            "metric_name": "Accuracy" if detection.problem_type == "classification" else "RMSE",
            "model_object": trained_models[model_name],
        }
        for model_name, metrics in results.items()
    }
    best_model_name, _best_model_object, _reasoning = select_best_model(selection_ready_results)
    best_score = float(selection_ready_results[best_model_name]["mean_score"])

    benchmark_result = {
        "Dataset": dataset.name,
        "Model": best_model_name,
        "Score": best_score,
        "Metric": selection_ready_results[best_model_name]["metric_name"],
        "ProblemType": detection.problem_type,
    }
    LOGGER.info("Benchmark completed for %s: %s (%.4f)", dataset.name, best_model_name, best_score)
    return benchmark_result


def save_benchmark_plot(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a grouped benchmark comparison plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Dataset", y="Score", hue="ProblemType", palette="viridis")
    plt.title("Benchmark Results Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info("Saved benchmark comparison plot to %s", output_path)
    return output_path


def run_all_benchmarks(reports_dir: str | Path = REPORTS_DIR) -> tuple[pd.DataFrame, Path, Path]:
    """Run the benchmark suite and persist the CSV summary plus comparison plot."""
    output_dir = ensure_reports_dir(reports_dir)
    benchmark_rows = [run_benchmark_dataset(dataset) for dataset in load_benchmark_datasets()]
    results_df = pd.DataFrame(benchmark_rows)

    csv_path = output_dir / "results.csv"
    results_df[["Dataset", "Model", "Score"]].to_csv(csv_path, index=False)
    LOGGER.info("Saved benchmark results table to %s", csv_path)

    plot_path = save_benchmark_plot(results_df, output_dir / "benchmark_comparison.png")
    return results_df, csv_path, plot_path
