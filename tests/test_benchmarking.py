"""Tests for benchmark reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarking import save_benchmark_plot


def test_save_benchmark_plot_creates_png(tmp_path: Path) -> None:
    results_df = pd.DataFrame(
        [
            {"Dataset": "A", "Model": "LogisticRegression", "Score": 0.91, "ProblemType": "classification"},
            {"Dataset": "B", "Model": "LinearRegression", "Score": 2.13, "ProblemType": "regression"},
        ]
    )

    output_path = save_benchmark_plot(results_df, tmp_path / "benchmark_comparison.png")

    assert output_path.exists()
    assert output_path.suffix == ".png"
