"""Tests for the FastAPI inference service helpers."""

from __future__ import annotations

import joblib
from pathlib import Path

from src.utils import find_latest_model_path


def test_find_latest_model_path_prefers_versioned_models(tmp_path: Path) -> None:
    older_model = tmp_path / "model_20260101_010101.pkl"
    newer_model = tmp_path / "model_20260102_020202.pkl"
    fallback_model = tmp_path / "best_model.pkl"

    older_model.write_bytes(b"older")
    newer_model.write_bytes(b"newer")
    fallback_model.write_bytes(b"fallback")

    assert find_latest_model_path(tmp_path) == newer_model


def test_find_latest_model_path_falls_back_to_best_model(tmp_path: Path) -> None:
    fallback_model = tmp_path / "best_model.pkl"
    fallback_model.write_bytes(b"fallback")

    assert find_latest_model_path(tmp_path) == fallback_model


def test_find_latest_model_path_returns_none_when_no_models_exist(tmp_path: Path) -> None:
    assert find_latest_model_path(tmp_path) is None
