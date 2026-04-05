"""Tests for shared utility helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils import build_versioned_model_paths


def test_build_versioned_model_paths_returns_model_and_metadata_paths(
    tmp_path: Path,
) -> None:
    model_path, metadata_path = build_versioned_model_paths(tmp_path)

    assert model_path.parent == tmp_path
    assert metadata_path.parent == tmp_path
    assert model_path.suffix == ".pkl"
    assert metadata_path.suffix == ".json"
    assert metadata_path.stem == model_path.stem
    assert model_path.name.startswith("model_")
