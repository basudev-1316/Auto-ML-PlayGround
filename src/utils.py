"""Shared helper utilities for the AutoML project."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the root directory of the AutoML project."""
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return the resolved path."""
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def save_json_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a JSON-serializable report payload to disk."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2, default=str)
    return resolved_path


def build_timestamped_filename(prefix: str, extension: str) -> str:
    """Build a timestamped filename for model and report artifacts."""
    sanitized_extension = extension.lstrip(".")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{sanitized_extension}"


def build_versioned_model_paths(models_dir: str | Path) -> tuple[Path, Path]:
    """Return coordinated versioned paths for a model artifact and its metadata."""
    resolved_models_dir = ensure_directory(models_dir)
    model_filename = build_timestamped_filename("model", "pkl")
    model_path = resolved_models_dir / model_filename
    metadata_path = model_path.with_suffix(".json")
    return model_path, metadata_path


def find_latest_model_path(models_dir: str | Path) -> Path | None:
    """Return the latest saved model path, preferring versioned artifacts."""
    resolved_models_dir = Path(models_dir)
    versioned_models = sorted(resolved_models_dir.glob("model_*.pkl"), reverse=True)
    if versioned_models:
        return versioned_models[0]

    fallback_model = resolved_models_dir / "best_model.pkl"
    if fallback_model.exists():
        return fallback_model

    return None
