"""Shared helper utilities for the AutoML project."""

from __future__ import annotations

import json
from datetime import datetime
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
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{sanitized_extension}"
