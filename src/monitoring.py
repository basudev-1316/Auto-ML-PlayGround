"""Lightweight monitoring helpers for project metrics and runtime events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.utils import ensure_directory, save_json_report


LOGGER = get_logger(__name__)
METRICS_PATH = Path(__file__).resolve().parent.parent / "logs" / "metrics.json"
DEFAULT_METRICS = {
    "total_predictions": 0,
    "error_count": 0,
    "last_model_used": None,
}


def load_metrics(metrics_path: str | Path = METRICS_PATH) -> dict[str, Any]:
    """Load monitoring metrics from disk or return the default state."""
    resolved_path = Path(metrics_path)
    if not resolved_path.exists():
        return dict(DEFAULT_METRICS)

    try:
        import json

        with resolved_path.open("r", encoding="utf-8") as metrics_file:
            payload = json.load(metrics_file)
    except Exception:
        LOGGER.exception("Failed to load monitoring metrics from %s", resolved_path)
        return dict(DEFAULT_METRICS)

    return {
        "total_predictions": int(payload.get("total_predictions", 0)),
        "error_count": int(payload.get("error_count", 0)),
        "last_model_used": payload.get("last_model_used"),
    }


def save_metrics(metrics: dict[str, Any], metrics_path: str | Path = METRICS_PATH) -> Path:
    """Persist monitoring metrics to disk."""
    resolved_path = Path(metrics_path)
    ensure_directory(resolved_path.parent)
    return save_json_report(metrics, resolved_path)


def record_prediction_event(
    model_name: str,
    prediction_count: int,
    metrics_path: str | Path = METRICS_PATH,
) -> dict[str, Any]:
    """Update metrics for a successful prediction event."""
    metrics = load_metrics(metrics_path)
    metrics["total_predictions"] = int(metrics.get("total_predictions", 0)) + int(prediction_count)
    metrics["last_model_used"] = model_name
    save_metrics(metrics, metrics_path)
    LOGGER.info(
        "Prediction event recorded. model=%s prediction_count=%d total_predictions=%d",
        model_name,
        prediction_count,
        metrics["total_predictions"],
    )
    return metrics


def record_failure(
    error_message: str,
    metrics_path: str | Path = METRICS_PATH,
) -> dict[str, Any]:
    """Update metrics for a failed event and log the failure."""
    metrics = load_metrics(metrics_path)
    metrics["error_count"] = int(metrics.get("error_count", 0)) + 1
    save_metrics(metrics, metrics_path)
    LOGGER.error(
        "Failure event recorded. error_count=%d message=%s",
        metrics["error_count"],
        error_message,
    )
    return metrics
