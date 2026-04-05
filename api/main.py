"""FastAPI inference service for AutoML Playground."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.monitoring import METRICS_PATH, record_failure, record_prediction_event
from src.utils import find_latest_model_path


LOGGER = get_logger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"


class PredictionRequest(BaseModel):
    """Prediction payload containing one or more input records."""

    records: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Prediction response returned by the inference API."""

    model_name: str
    predictions: list[Any]
    model_path: str


def load_model_bundle() -> dict[str, Any]:
    """Load the most recent saved model bundle from the models directory."""
    model_path = find_latest_model_path(MODELS_DIR)
    if model_path is None:
        raise FileNotFoundError("No saved model artifacts found in the models directory.")

    model_bundle = joblib.load(model_path)
    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError("Saved model artifact has an unexpected format.")

    model_bundle["model_path"] = str(model_path)
    LOGGER.info("Loaded model bundle from %s", model_path)
    return model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once during application startup."""
    try:
        app.state.model_bundle = load_model_bundle()
        app.state.startup_error = None
    except Exception as exc:
        app.state.model_bundle = None
        app.state.startup_error = str(exc)
        LOGGER.exception("API startup failed while loading model bundle.")
        record_failure(f"api_startup_failure: {exc}")
    yield


app = FastAPI(title="AutoML Playground API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health and model readiness information."""
    model_bundle = getattr(app.state, "model_bundle", None)
    startup_error = getattr(app.state, "startup_error", None)

    if model_bundle is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": startup_error,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": model_bundle.get("model_name"),
        "problem_type": model_bundle.get("problem_type"),
        "target_column": model_bundle.get("target_column"),
        "model_path": model_bundle.get("model_path"),
        "metrics_path": str(METRICS_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Generate predictions for the supplied records."""
    model_bundle = getattr(app.state, "model_bundle", None)
    if model_bundle is None:
        error_message = getattr(app.state, "startup_error", "Model is not loaded.")
        record_failure(f"api_predict_without_model: {error_message}")
        raise HTTPException(status_code=503, detail=error_message)

    model = model_bundle["model"]
    records_df = pd.DataFrame(payload.records)

    try:
        predictions = model.predict(records_df).tolist()
    except Exception as exc:
        LOGGER.exception("Prediction failed for API request.")
        record_failure(f"api_prediction_failure: {exc}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    model_name = str(model_bundle.get("model_name", "unknown_model"))
    record_prediction_event(model_name, len(predictions))
    LOGGER.info(
        "API prediction completed. model=%s prediction_count=%d",
        model_name,
        len(predictions),
    )

    return PredictionResponse(
        model_name=model_name,
        predictions=predictions,
        model_path=str(model_bundle.get("model_path", "")),
    )
