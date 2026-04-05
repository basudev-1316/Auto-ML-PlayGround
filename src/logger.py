"""Centralized logging utilities for the AutoML project."""

from __future__ import annotations

import logging
from pathlib import Path


LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logging() -> Path:
    """Configure shared file logging for the project and return the log path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()

    file_handler_exists = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == LOG_FILE
        for handler in root_logger.handlers
    )

    if not file_handler_exists:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    return LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger backed by the centralized file logger."""
    configure_logging()
    return logging.getLogger(name)
