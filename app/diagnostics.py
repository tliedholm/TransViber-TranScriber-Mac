"""Rotating diagnostic log.

The shipped app is a windowed bundle with no console, so an exception that
only reaches a QMessageBox as str(exc) leaves nothing to investigate. Every
engine failure is written here with its traceback and the context needed to
reproduce it — model, requested device, backend actually used, track, and the
versions of the packages that do the work.

User-facing dialogs stay short; this file holds the detail.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

LOG_NAME = "transviber.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_logger: logging.Logger | None = None


def log_dir() -> Path:
    """Where the log lives: ~/Library/Logs/TransViber TransScriber on macOS."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "TransViber TransScriber"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "transviber"


def log_path() -> Path:
    return log_dir() / LOG_NAME


def _package_versions() -> str:
    from importlib.metadata import version

    names = ("faster-whisper", "ctranslate2", "mlx", "mlx-whisper", "mlx-lm",
             "mlx-qwen3-asr", "onnx-asr", "onnxruntime", "av", "transformers",
             "huggingface_hub", "numpy", "PySide6")
    found = []
    for name in names:
        try:
            found.append(f"{name}={version(name)}")
        except Exception:  # not installed, or a broken dist-info: never break logging
            continue
    return " ".join(found)


def get_logger() -> logging.Logger:
    """The app logger, created on first use.

    Never raises: a log that cannot be written must not take the app down, so
    a failing file handler degrades to a logger with no handlers at all.
    """
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("transviber")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        log_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.info("--- session start · %s · python %s · frozen=%s",
                    platform.platform(), platform.python_version(),
                    getattr(sys, "frozen", False))
        logger.info("packages: %s", _package_versions())
    except Exception:
        pass
    _logger = logger
    return _logger


def _context(fields: dict) -> str:
    return " ".join(f"{k}={v!r}" for k, v in fields.items() if v not in (None, ""))


def log_exception(what: str, exc: BaseException, **fields) -> None:
    """Record a failure with its traceback and reproduction context."""
    try:
        get_logger().error("%s failed: %s: %s | %s", what, type(exc).__name__, exc,
                           _context(fields), exc_info=exc)
    except Exception:
        pass


def log_event(message: str, **fields) -> None:
    """Record a notable non-failure (run started, backend chosen, export written)."""
    try:
        get_logger().info("%s | %s", message, _context(fields))
    except Exception:
        pass
