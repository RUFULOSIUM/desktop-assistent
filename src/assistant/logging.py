"""Logging configuration for the Desktop Assistant."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from io import StringIO
from pathlib import Path

from .config import Config


def setup_logging(
    config: Config,
    stream: StringIO | None = None,
    file_path: str | Path | None = None,
) -> None:
    """Configure logging based on configuration.

    Args:
        config: Configuration object with logging settings.
        stream: Optional stream for console output (for testing).
        file_path: Optional file path for file logging.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.logging.format)

    # Console handler
    if stream is not None:
        console_handler = logging.StreamHandler(stream)
    else:
        console_handler = logging.StreamHandler(sys.stdout)  # type: ignore[arg-type,assignment]

    console_handler.setLevel(config.logging.level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    log_file = file_path or config.logging.file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.logging.max_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(config.logging.level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)
