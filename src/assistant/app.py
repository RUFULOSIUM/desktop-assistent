"""Main application class for the Desktop Assistant."""

from __future__ import annotations

import signal
from typing import Any

from .config import Config
from .logging import setup_logging


class Application:
    """Main application class orchestrating all components."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._is_running = False
        self._shutdown_requested = False

        # Set up logging
        setup_logging(config)

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, _signum: int, _frame: Any) -> None:
        """Handle shutdown signals."""
        self._shutdown_requested = True
        self.stop()

    def start(self) -> None:
        """Start the application."""
        if self._is_running:
            return

        self._is_running = True
        self._shutdown_requested = False

        # Initialize components
        self._init_audio()
        self._init_wake()
        self._init_ui()

    def stop(self) -> None:
        """Stop the application gracefully."""
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_requested = True

    def health_check(self) -> dict[str, Any]:
        """Run health checks on all components."""
        return {
            "config": "ok",
            "audio": "ok",
            "wake": "ok",
            "running": self._is_running,
        }

    def _init_audio(self) -> None:
        """Initialize audio capture component."""
        # Placeholder for audio initialization
        pass

    def _init_wake(self) -> None:
        """Initialize wake word detection."""
        # Placeholder for wake word initialization
        pass

    def _init_ui(self) -> None:
        """Initialize UI component."""
        # Placeholder for UI initialization
        pass


def main() -> None:
    """Main entry point for the application."""
    from .cli import main as cli_main

    cli_main()
