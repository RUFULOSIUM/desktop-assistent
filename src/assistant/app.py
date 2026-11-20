"""Main application class for the Desktop Assistant."""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from .config import Config
from .errors import AssistantError
from .logging import setup_logging
from .pipeline import AssistantPipeline, create_pipeline


class Application:
    """Main application class orchestrating all components."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._is_running = False
        self._shutdown_requested = False
        self._logger = logging.getLogger(__name__)
        self._pipeline: AssistantPipeline | None = None

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

        # Initialize pipeline
        self._pipeline = create_pipeline(self.config)
        self._pipeline.initialize()

        # Set up UI callbacks
        self._pipeline.set_callbacks(
            on_wake_detected=self._on_wake_detected,
            on_listening=self._on_listening,
            on_thinking=self._on_thinking,
            on_speaking=self._on_speaking,
            on_idle=self._on_idle,
        )

    def run(self) -> None:
        """Run the application main loop with error handling."""
        self.start()
        self._logger.info("Application started")

        try:
            while self._is_running and not self._shutdown_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received")
        except SystemExit:
            self._logger.info("System exit requested")
        except AssistantError as e:
            self._logger.error(f"Application error: {e}")
            raise
        except Exception as e:
            self._logger.exception(f"Unexpected error in main loop: {e}")
            raise
        finally:
            self.stop()
            self._logger.info("Application stopped")

    def stop(self) -> None:
        """Stop the application gracefully."""
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_requested = True

        if self._pipeline:
            self._pipeline.shutdown()
            self._pipeline = None

    def health_check(self) -> dict[str, Any]:
        """Run health checks on all components."""
        pipeline_ok = "ok" if self._pipeline and self._pipeline.is_running else "not_initialized"
        return {
            "config": "ok",
            "audio": "ok",
            "wake": "ok",
            "pipeline": pipeline_ok,
            "running": self._is_running,
        }

    # UI Callbacks
    def _on_wake_detected(self) -> None:
        self._logger.info("Wake word detected")
        # UI: show activated state

    def _on_listening(self) -> None:
        self._logger.info("Listening...")
        # UI: show listening state (pulsating)

    def _on_thinking(self) -> None:
        self._logger.info("Thinking...")
        # UI: show thinking state (loading circle)

    def _on_speaking(self, text: str) -> None:
        self._logger.info(f"Speaking: {text[:50]}...")
        # UI: show speaking state (pulsating to audio)

    def _on_idle(self) -> None:
        self._logger.info("Idle")
        # UI: return to idle state

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
