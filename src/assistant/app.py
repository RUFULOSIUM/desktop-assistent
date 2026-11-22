"""Main application class for the Desktop Assistant."""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from .config import Config
from .doctor import run_doctor
from .errors import AssistantError
from .hotkeys import HotkeyManager, create_hotkey_manager, HotkeyConfig
from .logging import setup_logging
from .pipeline import AssistantPipeline, create_pipeline
from .ui import IslandWindow, create_island_window, MenuBarController, create_menu_bar


class Application:
    """Main application class orchestrating all components."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._is_running = False
        self._shutdown_requested = False
        self._logger = logging.getLogger(__name__)
        self._pipeline: AssistantPipeline | None = None
        self._island_window: IslandWindow | None = None
        self._menu_bar: MenuBarController | None = None
        self._hotkey_manager: HotkeyManager | None = None

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

        # Initialize UI first (needs to be ready for pipeline callbacks)
        self._init_ui()

        # Initialize pipeline
        self._pipeline = create_pipeline(self.config)
        self._pipeline.initialize()

        # Wire UI to pipeline callbacks
        self._wire_ui_callbacks()

        # Initialize hotkeys (needs pipeline and island view)
        self._init_hotkeys()

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

        # Stop hotkeys
        if self._hotkey_manager:
            self._hotkey_manager.stop()
            self._hotkey_manager = None

        # Shutdown UI
        if self._island_window:
            self._island_window.shutdown()
            self._island_window = None
        if self._menu_bar:
            self._menu_bar.shutdown()
            self._menu_bar = None

        if self._pipeline:
            self._pipeline.shutdown()
            self._pipeline = None

    def health_check(self) -> dict[str, Any]:
        """Run health checks on all components."""
        from .doctor import Doctor

        doctor = Doctor(self.config)
        results = doctor.run_all_checks()

        # Convert to simple dict for backward compatibility
        health_dict = {}
        for result in results:
            key = result.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            health_dict[key] = result.status

        # Add legacy keys for backward compatibility
        health_dict["config"] = health_dict.get("configuration", "ok")
        health_dict["audio"] = health_dict.get("microphone_hardware", "ok")
        health_dict["wake"] = health_dict.get("wake_word_models", "ok")
        health_dict["running"] = self._is_running
        health_dict["pipeline"] = "ok" if self._pipeline and self._pipeline.is_running else "not_initialized"
        health_dict["ui"] = "ok" if self._island_window else "not_initialized"

        return health_dict

    def run_doctor(self) -> int:
        """Run full health check and print report."""
        return run_doctor(self.config)

    def _init_ui(self) -> None:
        """Initialize UI components."""
        if not self.config.ui.enabled:
            self._logger.info("UI disabled in config")
            return

        # Create island window
        self._island_window = create_island_window(self.config.ui.position)
        self._island_window.initialize()

        # Create menu bar (optional)
        self._menu_bar = create_menu_bar()
        self._menu_bar.initialize()

        self._logger.info("UI initialized")

    def _wire_ui_callbacks(self) -> None:
        """Wire UI state changes to pipeline callbacks."""
        if not self._pipeline or not self._island_window:
            return

        def on_wake_detected() -> None:
            self._logger.info("Wake word detected")
            self._island_window.set_state("activated")
            if self._menu_bar:
                self._menu_bar.set_running(True, "listening")

        def on_listening() -> None:
            self._logger.info("Listening...")
            self._island_window.set_state("listening")
            if self._menu_bar:
                self._menu_bar.set_running(True, "listening")

        def on_thinking() -> None:
            self._logger.info("Thinking...")
            self._island_window.set_state("thinking")
            if self._menu_bar:
                self._menu_bar.set_running(True, "thinking")

        def on_speaking(text: str) -> None:
            self._logger.info(f"Speaking: {text[:50]}...")
            self._island_window.set_state("speaking")
            if self._menu_bar:
                self._menu_bar.set_running(True, "speaking")

        def on_idle() -> None:
            self._logger.info("Idle")
            # On notch Macs, hide completely when idle
            if self.config.ui.position == "notch":
                self._island_window.set_state("hidden")
            else:
                self._island_window.set_state("idle")
            if self._menu_bar:
                self._menu_bar.set_running(False, "idle")

        def on_files_dropped(file_paths: list[str]) -> None:
            self._logger.info(f"Files dropped on island: {file_paths}")
            self._pipeline.set_dropped_files(file_paths)

        self._pipeline.set_callbacks(
            on_wake_detected=on_wake_detected,
            on_listening=on_listening,
            on_thinking=on_thinking,
            on_speaking=on_speaking,
            on_idle=on_idle,
        )

        # Wire island window file drop callback
        self._island_window.set_callbacks(
            on_wake_detected=on_wake_detected,
            on_listening=on_listening,
            on_thinking=on_thinking,
            on_speaking=on_speaking,
            on_idle=on_idle,
            on_files_dropped=on_files_dropped,
        )

        # Also wire menu bar toggle
        if self._menu_bar:
            self._menu_bar.set_callbacks(on_toggle=self._toggle_listening)

    def _toggle_listening(self) -> None:
        """Toggle listening state (for menu bar)."""
        if not self._pipeline:
            return
        # This would trigger wake word detection manually
        # For now, just log
        self._logger.info("Manual toggle requested")

    def _init_hotkeys(self) -> None:
        """Initialize global hotkey manager."""
        if not self._pipeline or not self._island_window:
            return

        # Build pipeline callbacks for hotkeys
        pipeline_callbacks = {
            'on_wake_detected': self._on_hotkey_wake,
            'on_toggle_listening': self._on_toggle_listening,
            'on_show_hide_window': self._on_show_hide_window,
        }

        # Create hotkey config from app config
        hotkey_config = HotkeyConfig(
            triple_shift=self.config.hotkeys.triple_shift,
            double_click=self.config.hotkeys.double_click,
            push_to_talk=self.config.hotkeys.push_to_talk,
            toggle_listening=self.config.hotkeys.toggle_listening,
            show_hide_window=self.config.hotkeys.show_hide_window,
        )

        # Get island view for double-click monitoring
        island_view = self._island_window.get_view() if self._island_window else None

        self._hotkey_manager = create_hotkey_manager(
            hotkey_config,
            pipeline_callbacks,
            island_view
        )
        self._hotkey_manager.start()
        self._logger.info("Hotkeys initialized")

    def _on_hotkey_wake(self) -> None:
        """Handle wake activation from hotkeys (triple-shift, double-click, push-to-talk)."""
        self._logger.info("Hotkey wake activated")
        if self._island_window:
            self._island_window.set_state("activated")
        if self._menu_bar:
            self._menu_bar.set_running(True, "listening")
        # Trigger pipeline to start listening
        if self._pipeline:
            # Call the internal wake word handler which starts listening
            self._pipeline._on_wake_word("hotkey")

    def _on_toggle_listening(self) -> None:
        """Handle toggle listening hotkey."""
        self._logger.info("Toggle listening hotkey")
        self._toggle_listening()

    def _on_show_hide_window(self) -> None:
        """Handle show/hide window hotkey."""
        self._logger.info("Show/hide window hotkey")
        if self._island_window:
            if self._island_window.is_visible():
                self._island_window.hide()
            else:
                self._island_window.show()

    def _init_audio(self) -> None:
        """Initialize audio capture component."""
        # Placeholder for audio initialization
        pass

    def _init_wake(self) -> None:
        """Initialize wake word detection."""
        # Placeholder for wake word initialization
        pass


def main() -> None:
    """Main entry point for the application."""
    from .cli import main as cli_main

    cli_main()
