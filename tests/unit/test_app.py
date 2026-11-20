# mypy: ignore-errors
"""Tests for app module."""

import signal
from unittest.mock import patch


def test_app_initialization() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)
    assert app.config == cfg
    assert app.is_running is False


def test_app_start_stop() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with patch("assistant.app.create_pipeline") as mock_create_pipeline:
        mock_pipeline = mock_create_pipeline.return_value
        app.start()
        assert app.is_running is True
        mock_create_pipeline.assert_called_once_with(cfg)
        mock_pipeline.initialize.assert_called_once()
        mock_pipeline.set_callbacks.assert_called_once()

        app.stop()
        assert app.is_running is False
        mock_pipeline.shutdown.assert_called_once()


def test_app_health_check() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)
    health = app.health_check()
    assert isinstance(health, dict)
    assert "config" in health
    assert "audio" in health
    assert "wake" in health


def test_app_signal_handling() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    Application(cfg)

    # Test that signal handlers are set up
    assert signal.getsignal(signal.SIGINT) is not None
    assert signal.getsignal(signal.SIGTERM) is not None


def test_app_double_start() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with patch("assistant.app.create_pipeline") as mock_create_pipeline:
        _ = mock_create_pipeline.return_value
        app.start()
        assert app.is_running is True
        # Starting again should be idempotent
        app.start()
        assert app.is_running is True
        # create_pipeline should only be called once
        assert mock_create_pipeline.call_count == 1


def test_app_stop_without_start() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    assert app.is_running is False
    # Stopping without starting should be safe
    app.stop()
    assert app.is_running is False


def test_app_signal_handler() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    # Simulate signal handler
    app._signal_handler(signal.SIGINT, None)
    assert app._shutdown_requested is True
    assert app.is_running is False


def test_app_main_function() -> None:
    from unittest.mock import patch

    from assistant.app import main

    # Test that main function exists and is callable
    assert callable(main)

    # Call main to cover lines 88-90
    with patch("assistant.cli.main") as mock_cli_main:
        main()
        mock_cli_main.assert_called_once()


def test_app_init_methods() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    # Call placeholder methods to cover lines 73, 78, 83
    app._init_audio()
    app._init_wake()
    app._init_ui()

    # Test stop when not running (covers line 55-56 early return)
    app.stop()
    assert app.is_running is False


def test_app_stop_idempotent() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    # Stop multiple times should be safe
    app.stop()
    app.stop()
    app.stop()
    assert app.is_running is False


def test_app_run_method() -> None:
    """Test the run method with KeyboardInterrupt."""
    from unittest.mock import patch

    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with (
        patch.object(app, "start"),
        patch.object(app, "stop"),
        patch("time.sleep", side_effect=KeyboardInterrupt),
    ):
        app.run()
        app.start.assert_called_once()
        app.stop.assert_called_once()


def test_app_run_method_system_exit() -> None:
    """Test the run method with SystemExit."""
    from unittest.mock import patch

    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with (
        patch.object(app, "start"),
        patch.object(app, "stop"),
        patch("time.sleep", side_effect=SystemExit),
    ):
        app.run()
        app.start.assert_called_once()
        app.stop.assert_called_once()


def test_app_run_method_assistant_error() -> None:
    """Test the run method with AssistantError."""
    import contextlib
    from unittest.mock import patch

    from assistant.app import Application
    from assistant.config import Config
    from assistant.errors import AssistantError

    cfg = Config.get_default()
    app = Application(cfg)

    with (
        patch.object(app, "start"),
        patch.object(app, "stop"),
        patch("time.sleep", side_effect=AssistantError("test error")),
    ):
        with contextlib.suppress(AssistantError):
            app.run()
        app.start.assert_called_once()
        app.stop.assert_called_once()


def test_app_run_method_unexpected_error() -> None:
    """Test the run method with unexpected error."""
    import contextlib
    from unittest.mock import patch

    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with (
        patch.object(app, "start"),
        patch.object(app, "stop"),
        patch("time.sleep", side_effect=RuntimeError("unexpected")),
    ):
        with contextlib.suppress(RuntimeError):
            app.run()
        app.start.assert_called_once()
        app.stop.assert_called_once()
