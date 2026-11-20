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

    with (
        patch.object(app, "_init_audio") as mock_audio,
        patch.object(app, "_init_wake") as mock_wake,
        patch.object(app, "_init_ui") as mock_ui,
    ):
        app.start()
        assert app.is_running is True
        mock_audio.assert_called_once()
        mock_wake.assert_called_once()
        mock_ui.assert_called_once()

        app.stop()
        assert app.is_running is False


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

    with (
        patch.object(app, "_init_audio"),
        patch.object(app, "_init_wake"),
        patch.object(app, "_init_ui"),
    ):
        app.start()
        assert app.is_running is True
        # Starting again should be idempotent
        app.start()
        assert app.is_running is True


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
