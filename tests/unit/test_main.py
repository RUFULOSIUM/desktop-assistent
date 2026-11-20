"""Tests for __main__ module."""

from unittest.mock import patch


def test_main_module() -> None:
    """Test that __main__ module can be executed."""
    from assistant import __main__

    # Test that the module has the main function
    assert hasattr(__main__, "main")


def test_main_function() -> None:
    """Test the main function in __main__."""
    import runpy
    from unittest.mock import patch

    with patch("assistant.app.main") as mock_app_main:
        # Run the module as __main__
        runpy.run_module("assistant.__main__", run_name="__main__")
        mock_app_main.assert_called_once()


def test_main_module_execution() -> None:
    """Test executing the module as __main__."""
    with patch("assistant.app.main") as mock_app_main:
        # Simulate running as __main__
        import runpy

        runpy.run_module("assistant.__main__", run_name="__main__")
        mock_app_main.assert_called_once()