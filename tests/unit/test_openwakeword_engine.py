"""Tests for openWakeWord engine wrapper."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch


def create_fake_model_dir() -> Path:
    """Create a temporary directory with a fake .onnx model file."""
    tmpdir = tempfile.mkdtemp()
    model_file = Path(tmpdir) / "hey_assistant.onnx"
    model_file.write_bytes(b"fake onnx model")
    return Path(tmpdir)


def make_mock_openwakeword(
    mock_predict_return: dict[str, float] | None = None,
    mock_side_effect: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock openwakeword module with configurable behavior."""
    mock_openwakeword = MagicMock()
    mock_model_class = MagicMock()

    if mock_side_effect:
        mock_model_class.side_effect = mock_side_effect
    else:
        mock_model = MagicMock()
        if mock_predict_return is not None:
            mock_model.predict.return_value = mock_predict_return
        mock_model_class.return_value = mock_model

    mock_openwakeword.model.Model = mock_model_class
    return mock_openwakeword, mock_model_class


def patch_sys_modules(mock_openwakeword: MagicMock) -> Any:
    """Patch sys.modules with mock openwakeword."""
    return patch.dict(
        "sys.modules",
        {"openwakeword": mock_openwakeword, "openwakeword.model": mock_openwakeword.model},
    )


def test_wake_engine_init() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)
        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            assert engine.config == config
            assert engine.is_running() is False
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_process_frame() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, mock_model_class = make_mock_openwakeword(
        mock_predict_return={"hey_assistant": 0.0}
    )
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)
        callback = Mock()

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.set_callback(callback)
            engine.start()

            frame = bytes(960)

            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            assert callback.call_count == 0

            # Update the mock to return detection
            mock_model = mock_model_class.return_value
            mock_model.predict.return_value = {"hey_assistant": 0.8}
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is True
            callback.assert_called_once_with("hey_assistant")

            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_model_loading_error() -> None:
    from assistant.config import WakeConfig
    from assistant.errors import WakeError
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        try:
            engine = WakeWordEngine(config, model_dir=Path("/nonexistent"))
            engine.start()
            raise AssertionError("Should have raised WakeError")
        except WakeError as e:
            assert "model" in str(e).lower()


def test_wake_engine_config_integration() -> None:
    from assistant.config import Config
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        cfg = Config.get_default()
        cfg.wake.word = "computer"
        cfg.wake.enabled = True

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(cfg.wake, model_dir=model_dir)
            assert engine.config.word == "computer"
            assert engine.config.enabled is True
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_start_stop() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            assert engine.is_running() is False

            engine.start()
            assert engine.is_running() is True

            engine.stop()
            assert engine.is_running() is False
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_double_start() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.start()
            engine.start()
            assert engine.is_running() is True
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_callback_none() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, mock_model_class = make_mock_openwakeword(
        mock_predict_return={"hey_assistant": 0.8}
    )
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(960)
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is True
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_import_error() -> None:
    from assistant.config import WakeConfig
    from assistant.errors import WakeError
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)

            with patch.dict("sys.modules", {"openwakeword": None, "openwakeword.model": None}):
                try:
                    engine.start()
                    raise AssertionError("Should have raised WakeError")
                except WakeError as e:
                    assert "openWakeWord not installed" in str(e)
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_model_load_error() -> None:
    from assistant.config import WakeConfig
    from assistant.errors import WakeError
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, mock_model_class = make_mock_openwakeword(
        mock_side_effect=RuntimeError("Model load failed")
    )
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            try:
                engine.start()
                raise AssertionError("Should have raised WakeError")
            except WakeError as e:
                assert "Failed to load wake word model" in str(e)
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_callback_error() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, mock_model_class = make_mock_openwakeword(
        mock_predict_return={"hey_assistant": 0.8}
    )
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)
        callback = Mock(side_effect=RuntimeError("Callback failed"))

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.set_callback(callback)
            engine.start()

            frame = bytes(960)
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is True
            callback.assert_called_once_with("hey_assistant")
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_prediction_error() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, mock_model_class = make_mock_openwakeword(
        mock_predict_return={"hey_assistant": 0.0}
    )
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        mock_model = mock_model_class.return_value
        mock_model.predict.side_effect = RuntimeError("Prediction failed")

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(960)
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_stop_not_running() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.stop()
            assert engine.is_running() is False
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_disabled() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    mock_openwakeword, _ = make_mock_openwakeword()
    with patch_sys_modules(mock_openwakeword):
        config = WakeConfig(word="hey assistant", enabled=False)

        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(960)
            result = engine.process_frame(frame)

            assert result is False
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()
