"""Tests for openWakeWord engine wrapper."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Create a mock openwakeword module before any imports
mock_openwakeword = MagicMock()
mock_model_class = MagicMock()
mock_openwakeword.model.Model = mock_model_class
sys.modules["openwakeword"] = mock_openwakeword
sys.modules["openwakeword.model"] = mock_openwakeword.model


def create_fake_model_dir() -> Path:
    """Create a temporary directory with a fake .onnx model file."""
    tmpdir = tempfile.mkdtemp()
    model_file = Path(tmpdir) / "hey_assistant.onnx"
    model_file.write_bytes(b"fake onnx model")
    return Path(tmpdir)


def reset_mocks() -> None:
    """Reset all mocks to default state."""
    mock_model_class.reset_mock()
    mock_model_class.return_value = None
    mock_model_class.side_effect = None


def test_wake_engine_init() -> None:

    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    reset_mocks()
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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)
    callback = Mock()

    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_assistant": 0.0}
    mock_model_class.return_value = mock_model

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

    reset_mocks()
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

    reset_mocks()
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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model

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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)

    mock_model = MagicMock()
    mock_model_class.return_value = mock_model

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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)

    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_assistant": 0.8}
    mock_model_class.return_value = mock_model

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

    reset_mocks()
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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)

    mock_model_class.side_effect = RuntimeError("Model load failed")

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
        mock_model_class.side_effect = None


def test_wake_engine_callback_error() -> None:

    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)
    callback = Mock(side_effect=RuntimeError("Callback failed"))

    mock_model = MagicMock()
    mock_model.predict.return_value = {"hey_assistant": 0.8}
    mock_model_class.return_value = mock_model

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

    reset_mocks()
    config = WakeConfig(word="hey assistant", enabled=True)

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Prediction failed")
    mock_model_class.return_value = mock_model

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

    reset_mocks()
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

    reset_mocks()
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
