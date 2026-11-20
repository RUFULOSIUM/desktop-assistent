"""Tests for sherpa-onnx wake word engine."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


def create_fake_model_dir() -> Path:
    """Create a temporary directory with fake sherpa-onnx model files."""
    tmpdir = tempfile.mkdtemp()
    model_dir = Path(tmpdir)

    # Create required model files
    (model_dir / "tokens.txt").write_text("0\n1\n2\n")
    (model_dir / "encoder.onnx").write_bytes(b"fake encoder")
    (model_dir / "decoder.onnx").write_bytes(b"fake decoder")
    (model_dir / "joiner.onnx").write_bytes(b"fake joiner")
    (model_dir / "keywords.txt").write_text("hey assistant\nhello world\n")

    return model_dir


def make_mock_sherpa_onnx(
    mock_get_result_return: str = "",
    mock_side_effect: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock sherpa_onnx module with configurable behavior."""
    mock_sherpa = MagicMock()
    mock_kws = MagicMock()
    mock_stream = MagicMock()

    mock_kws.create_stream.return_value = mock_stream
    mock_kws.is_ready.return_value = False
    mock_kws.get_result.return_value = mock_get_result_return

    if mock_side_effect:
        mock_sherpa.KeywordSpotter.side_effect = mock_side_effect
    else:
        mock_sherpa.KeywordSpotter.return_value = mock_kws

    return mock_sherpa, mock_kws


def patch_sys_modules(mock_sherpa: MagicMock) -> Any:
    """Patch sys.modules with mock sherpa_onnx."""
    return patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa})


def test_wake_engine_init() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)
        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            assert engine.config == config
            assert engine.is_running() is False
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_process_frame() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, mock_kws = make_mock_sherpa_onnx()
    # get_result is only called when is_ready=True
    # is_ready sequence: 3x False (no detection), then True, False, True (detection)
    # get_result sequence: "" (for first True), "hey assistant" (for second True)
    mock_kws.is_ready.side_effect = [False, False, False, True, False, True]
    mock_kws.get_result.side_effect = ["", "hey assistant"]

    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)
        callback = MagicMock()

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.set_callback(callback)
            engine.start()

            # sherpa-onnx expects 960 bytes (30ms at 16kHz)
            frame = bytes(1920)  # 960 samples * 2 bytes = 1920 bytes

            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is False
            assert callback.call_count == 0

            # Next call: is_ready=True, get_result="", is_ready=False -> no detection
            result = engine.process_frame(frame)
            assert result is False

            # Next call: is_ready=True, get_result="hey assistant" -> detected!
            result = engine.process_frame(frame)
            assert result is True
            callback.assert_called_once_with("hey assistant")

            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_model_loading_error() -> None:
    from assistant.config import WakeConfig
    from assistant.errors import WakeError
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        try:
            engine = SherpaWakeEngine(config, model_dir=Path("/nonexistent"))
            engine.start()
            raise AssertionError("Should have raised WakeError")
        except WakeError as e:
            assert "model" in str(e).lower()


def test_wake_engine_config_integration() -> None:
    from assistant.config import Config
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        cfg = Config.get_default()
        cfg.wake.word = "computer"
        cfg.wake.enabled = True

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(cfg.wake, model_dir=model_dir)
            assert engine.config.word == "computer"
            assert engine.config.enabled is True
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_start_stop() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
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
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
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
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, mock_kws = make_mock_sherpa_onnx()
    # 2x False, then True (get_result="hey assistant"), then False
    mock_kws.is_ready.side_effect = [False, False, True, False]
    mock_kws.get_result.side_effect = ["hey assistant"]

    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(1920)  # 960 samples * 2 bytes = 1920 bytes
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
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)

            with patch.dict("sys.modules", {"sherpa_onnx": None}):
                try:
                    engine.start()
                    raise AssertionError("Should have raised WakeError")
                except WakeError as e:
                    assert "sherpa-onnx not installed" in str(e)
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_model_load_error() -> None:
    from assistant.config import WakeConfig
    from assistant.errors import WakeError
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx(mock_side_effect=RuntimeError("Model load failed"))
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            try:
                engine.start()
                raise AssertionError("Should have raised WakeError")
            except WakeError as e:
                assert "Failed to load keyword spotter model" in str(e)
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_callback_error() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, mock_kws = make_mock_sherpa_onnx()
    # 1x False, then True (get_result="hey assistant"), then False
    mock_kws.is_ready.side_effect = [False, True, False]
    mock_kws.get_result.side_effect = ["hey assistant"]

    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.set_callback(callback)
            engine.start()

            frame = bytes(1920)  # 960 samples * 2 bytes = 1920 bytes
            result = engine.process_frame(frame)
            assert result is False
            result = engine.process_frame(frame)
            assert result is True
            callback.assert_called_once_with("hey assistant")
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_process_error() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, mock_kws = make_mock_sherpa_onnx(mock_get_result_return="")
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        mock_kws.is_ready.side_effect = RuntimeError("Prediction failed")

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(1920)  # 960 samples * 2 bytes = 1920 bytes
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
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.stop()
            assert engine.is_running() is False
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_disabled() -> None:
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=False)

        model_dir = create_fake_model_dir()
        try:
            engine = SherpaWakeEngine(config, model_dir=model_dir)
            engine.start()

            frame = bytes(1920)  # 960 samples * 2 bytes = 1920 bytes
            result = engine.process_frame(frame)

            assert result is False
            engine.stop()
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()


def test_wake_engine_alias() -> None:
    """Test that WakeWordEngine alias works."""
    from assistant.config import WakeConfig
    from assistant.wake import WakeWordEngine

    mock_sherpa, _ = make_mock_sherpa_onnx()
    with patch_sys_modules(mock_sherpa):
        config = WakeConfig(word="hey assistant", enabled=True)
        model_dir = create_fake_model_dir()
        try:
            engine = WakeWordEngine(config, model_dir=model_dir)
            from assistant.wake.sherpa_engine import SherpaWakeEngine
            assert isinstance(engine, SherpaWakeEngine)
        finally:
            for f in model_dir.glob("*"):
                f.unlink()
            model_dir.rmdir()