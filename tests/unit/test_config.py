# mypy: ignore-errors
"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest


def test_config_defaults() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    assert cfg.security.level == "standard"
    assert cfg.wake.word == "HEY_ASSISTANT"
    assert cfg.wake.enabled is True
    assert cfg.stt.provider == "cloud"
    assert cfg.llm.provider == "local"
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.channels == 1


def test_config_load_save() -> None:
    from assistant.config import Config

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        path = Path(f.name)

    try:
        cfg = Config.get_default()
        cfg.security.level = "strict"
        cfg.wake.word = "computer"
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.security.level == "strict"
        assert loaded.wake.word == "computer"
    finally:
        path.unlink(missing_ok=True)


def test_config_validation() -> None:
    from assistant.config import Config

    # Invalid security level
    cfg = Config.get_default()
    cfg.security.level = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid sample rate
    cfg = Config.get_default()
    cfg.audio.sample_rate = 100
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid channels
    cfg = Config.get_default()
    cfg.audio.channels = 3
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid frame duration
    cfg = Config.get_default()
    cfg.audio.frame_duration_ms = 40
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid VAD aggressiveness
    cfg = Config.get_default()
    cfg.audio.vad_aggressiveness = 5
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid UI position
    cfg = Config.get_default()
    cfg.ui.position = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid verbosity
    cfg = Config.get_default()
    cfg.ui.verbosity = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid log level
    cfg = Config.get_default()
    cfg.logging.level = "INVALID"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid STT provider
    cfg = Config.get_default()
    cfg.stt.provider = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid LLM provider
    cfg = Config.get_default()
    cfg.llm.provider = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid wake word (empty)
    cfg = Config.get_default()
    cfg.wake.word = ""
    with pytest.raises(ValueError):
        cfg.validate()

    # Private path validation
    cfg = Config.get_default()
    cfg.permissions.private_paths = ["/nonexistent/path"]
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_first_run_api_scan() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    # Should not crash
    detected = cfg.scan_installed_apis()
    assert isinstance(detected, dict)
    assert "llm" in detected
    assert "stt" in detected


def test_config_to_dict_excludes_none() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    # input_device and file are None by default
    d = cfg.to_dict()
    assert "input_device" not in d["audio"]
    assert "file" not in d["logging"]
    # api_key should be excluded when empty
    assert "api_key" not in d["llm"]


def test_config_from_dict_partial() -> None:
    from assistant.config import Config

    data = {
        "security": {"level": "paranoid"},
        "wake": {"word": "jarvis"},
    }
    cfg = Config.from_dict(data)
    assert cfg.security.level == "paranoid"
    assert cfg.wake.word == "jarvis"
    # Other values should be defaults
    assert cfg.stt.provider == "cloud"
    assert cfg.audio.sample_rate == 16000


def test_config_save_load_roundtrip() -> None:
    from assistant.config import Config

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        path = Path(f.name)

    try:
        cfg = Config.get_default()
        cfg.security.level = "strict"
        cfg.wake.word = "computer"
        cfg.audio.sample_rate = 48000
        cfg.permissions.private_paths = ["/private/data"]
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.security.level == "strict"
        assert loaded.wake.word == "computer"
        assert loaded.audio.sample_rate == 48000
        assert loaded.permissions.private_paths == ["/private/data"]
    finally:
        path.unlink(missing_ok=True)


def test_config_default_config_path() -> None:
    from assistant.config import Config

    path = Config.get_default_config_path()
    assert path.name == "config.toml"
    assert "Desktop Assistant" in str(path) or "desktop-assistant" in str(path)


def test_config_missing_file_returns_default() -> None:
    from assistant.config import Config

    cfg = Config.load(Path("/nonexistent/path/config.toml"))
    assert cfg.security.level == "standard"
    assert cfg.wake.word == "HEY_ASSISTANT"


def test_config_api_key_excluded_when_empty() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    d = cfg.to_dict()
    assert "api_key" not in d["llm"]

    cfg.llm.api_key = "test-key"
    d = cfg.to_dict()
    assert d["llm"]["api_key"] == "test-key"


def test_security_config_empty_level() -> None:
    from assistant.config import SecurityConfig

    cfg = SecurityConfig(level="")
    with pytest.raises(ValueError):
        cfg.validate()


def test_wake_config_long_word() -> None:
    from assistant.config import WakeConfig

    cfg = WakeConfig(word="x" * 51)
    with pytest.raises(ValueError, match="too long"):
        cfg.validate()


def test_stt_config_validate() -> None:
    from assistant.config import STTConfig

    cfg = STTConfig(provider="invalid")
    with pytest.raises(ValueError):
        cfg.validate()


def test_llm_config_validate() -> None:
    from assistant.config import LLMConfig

    cfg = LLMConfig(provider="invalid")
    with pytest.raises(ValueError):
        cfg.validate()


def test_permissions_config_validate() -> None:
    from assistant.config import PermissionsConfig

    cfg = PermissionsConfig(private_paths=["/nonexistent/path"])
    with pytest.raises(ValueError):
        cfg.validate()
