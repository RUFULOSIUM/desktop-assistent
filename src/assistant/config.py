"""Configuration system for the Desktop Assistant."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli
import tomli_w


@dataclass
class SecurityConfig:
    """Security level configuration."""

    level: str = "standard"  # standard, strict, paranoid

    def validate(self) -> None:
        valid_levels = ("standard", "strict", "paranoid")
        if self.level not in valid_levels:
            raise ValueError(f"Invalid security level: {self.level}. Must be one of {valid_levels}")


@dataclass
class WakeConfig:
    """Wake word configuration."""

    enabled: bool = True
    word: str = "HEY_ASSISTANT"
    fallback_double_click: bool = True
    fallback_triple_shift: bool = True

    def validate(self) -> None:
        if not self.word:
            raise ValueError("Wake word cannot be empty")
        if len(self.word) > 50:
            raise ValueError("Wake word too long (max 50 characters)")


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    provider: str = "cloud"  # cloud, local, auto
    cloud_service: str = "whisper-api"  # whisper-api, google, azure
    local_model: str = "base"  # tiny, base, small, medium, large
    language: str = "en"
    fallback_chain: list[str] = field(default_factory=lambda: ["cloud", "local"])

    def validate(self) -> None:
        valid_providers = ("cloud", "local", "auto")
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid STT provider: {self.provider}. Must be one of {valid_providers}"
            )


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "local"  # local, openai, anthropic, auto
    local_model: str = "llama3"
    remote_model: str = "gpt-4"
    api_key: str = ""
    fallback_chain: list[str] = field(default_factory=lambda: ["local", "remote"])

    def validate(self) -> None:
        valid_providers = ("local", "openai", "anthropic", "auto")
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be one of {valid_providers}"
            )


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 3
    input_device: str | None = None

    def validate(self) -> None:
        valid_rates = (8000, 16000, 32000, 48000)
        if self.sample_rate not in valid_rates:
            raise ValueError(
                f"Invalid sample rate: {self.sample_rate}. Must be one of {valid_rates}"
            )
        if self.channels not in (1, 2):
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
        if self.frame_duration_ms not in (10, 20, 30):
            raise ValueError("Frame duration must be 10, 20, or 30 ms")
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError("VAD aggressiveness must be between 0 and 3")


@dataclass
class UIConfig:
    """UI configuration."""

    enabled: bool = True
    position: str = "notch"  # notch, menu_bar, center
    animation_enabled: bool = True
    sound_effects: bool = True
    notifications: bool = True
    verbosity: str = "normal"  # silent, minimal, normal, verbose

    def validate(self) -> None:
        valid_positions = ("notch", "menu_bar", "center")
        if self.position not in valid_positions:
            raise ValueError(
                f"Invalid UI position: {self.position}. Must be one of {valid_positions}"
            )
        valid_verbosity = ("silent", "minimal", "normal", "verbose")
        if self.verbosity not in valid_verbosity:
            raise ValueError(
                f"Invalid verbosity: {self.verbosity}. Must be one of {valid_verbosity}"
            )


@dataclass
class PermissionsConfig:
    """Permissions configuration."""

    allow_shell: bool = True
    allow_files: bool = True
    allow_apps: bool = True
    allow_web: bool = True
    allow_clipboard: bool = False
    private_paths: list[str] = field(default_factory=list)

    def validate(self) -> None:
        for path in self.private_paths:
            p = Path(path).expanduser()
            if not p.exists():
                raise ValueError(f"Private path does not exist: {path}")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: str | None = None
    max_size_mb: int = 10
    backup_count: int = 5

    def validate(self) -> None:
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")


@dataclass
class Config:
    """Main configuration class."""

    security: SecurityConfig = field(default_factory=SecurityConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def get_default(cls) -> Config:
        """Create a configuration with default values."""
        return cls()

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load configuration from TOML file."""
        if path is None:
            path = cls.get_default_config_path()

        if not path.exists():
            return cls.get_default()

        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls.from_dict(data)

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Desktop Assistant"
                / "config.toml"
            )
        return Path.home() / ".config" / "desktop-assistant" / "config.toml"

    def save(self, path: Path | None = None) -> None:
        """Save configuration to TOML file."""
        if path is None:
            path = self.get_default_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""

        def _filter_none(d: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in d.items() if v is not None}

        return {
            "security": {"level": self.security.level},
            "wake": _filter_none(
                {
                    "enabled": self.wake.enabled,
                    "word": self.wake.word,
                    "fallback_double_click": self.wake.fallback_double_click,
                    "fallback_triple_shift": self.wake.fallback_triple_shift,
                }
            ),
            "stt": _filter_none(
                {
                    "provider": self.stt.provider,
                    "cloud_service": self.stt.cloud_service,
                    "local_model": self.stt.local_model,
                    "language": self.stt.language,
                    "fallback_chain": self.stt.fallback_chain,
                }
            ),
            "llm": _filter_none(
                {
                    "provider": self.llm.provider,
                    "local_model": self.llm.local_model,
                    "remote_model": self.llm.remote_model,
                    "api_key": self.llm.api_key or None,
                    "fallback_chain": self.llm.fallback_chain,
                }
            ),
            "audio": _filter_none(
                {
                    "sample_rate": self.audio.sample_rate,
                    "channels": self.audio.channels,
                    "frame_duration_ms": self.audio.frame_duration_ms,
                    "vad_aggressiveness": self.audio.vad_aggressiveness,
                    "input_device": self.audio.input_device,
                }
            ),
            "ui": _filter_none(
                {
                    "enabled": self.ui.enabled,
                    "position": self.ui.position,
                    "animation_enabled": self.ui.animation_enabled,
                    "sound_effects": self.ui.sound_effects,
                    "notifications": self.ui.notifications,
                    "verbosity": self.ui.verbosity,
                }
            ),
            "permissions": _filter_none(
                {
                    "allow_shell": self.permissions.allow_shell,
                    "allow_files": self.permissions.allow_files,
                    "allow_apps": self.permissions.allow_apps,
                    "allow_web": self.permissions.allow_web,
                    "allow_clipboard": self.permissions.allow_clipboard,
                    "private_paths": self.permissions.private_paths,
                }
            ),
            "logging": _filter_none(
                {
                    "level": self.logging.level,
                    "format": self.logging.format,
                    "file": self.logging.file,
                    "max_size_mb": self.logging.max_size_mb,
                    "backup_count": self.logging.backup_count,
                }
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        config = cls()

        if "security" in data:
            config.security.level = data["security"].get("level", "standard")

        if "wake" in data:
            w = data["wake"]
            config.wake.enabled = w.get("enabled", True)
            config.wake.word = w.get("word", "hey assistant")
            config.wake.fallback_double_click = w.get("fallback_double_click", True)
            config.wake.fallback_triple_shift = w.get("fallback_triple_shift", True)

        if "stt" in data:
            s = data["stt"]
            config.stt.provider = s.get("provider", "cloud")
            config.stt.cloud_service = s.get("cloud_service", "whisper-api")
            config.stt.local_model = s.get("local_model", "base")
            config.stt.language = s.get("language", "en")
            config.stt.fallback_chain = s.get("fallback_chain", ["cloud", "local"])

        if "llm" in data:
            llm_data = data["llm"]
            config.llm.provider = llm_data.get("provider", "local")
            config.llm.local_model = llm_data.get("local_model", "llama3")
            config.llm.remote_model = llm_data.get("remote_model", "gpt-4")
            config.llm.api_key = llm_data.get("api_key", "")
            config.llm.fallback_chain = llm_data.get("fallback_chain", ["local", "remote"])

        if "audio" in data:
            a = data["audio"]
            config.audio.sample_rate = a.get("sample_rate", 16000)
            config.audio.channels = a.get("channels", 1)
            config.audio.frame_duration_ms = a.get("frame_duration_ms", 30)
            config.audio.vad_aggressiveness = a.get("vad_aggressiveness", 3)
            config.audio.input_device = a.get("input_device")

        if "ui" in data:
            u = data["ui"]
            config.ui.enabled = u.get("enabled", True)
            config.ui.position = u.get("position", "notch")
            config.ui.animation_enabled = u.get("animation_enabled", True)
            config.ui.sound_effects = u.get("sound_effects", True)
            config.ui.notifications = u.get("notifications", True)
            config.ui.verbosity = u.get("verbosity", "normal")

        if "permissions" in data:
            p = data["permissions"]
            config.permissions.allow_shell = p.get("allow_shell", True)
            config.permissions.allow_files = p.get("allow_files", True)
            config.permissions.allow_apps = p.get("allow_apps", True)
            config.permissions.allow_web = p.get("allow_web", True)
            config.permissions.allow_clipboard = p.get("allow_clipboard", False)
            config.permissions.private_paths = p.get("private_paths", [])

        if "logging" in data:
            log_data = data["logging"]
            config.logging.level = log_data.get("level", "INFO")
            default_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            config.logging.format = log_data.get("format", default_format)
            config.logging.file = log_data.get("file")
            config.logging.max_size_mb = log_data.get("max_size_mb", 10)
            config.logging.backup_count = log_data.get("backup_count", 5)

        return config

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.security.validate()
        self.wake.validate()
        self.stt.validate()
        self.llm.validate()
        self.audio.validate()
        self.ui.validate()
        self.permissions.validate()
        self.logging.validate()

    def scan_installed_apis(self) -> dict[str, list[str]]:
        """Scan system for installed APIs that can be used."""
        detected: dict[str, list[str]] = {"llm": [], "stt": []}

        # Check for local LLM tools
        llm_tools = [
            ("ollama", "ollama"),
            ("llama.cpp", "llama-server"),
            ("vllm", "vllm"),
            ("lmstudio", "lms"),
        ]
        for name, cmd in llm_tools:
            if shutil.which(cmd):
                detected["llm"].append(name)

        # Check for local STT tools
        stt_tools = [
            ("faster-whisper", "faster-whisper"),
            ("whisper.cpp", "whisper-cli"),
            ("mlx-whisper", "mlx_whisper"),
        ]
        for name, cmd in stt_tools:
            if shutil.which(cmd):
                detected["stt"].append(name)

        # Check for Python packages using importlib
        if importlib.util.find_spec("openai") is not None:
            detected["llm"].append("openai")

        if importlib.util.find_spec("anthropic") is not None:
            detected["llm"].append("anthropic")

        if importlib.util.find_spec("whisper") is not None:
            detected["stt"].append("openai-whisper")

        return detected
