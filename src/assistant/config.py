"""Configuration system for the Desktop Assistant."""

from __future__ import annotations

import importlib.util
import json
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
    sensitivity: float = 0.5
    custom_models_dir: str | None = None

    def validate(self) -> None:
        if not self.word:
            raise ValueError("Wake word cannot be empty")
        if len(self.word) > 50:
            raise ValueError("Wake word too long (max 50 characters)")
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0")


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    provider: str = "auto"  # cloud, local, auto
    cloud_service: str = "whisper-api"  # whisper-api, google, azure
    local_model: str = "base"  # tiny, base, small, medium, large
    language: str = "en"
    fallback_chain: list[str] = field(default_factory=lambda: ["cloud", "local"])
    cloud_timeout: int = 30
    local_compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True

    def validate(self) -> None:
        valid_providers = ("cloud", "local", "auto")
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid STT provider: {self.provider}. Must be one of {valid_providers}"
            )
        if self.cloud_timeout < 1:
            raise ValueError("Cloud timeout must be at least 1 second")
        if self.beam_size < 1:
            raise ValueError("Beam size must be at least 1")


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "auto"  # local, openai, anthropic, auto
    local_model: str = "llama3"
    remote_model: str = "gpt-4"
    api_key: str = ""
    fallback_chain: list[str] = field(default_factory=lambda: ["local", "remote"])
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    system_prompt: str = ""
    context_window: int = 8192

    def validate(self) -> None:
        valid_providers = ("local", "openai", "anthropic", "auto")
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be one of {valid_providers}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be at least 1")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("Presence penalty must be between -2.0 and 2.0")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("Frequency penalty must be between -2.0 and 2.0")


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 3
    input_device: str | None = None
    output_device: str | None = None
    input_gain: float = 1.0
    output_volume: float = 1.0
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True

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
        if self.input_gain < 0.0:
            raise ValueError("Input gain must be non-negative")
        if not 0.0 <= self.output_volume <= 2.0:
            raise ValueError("Output volume must be between 0.0 and 2.0")


@dataclass
class SoundConfig:
    """Sound effects configuration."""

    enabled: bool = True
    volume: float = 0.5  # 0.0 to 1.0

    # Custom sound file paths (None = use system defaults)
    wake_detected: str | None = None
    listening_start: str | None = None
    listening_end: str | None = None
    thinking_start: str | None = None
    speaking_start: str | None = None
    task_complete: str | None = None
    error: str | None = None
    permission_needed: str | None = None

    def validate(self) -> None:
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("Sound volume must be between 0.0 and 1.0")
        for attr in (
            "wake_detected",
            "listening_start",
            "listening_end",
            "thinking_start",
            "speaking_start",
            "task_complete",
            "error",
            "permission_needed",
        ):
            path = getattr(self, attr)
            if path and not Path(path).expanduser().exists():
                raise ValueError(f"Sound file not found: {path}")


@dataclass
class TTSSystemConfig:
    """System TTS (macOS say) configuration."""

    voice: str = "Samantha"
    rate: int = 180
    volume: float = 1.0


@dataclass
class TTS_PiperConfig:
    """Piper TTS configuration."""

    model_path: str | None = None
    voice: str = "en_US-lessac-medium"
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence: float = 0.2


@dataclass
class TTS_CoquiConfig:
    """Coqui TTS configuration."""

    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    speaker_id: str | None = None
    language_id: str | None = None


@dataclass
class TTSElevenLabsConfig:
    """ElevenLabs TTS configuration."""

    api_key: str = ""
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


@dataclass
class TTSOpenAIConfig:
    """OpenAI TTS configuration."""

    api_key: str = ""
    voice: str = "nova"
    model: str = "tts-1-hd"
    speed: float = 1.0


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""

    provider: str = "system"  # system, piper, coqui, elevenlabs, openai
    system: TTSSystemConfig = field(default_factory=TTSSystemConfig)
    piper: TTS_PiperConfig = field(default_factory=TTS_PiperConfig)
    coqui: TTS_CoquiConfig = field(default_factory=TTS_CoquiConfig)
    elevenlabs: TTSElevenLabsConfig = field(default_factory=TTSElevenLabsConfig)
    openai: TTSOpenAIConfig = field(default_factory=TTSOpenAIConfig)
    cache_enabled: bool = True
    cache_dir: str = "~/.cache/desktop-assistant/tts"
    stream: bool = False


@dataclass
class VoicePersonaConfig:
    """Voice persona configuration."""

    name: str
    description: str
    system_prompt_addendum: str


@dataclass
class VoiceConfig:
    """Voice personalization configuration."""

    persona: str = "assistant"
    custom_personas: dict[str, VoicePersonaConfig] = field(default_factory=dict)
    language: str = "en"
    wake_word_custom: bool = False
    pronunciation_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.custom_personas:
            self.custom_personas = {
                "assistant": VoicePersonaConfig(
                    name="Assistant",
                    description="Helpful, concise, professional",
                    system_prompt_addendum="You are a helpful assistant. Be concise and direct.",
                ),
                "creative": VoicePersonaConfig(
                    name="Creative",
                    description="Imaginative, expressive, conversational",
                    system_prompt_addendum="You are a creative companion. Be expressive and engaging.",
                ),
                "technical": VoicePersonaConfig(
                    name="Technical",
                    description="Precise, detailed, code-focused",
                    system_prompt_addendum="You are a technical expert. Be precise and thorough.",
                ),
            }


@dataclass
class UIMenuBarConfig:
    """Menu bar configuration."""

    enabled: bool = True
    show_status: bool = True
    icon_style: str = "emoji"  # emoji, text, icon


@dataclass
class UIIslandConfig:
    """Island UI configuration."""

    expanded_width: float = 400.0
    expanded_height: float = 64.0
    pulsate_amplitude: float = 0.08
    pulsate_frequency: float = 2.0
    spring_stiffness: float = 220.0
    spring_damping: float = 22.0


@dataclass
class UIConfig:
    """UI configuration."""

    enabled: bool = True
    position: str = "notch"  # notch, menu_bar, center
    animation_enabled: bool = True
    sound_effects: bool = True
    notifications: bool = True
    verbosity: str = "normal"  # silent, minimal, normal, verbose
    theme: str = "auto"  # auto, light, dark
    opacity: float = 0.95
    corner_radius: float = 18.0
    idle_timeout: int = 30
    show_transcript: bool = True
    transcript_lines: int = 3
    menu_bar: UIMenuBarConfig = field(default_factory=UIMenuBarConfig)
    island: UIIslandConfig = field(default_factory=UIIslandConfig)

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
        valid_themes = ("auto", "light", "dark")
        if self.theme not in valid_themes:
            raise ValueError(f"Invalid theme: {self.theme}. Must be one of {valid_themes}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("Opacity must be between 0.0 and 1.0")
        if self.idle_timeout < 0:
            raise ValueError("Idle timeout must be non-negative")


@dataclass
class HotkeysConfig:
    """Hotkeys configuration."""

    triple_shift: bool = True
    double_click: bool = True
    push_to_talk: str | None = None
    toggle_listening: str | None = None
    show_hide_window: str = "cmd+shift+space"


@dataclass
class PermissionsConfig:
    """Permissions configuration."""

    allow_shell: bool = True
    allow_files: bool = True
    allow_apps: bool = True
    allow_web: bool = True
    allow_clipboard: bool = False
    private_paths: list[str] = field(default_factory=list)
    confirm_destructive: bool = True
    confirm_network: bool = True
    max_file_size_mb: int = 100
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    def validate(self) -> None:
        for path in self.private_paths:
            p = Path(path).expanduser()
            if not p.exists():
                raise ValueError(f"Private path does not exist: {path}")
        if self.max_file_size_mb < 1:
            raise ValueError("Max file size must be at least 1 MB")


@dataclass
class PrivacyConfig:
    """Privacy configuration."""

    telemetry: bool = False
    crash_reports: bool = False
    usage_stats: bool = False
    local_only_mode: bool = False
    data_retention_days: int = 30
    auto_delete_transcripts: bool = True

    def validate(self) -> None:
        if self.data_retention_days < 0:
            raise ValueError("Data retention days must be non-negative")


@dataclass
class AgentConfig:
    """Agent configuration."""

    enabled: bool = True
    pi_binary: str = "pi"
    working_dir: str = "~"
    timeout: int = 300
    max_concurrent_tasks: int = 3
    default_permissions: list[str] = field(default_factory=lambda: ["shell", "files"])
    confirmation_required: bool = True
    presets_dir: str = "~/.config/desktop-assistant/agent/presets"

    def validate(self) -> None:
        if self.timeout < 1:
            raise ValueError("Timeout must be at least 1 second")
        if self.max_concurrent_tasks < 1:
            raise ValueError("Max concurrent tasks must be at least 1")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: str | None = None
    max_size_mb: int = 10
    backup_count: int = 5
    console_colors: bool = True
    json_format: bool = False

    def validate(self) -> None:
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        if self.max_size_mb < 1:
            raise ValueError("Max size must be at least 1 MB")
        if self.backup_count < 0:
            raise ValueError("Backup count must be non-negative")


@dataclass
class PerformanceConfig:
    """Performance configuration."""

    cpu_threads: int = 0  # 0 = auto
    gpu_acceleration: bool = False
    low_power_mode: bool = False
    preload_models: bool = False
    model_cache_size_mb: int = 500

    def validate(self) -> None:
        if self.cpu_threads < 0:
            raise ValueError("CPU threads must be non-negative")
        if self.model_cache_size_mb < 10:
            raise ValueError("Model cache size must be at least 10 MB")


@dataclass
class Config:
    """Main configuration class."""

    security: SecurityConfig = field(default_factory=SecurityConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def get_default(cls) -> Config:
        """Create a configuration with default values."""
        return cls()

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load configuration from TOML or JSON file."""
        if path is None:
            path = cls.get_default_config_path()

        if not path.exists():
            # Try JSON settings file as fallback
            json_path = path.with_suffix(".json")
            if json_path.exists():
                return cls.load_json(json_path)
            return cls.get_default()

        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls.load_json(path)
        else:
            # Default to TOML
            with open(path, "rb") as f:
                data = tomli.load(f)
            return cls.from_dict(data)

    @classmethod
    def load_json(cls, path: Path) -> Config:
        """Load configuration from JSON file."""
        if not path.exists():
            return cls.get_default()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

        if path.suffix.lower() == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "wb") as f:
                tomli_w.dump(data, f)

    def save_json(self, path: Path | None = None) -> None:
        """Save configuration to JSON file."""
        if path is None:
            path = self.get_default_config_path().with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""

        def _filter_none(d: dict[str, Any]) -> dict[str, Any]:
            result = {}
            for k, v in d.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    filtered = _filter_none(v)
                    if filtered:  # Only add non-empty dicts
                        result[k] = filtered
                else:
                    result[k] = v
            return result

        return {
            "security": {"level": self.security.level},
            "wake": _filter_none(
                {
                    "enabled": self.wake.enabled,
                    "word": self.wake.word,
                    "fallback_double_click": self.wake.fallback_double_click,
                    "fallback_triple_shift": self.wake.fallback_triple_shift,
                    "sensitivity": self.wake.sensitivity,
                    "custom_models_dir": self.wake.custom_models_dir,
                }
            ),
            "stt": _filter_none(
                {
                    "provider": self.stt.provider,
                    "cloud_service": self.stt.cloud_service,
                    "local_model": self.stt.local_model,
                    "language": self.stt.language,
                    "fallback_chain": self.stt.fallback_chain,
                    "cloud_timeout": self.stt.cloud_timeout,
                    "local_compute_type": self.stt.local_compute_type,
                    "beam_size": self.stt.beam_size,
                    "vad_filter": self.stt.vad_filter,
                }
            ),
            "llm": _filter_none(
                {
                    "provider": self.llm.provider,
                    "local_model": self.llm.local_model,
                    "remote_model": self.llm.remote_model,
                    "api_key": self.llm.api_key or None,
                    "fallback_chain": self.llm.fallback_chain,
                    "temperature": self.llm.temperature,
                    "max_tokens": self.llm.max_tokens,
                    "top_p": self.llm.top_p,
                    "presence_penalty": self.llm.presence_penalty,
                    "frequency_penalty": self.llm.frequency_penalty,
                    "system_prompt": self.llm.system_prompt or None,
                    "context_window": self.llm.context_window,
                }
            ),
            "audio": _filter_none(
                {
                    "sample_rate": self.audio.sample_rate,
                    "channels": self.audio.channels,
                    "frame_duration_ms": self.audio.frame_duration_ms,
                    "vad_aggressiveness": self.audio.vad_aggressiveness,
                    "input_device": self.audio.input_device,
                    "output_device": self.audio.output_device,
                    "input_gain": self.audio.input_gain,
                    "output_volume": self.audio.output_volume,
                    "echo_cancellation": self.audio.echo_cancellation,
                    "noise_suppression": self.audio.noise_suppression,
                    "auto_gain_control": self.audio.auto_gain_control,
                }
            ),
            "sound": _filter_none(
                {
                    "enabled": self.sound.enabled,
                    "volume": self.sound.volume,
                    "wake_detected": self.sound.wake_detected,
                    "listening_start": self.sound.listening_start,
                    "listening_end": self.sound.listening_end,
                    "thinking_start": self.sound.thinking_start,
                    "speaking_start": self.sound.speaking_start,
                    "task_complete": self.sound.task_complete,
                    "error": self.sound.error,
                    "permission_needed": self.sound.permission_needed,
                }
            ),
            "tts": _filter_none(
                {
                    "provider": self.tts.provider,
                    "system": {
                        "voice": self.tts.system.voice,
                        "rate": self.tts.system.rate,
                        "volume": self.tts.system.volume,
                    },
                    "piper": {
                        "model_path": self.tts.piper.model_path,
                        "voice": self.tts.piper.voice,
                        "length_scale": self.tts.piper.length_scale,
                        "noise_scale": self.tts.piper.noise_scale,
                        "noise_w": self.tts.piper.noise_w,
                        "sentence_silence": self.tts.piper.sentence_silence,
                    },
                    "coqui": {
                        "model_name": self.tts.coqui.model_name,
                        "speaker_id": self.tts.coqui.speaker_id,
                        "language_id": self.tts.coqui.language_id,
                    },
                    "elevenlabs": {
                        "api_key": self.tts.elevenlabs.api_key or None,
                        "voice_id": self.tts.elevenlabs.voice_id,
                        "model_id": self.tts.elevenlabs.model_id,
                        "stability": self.tts.elevenlabs.stability,
                        "similarity_boost": self.tts.elevenlabs.similarity_boost,
                        "style": self.tts.elevenlabs.style,
                        "use_speaker_boost": self.tts.elevenlabs.use_speaker_boost,
                    },
                    "openai": {
                        "api_key": self.tts.openai.api_key or None,
                        "voice": self.tts.openai.voice,
                        "model": self.tts.openai.model,
                        "speed": self.tts.openai.speed,
                    },
                    "cache_enabled": self.tts.cache_enabled,
                    "cache_dir": self.tts.cache_dir,
                    "stream": self.tts.stream,
                }
            ),
            "voice": _filter_none(
                {
                    "persona": self.voice.persona,
                    "custom_personas": {
                        k: {
                            "name": v.name,
                            "description": v.description,
                            "system_prompt_addendum": v.system_prompt_addendum,
                        }
                        for k, v in self.voice.custom_personas.items()
                    },
                    "language": self.voice.language,
                    "wake_word_custom": self.voice.wake_word_custom,
                    "pronunciation_rules": self.voice.pronunciation_rules,
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
                    "theme": self.ui.theme,
                    "opacity": self.ui.opacity,
                    "corner_radius": self.ui.corner_radius,
                    "idle_timeout": self.ui.idle_timeout,
                    "show_transcript": self.ui.show_transcript,
                    "transcript_lines": self.ui.transcript_lines,
                    "menu_bar": {
                        "enabled": self.ui.menu_bar.enabled,
                        "show_status": self.ui.menu_bar.show_status,
                        "icon_style": self.ui.menu_bar.icon_style,
                    },
                    "island": {
                        "expanded_width": self.ui.island.expanded_width,
                        "expanded_height": self.ui.island.expanded_height,
                        "pulsate_amplitude": self.ui.island.pulsate_amplitude,
                        "pulsate_frequency": self.ui.island.pulsate_frequency,
                        "spring_stiffness": self.ui.island.spring_stiffness,
                        "spring_damping": self.ui.island.spring_damping,
                    },
                }
            ),
            "hotkeys": _filter_none(
                {
                    "triple_shift": self.hotkeys.triple_shift,
                    "double_click": self.hotkeys.double_click,
                    "push_to_talk": self.hotkeys.push_to_talk,
                    "toggle_listening": self.hotkeys.toggle_listening,
                    "show_hide_window": self.hotkeys.show_hide_window,
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
                    "confirm_destructive": self.permissions.confirm_destructive,
                    "confirm_network": self.permissions.confirm_network,
                    "max_file_size_mb": self.permissions.max_file_size_mb,
                    "allowed_domains": self.permissions.allowed_domains,
                    "blocked_domains": self.permissions.blocked_domains,
                }
            ),
            "privacy": _filter_none(
                {
                    "telemetry": self.privacy.telemetry,
                    "crash_reports": self.privacy.crash_reports,
                    "usage_stats": self.privacy.usage_stats,
                    "local_only_mode": self.privacy.local_only_mode,
                    "data_retention_days": self.privacy.data_retention_days,
                    "auto_delete_transcripts": self.privacy.auto_delete_transcripts,
                }
            ),
            "agent": _filter_none(
                {
                    "enabled": self.agent.enabled,
                    "pi_binary": self.agent.pi_binary,
                    "working_dir": self.agent.working_dir,
                    "timeout": self.agent.timeout,
                    "max_concurrent_tasks": self.agent.max_concurrent_tasks,
                    "default_permissions": self.agent.default_permissions,
                    "confirmation_required": self.agent.confirmation_required,
                    "presets_dir": self.agent.presets_dir,
                }
            ),
            "logging": _filter_none(
                {
                    "level": self.logging.level,
                    "format": self.logging.format,
                    "file": self.logging.file,
                    "max_size_mb": self.logging.max_size_mb,
                    "backup_count": self.logging.backup_count,
                    "console_colors": self.logging.console_colors,
                    "json_format": self.logging.json_format,
                }
            ),
            "performance": _filter_none(
                {
                    "cpu_threads": self.performance.cpu_threads,
                    "gpu_acceleration": self.performance.gpu_acceleration,
                    "low_power_mode": self.performance.low_power_mode,
                    "preload_models": self.performance.preload_models,
                    "model_cache_size_mb": self.performance.model_cache_size_mb,
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
            config.wake.sensitivity = w.get("sensitivity", 0.5)
            config.wake.custom_models_dir = w.get("custom_models_dir")

        if "stt" in data:
            s = data["stt"]
            config.stt.provider = s.get("provider", "auto")
            config.stt.cloud_service = s.get("cloud_service", "whisper-api")
            config.stt.local_model = s.get("local_model", "base")
            config.stt.language = s.get("language", "en")
            config.stt.fallback_chain = s.get("fallback_chain", ["cloud", "local"])
            config.stt.cloud_timeout = s.get("cloud_timeout", 30)
            config.stt.local_compute_type = s.get("local_compute_type", "float16")
            config.stt.beam_size = s.get("beam_size", 5)
            config.stt.vad_filter = s.get("vad_filter", True)

        if "llm" in data:
            llm_data = data["llm"]
            config.llm.provider = llm_data.get("provider", "auto")
            config.llm.local_model = llm_data.get("local_model", "llama3")
            config.llm.remote_model = llm_data.get("remote_model", "gpt-4")
            config.llm.api_key = llm_data.get("api_key", "")
            config.llm.fallback_chain = llm_data.get("fallback_chain", ["local", "remote"])
            config.llm.temperature = llm_data.get("temperature", 0.7)
            config.llm.max_tokens = llm_data.get("max_tokens", 2048)
            config.llm.top_p = llm_data.get("top_p", 0.9)
            config.llm.presence_penalty = llm_data.get("presence_penalty", 0.0)
            config.llm.frequency_penalty = llm_data.get("frequency_penalty", 0.0)
            config.llm.system_prompt = llm_data.get("system_prompt", "")
            config.llm.context_window = llm_data.get("context_window", 8192)

        if "audio" in data:
            a = data["audio"]
            config.audio.sample_rate = a.get("sample_rate", 16000)
            config.audio.channels = a.get("channels", 1)
            config.audio.frame_duration_ms = a.get("frame_duration_ms", 30)
            config.audio.vad_aggressiveness = a.get("vad_aggressiveness", 3)
            config.audio.input_device = a.get("input_device")
            config.audio.output_device = a.get("output_device")
            config.audio.input_gain = a.get("input_gain", 1.0)
            config.audio.output_volume = a.get("output_volume", 1.0)
            config.audio.echo_cancellation = a.get("echo_cancellation", True)
            config.audio.noise_suppression = a.get("noise_suppression", True)
            config.audio.auto_gain_control = a.get("auto_gain_control", True)

        if "sound" in data:
            s = data["sound"]
            config.sound.enabled = s.get("enabled", True)
            config.sound.volume = s.get("volume", 0.5)
            config.sound.wake_detected = s.get("wake_detected")
            config.sound.listening_start = s.get("listening_start")
            config.sound.listening_end = s.get("listening_end")
            config.sound.thinking_start = s.get("thinking_start")
            config.sound.speaking_start = s.get("speaking_start")
            config.sound.task_complete = s.get("task_complete")
            config.sound.error = s.get("error")
            config.sound.permission_needed = s.get("permission_needed")

        if "tts" in data:
            t = data["tts"]
            config.tts.provider = t.get("provider", "system")
            config.tts.cache_enabled = t.get("cache_enabled", True)
            config.tts.cache_dir = t.get("cache_dir", "~/.cache/desktop-assistant/tts")
            config.tts.stream = t.get("stream", False)

            if "system" in t:
                sys_t = t["system"]
                config.tts.system.voice = sys_t.get("voice", "Samantha")
                config.tts.system.rate = sys_t.get("rate", 180)
                config.tts.system.volume = sys_t.get("volume", 1.0)

            if "piper" in t:
                p = t["piper"]
                config.tts.piper.model_path = p.get("model_path")
                config.tts.piper.voice = p.get("voice", "en_US-lessac-medium")
                config.tts.piper.length_scale = p.get("length_scale", 1.0)
                config.tts.piper.noise_scale = p.get("noise_scale", 0.667)
                config.tts.piper.noise_w = p.get("noise_w", 0.8)
                config.tts.piper.sentence_silence = p.get("sentence_silence", 0.2)

            if "coqui" in t:
                c = t["coqui"]
                config.tts.coqui.model_name = c.get(
                    "model_name", "tts_models/en/ljspeech/tacotron2-DDC"
                )
                config.tts.coqui.speaker_id = c.get("speaker_id")
                config.tts.coqui.language_id = c.get("language_id")

            if "elevenlabs" in t:
                e = t["elevenlabs"]
                config.tts.elevenlabs.api_key = e.get("api_key", "")
                config.tts.elevenlabs.voice_id = e.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
                config.tts.elevenlabs.model_id = e.get("model_id", "eleven_multilingual_v2")
                config.tts.elevenlabs.stability = e.get("stability", 0.5)
                config.tts.elevenlabs.similarity_boost = e.get("similarity_boost", 0.75)
                config.tts.elevenlabs.style = e.get("style", 0.0)
                config.tts.elevenlabs.use_speaker_boost = e.get("use_speaker_boost", True)

            if "openai" in t:
                o = t["openai"]
                config.tts.openai.api_key = o.get("api_key", "")
                config.tts.openai.voice = o.get("voice", "nova")
                config.tts.openai.model = o.get("model", "tts-1-hd")
                config.tts.openai.speed = o.get("speed", 1.0)

        if "voice" in data:
            v = data["voice"]
            config.voice.persona = v.get("persona", "assistant")
            config.voice.language = v.get("language", "en")
            config.voice.wake_word_custom = v.get("wake_word_custom", False)
            config.voice.pronunciation_rules = v.get("pronunciation_rules", {})

            if "custom_personas" in v:
                for k, p in v["custom_personas"].items():
                    config.voice.custom_personas[k] = VoicePersonaConfig(
                        name=p.get("name", k),
                        description=p.get("description", ""),
                        system_prompt_addendum=p.get("system_prompt_addendum", ""),
                    )

        if "ui" in data:
            u = data["ui"]
            config.ui.enabled = u.get("enabled", True)
            config.ui.position = u.get("position", "notch")
            config.ui.animation_enabled = u.get("animation_enabled", True)
            config.ui.sound_effects = u.get("sound_effects", True)
            config.ui.notifications = u.get("notifications", True)
            config.ui.verbosity = u.get("verbosity", "normal")
            config.ui.theme = u.get("theme", "auto")
            config.ui.opacity = u.get("opacity", 0.95)
            config.ui.corner_radius = u.get("corner_radius", 18.0)
            config.ui.idle_timeout = u.get("idle_timeout", 30)
            config.ui.show_transcript = u.get("show_transcript", True)
            config.ui.transcript_lines = u.get("transcript_lines", 3)

            if "menu_bar" in u:
                mb = u["menu_bar"]
                config.ui.menu_bar.enabled = mb.get("enabled", True)
                config.ui.menu_bar.show_status = mb.get("show_status", True)
                config.ui.menu_bar.icon_style = mb.get("icon_style", "emoji")

            if "island" in u:
                i = u["island"]
                config.ui.island.expanded_width = i.get("expanded_width", 400.0)
                config.ui.island.expanded_height = i.get("expanded_height", 64.0)
                config.ui.island.pulsate_amplitude = i.get("pulsate_amplitude", 0.08)
                config.ui.island.pulsate_frequency = i.get("pulsate_frequency", 2.0)
                config.ui.island.spring_stiffness = i.get("spring_stiffness", 220.0)
                config.ui.island.spring_damping = i.get("spring_damping", 22.0)

        if "hotkeys" in data:
            h = data["hotkeys"]
            config.hotkeys.triple_shift = h.get("triple_shift", True)
            config.hotkeys.double_click = h.get("double_click", True)
            config.hotkeys.push_to_talk = h.get("push_to_talk")
            config.hotkeys.toggle_listening = h.get("toggle_listening")
            config.hotkeys.show_hide_window = h.get("show_hide_window", "cmd+shift+space")

        if "permissions" in data:
            p = data["permissions"]
            config.permissions.allow_shell = p.get("allow_shell", True)
            config.permissions.allow_files = p.get("allow_files", True)
            config.permissions.allow_apps = p.get("allow_apps", True)
            config.permissions.allow_web = p.get("allow_web", True)
            config.permissions.allow_clipboard = p.get("allow_clipboard", False)
            config.permissions.private_paths = p.get("private_paths", [])
            config.permissions.confirm_destructive = p.get("confirm_destructive", True)
            config.permissions.confirm_network = p.get("confirm_network", True)
            config.permissions.max_file_size_mb = p.get("max_file_size_mb", 100)
            config.permissions.allowed_domains = p.get("allowed_domains", [])
            config.permissions.blocked_domains = p.get("blocked_domains", [])

        if "privacy" in data:
            pr = data["privacy"]
            config.privacy.telemetry = pr.get("telemetry", False)
            config.privacy.crash_reports = pr.get("crash_reports", False)
            config.privacy.usage_stats = pr.get("usage_stats", False)
            config.privacy.local_only_mode = pr.get("local_only_mode", False)
            config.privacy.data_retention_days = pr.get("data_retention_days", 30)
            config.privacy.auto_delete_transcripts = pr.get("auto_delete_transcripts", True)

        if "agent" in data:
            ag = data["agent"]
            config.agent.enabled = ag.get("enabled", True)
            config.agent.pi_binary = ag.get("pi_binary", "pi")
            config.agent.working_dir = ag.get("working_dir", "~")
            config.agent.timeout = ag.get("timeout", 300)
            config.agent.max_concurrent_tasks = ag.get("max_concurrent_tasks", 3)
            config.agent.default_permissions = ag.get("default_permissions", ["shell", "files"])
            config.agent.confirmation_required = ag.get("confirmation_required", True)
            config.agent.presets_dir = ag.get(
                "presets_dir", "~/.config/desktop-assistant/agent/presets"
            )

        if "logging" in data:
            log_data = data["logging"]
            config.logging.level = log_data.get("level", "INFO")
            default_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            config.logging.format = log_data.get("format", default_format)
            config.logging.file = log_data.get("file")
            config.logging.max_size_mb = log_data.get("max_size_mb", 10)
            config.logging.backup_count = log_data.get("backup_count", 5)
            config.logging.console_colors = log_data.get("console_colors", True)
            config.logging.json_format = log_data.get("json_format", False)

        if "performance" in data:
            perf = data["performance"]
            config.performance.cpu_threads = perf.get("cpu_threads", 0)
            config.performance.gpu_acceleration = perf.get("gpu_acceleration", False)
            config.performance.low_power_mode = perf.get("low_power_mode", False)
            config.performance.preload_models = perf.get("preload_models", False)
            config.performance.model_cache_size_mb = perf.get("model_cache_size_mb", 500)

        return config

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.security.validate()
        self.wake.validate()
        self.stt.validate()
        self.llm.validate()
        self.audio.validate()
        self.sound.validate()
        self.ui.validate()
        self.permissions.validate()
        self.privacy.validate()
        self.agent.validate()
        self.logging.validate()
        self.performance.validate()

    def scan_installed_apis(self) -> dict[str, list[str]]:
        """Scan system for installed APIs that can be used."""
        detected: dict[str, list[str]] = {"llm": [], "stt": [], "tts": []}

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

        # Check for local TTS tools
        tts_tools = [
            ("piper", "piper"),
            ("coqui-tts", "tts"),
        ]
        for name, cmd in tts_tools:
            if shutil.which(cmd):
                detected["tts"].append(name)

        # Check for Python packages using importlib
        if importlib.util.find_spec("openai") is not None:
            detected["llm"].append("openai")
            detected["tts"].append("openai")

        if importlib.util.find_spec("anthropic") is not None:
            detected["llm"].append("anthropic")

        if importlib.util.find_spec("whisper") is not None:
            detected["stt"].append("openai-whisper")

        if importlib.util.find_spec("elevenlabs") is not None:
            detected["tts"].append("elevenlabs")

        return detected
