"""Sound effects system for the Desktop Assistant.

Plays short audio cues for different pipeline states using macOS NSSound.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from AppKit import NSSound


class SoundEvent(Enum):
    """Sound events that can trigger audio cues."""

    WAKE_DETECTED = "wake_detected"
    LISTENING_START = "listening_start"
    LISTENING_END = "listening_end"
    THINKING_START = "thinking_start"
    SPEAKING_START = "speaking_start"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    PERMISSION_NEEDED = "permission_needed"


@dataclass
class SoundConfig:
    """Configuration for sound effects."""

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

    # System sound names (macOS built-in)
    _system_sounds: dict[SoundEvent, str] = field(
        default_factory=lambda: {
            SoundEvent.WAKE_DETECTED: "Glass",
            SoundEvent.LISTENING_START: "Pop",
            SoundEvent.LISTENING_END: "Pop",
            SoundEvent.THINKING_START: "Basso",
            SoundEvent.SPEAKING_START: "Tink",
            SoundEvent.TASK_COMPLETE: "Hero",
            SoundEvent.ERROR: "Basso",
            SoundEvent.PERMISSION_NEEDED: "Sosumi",
        },
        init=False,
    )

    def get_sound_name(self, event: SoundEvent) -> str:
        """Get the sound file path or system sound name for an event."""
        custom_attr = event.value
        custom_path = getattr(self, custom_attr, None)
        if custom_path and Path(custom_path).expanduser().exists():
            return str(Path(custom_path).expanduser())
        return self._system_sounds.get(event, "Pop")


class SoundManager:
    """Manages and plays sound effects for the assistant."""

    def __init__(self, config: SoundConfig | None = None) -> None:
        self.config = config or SoundConfig()
        self._sounds: dict[SoundEvent, NSSound] = {}
        self._lock = threading.Lock()
        self._preload()

    def _preload(self) -> None:
        """Preload all sound effects."""
        for event in SoundEvent:
            self._load_sound(event)

    def _load_sound(self, event: SoundEvent) -> None:
        """Load a sound for an event."""
        with self._lock:
            if event in self._sounds:
                return

            sound_name = self.config.get_sound_name(event)
            sound: NSSound | None = None

            if os.path.isfile(sound_name):
                # Custom sound file
                sound = NSSound.alloc().initWithContentsOfFile_byReference_(sound_name, True)
            else:
                # System sound
                sound = NSSound.soundNamed_(sound_name)

            if sound:
                sound.setVolume_(self.config.volume)
                self._sounds[event] = sound

    def play(self, event: SoundEvent) -> bool:
        """Play a sound effect for the given event.

        Args:
            event: The sound event to play.

        Returns:
            True if sound was played, False if disabled or sound not found.
        """
        if not self.config.enabled:
            return False

        with self._lock:
            sound = self._sounds.get(event)
            if sound is None:
                # Try to load on demand
                self._load_sound(event)
                sound = self._sounds.get(event)

            if sound:
                sound.play()
                return True

        return False

    def set_volume(self, volume: float) -> None:
        """Set volume for all sounds (0.0 to 1.0)."""
        volume = max(0.0, min(1.0, volume))
        self.config.volume = volume
        with self._lock:
            for sound in self._sounds.values():
                sound.setVolume_(volume)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable sound effects."""
        self.config.enabled = enabled


# Global sound manager instance
_sound_manager: SoundManager | None = None


def get_sound_manager(config: SoundConfig | None = None) -> SoundManager:
    """Get or create the global sound manager."""
    global _sound_manager
    if _sound_manager is None:
        _sound_manager = SoundManager(config)
    elif config is not None:
        _sound_manager.config = config
        _sound_manager._preload()
    return _sound_manager


def play_sound(event: SoundEvent) -> bool:
    """Convenience function to play a sound event."""
    manager = get_sound_manager()
    return manager.play(event)
