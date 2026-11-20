"""Wake word detection module."""

from assistant.errors import WakeError
from assistant.wake.openwakeword_engine import WakeWordEngine

__all__ = ["WakeWordEngine", "WakeError"]
