"""Wake word detection module."""

from assistant.errors import WakeError
from assistant.wake.sherpa_engine import SherpaWakeEngine

WakeWordEngine = SherpaWakeEngine  # Backward compatibility alias

__all__ = ["SherpaWakeEngine", "WakeWordEngine", "WakeError"]