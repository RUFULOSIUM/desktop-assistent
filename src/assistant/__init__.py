"""Desktop Assistant - Voice-first macOS assistant living at the camera notch."""

from .hotkeys import (
    HotkeyConfig,
    HotkeyManager,
    create_hotkey_manager,
)
from .notify import (
    NotificationConfig,
    NotificationManager,
    NotificationType,
    get_notification_manager,
)

__version__ = "0.1.0"
__all__ = [
    "HotkeyConfig",
    "HotkeyManager",
    "create_hotkey_manager",
    "NotificationType",
    "NotificationConfig",
    "NotificationManager",
    "get_notification_manager",
]
