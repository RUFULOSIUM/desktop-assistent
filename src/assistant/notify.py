"""Notification system for the Desktop Assistant.

Uses macOS UserNotifications framework for low-verbosity notifications:
- Permission needed
- Task complete
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from Foundation import NSUserNotification, NSUserNotificationCenter

    _HAS_FOUNDATION = True
except ImportError:
    _HAS_FOUNDATION = False
    NSUserNotification = None
    NSUserNotificationCenter = None


class NotificationType(Enum):
    """Types of notifications the assistant can send."""

    PERMISSION_NEEDED = "permission_needed"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    WAKE_WORD_FAILED = "wake_word_failed"


@dataclass
class NotificationConfig:
    """Configuration for notifications."""

    enabled: bool = True
    # Only show notifications for these types
    allowed_types: set[NotificationType] = frozenset(
        {
            NotificationType.PERMISSION_NEEDED,
            NotificationType.TASK_COMPLETE,
            NotificationType.ERROR,
        }
    )
    # Verbosity filter: silent, minimal, normal, verbose
    min_verbosity: str = "minimal"


class NotificationManager:
    """Manages macOS user notifications for the assistant."""

    def __init__(self, config: NotificationConfig | None = None) -> None:
        self.config = config or NotificationConfig()
        self._center = None
        if _HAS_FOUNDATION:
            try:
                self._center = NSUserNotificationCenter.defaultUserNotificationCenter()
                # Allow notifications when app is active (may be None in tests)
                if self._center:
                    self._center.setDelegate_(self._center)
            except Exception:
                self._center = None

    def notify(
        self,
        title: str,
        subtitle: str = "",
        informative_text: str = "",
        notification_type: NotificationType = NotificationType.TASK_COMPLETE,
        verbosity: str = "normal",
    ) -> bool:
        """Show a notification if enabled and verbosity permits.

        Args:
            title: Notification title.
            subtitle: Optional subtitle.
            informative_text: Optional informative text.
            notification_type: Type of notification for filtering.
            verbosity: Current verbosity level (silent, minimal, normal, verbose).

        Returns:
            True if notification was shown, False otherwise.
        """
        if not self.config.enabled:
            return False

        if not _HAS_FOUNDATION or self._center is None:
            return False

        # Check verbosity
        verbosity_levels = {"silent": 0, "minimal": 1, "normal": 2, "verbose": 3}
        current_level = verbosity_levels.get(verbosity, 2)
        min_level = verbosity_levels.get(self.config.min_verbosity, 1)
        if current_level < min_level:
            return False

        # Check if this notification type is allowed
        if notification_type not in self.config.allowed_types:
            return False

        notification = NSUserNotification.alloc().init()
        notification.setTitle_(title)
        if subtitle:
            notification.setSubtitle_(subtitle)
        if informative_text:
            notification.setInformativeText_(informative_text)

        # Set a subtle sound for notifications
        notification.setSoundName_("default")

        self._center.deliverNotification_(notification)
        return True

    def notify_permission_needed(self, permission: str, verbosity: str = "normal") -> bool:
        """Notify that a permission is needed."""
        return self.notify(
            title="Permission Required",
            subtitle=f"Grant {permission} access",
            informative_text="Open System Settings > Privacy & Security to enable.",
            notification_type=NotificationType.PERMISSION_NEEDED,
            verbosity=verbosity,
        )

    def notify_task_complete(self, task_summary: str, verbosity: str = "normal") -> bool:
        """Notify that a task has completed."""
        return self.notify(
            title="Task Complete",
            subtitle=task_summary[:60] + ("..." if len(task_summary) > 60 else ""),
            informative_text="",
            notification_type=NotificationType.TASK_COMPLETE,
            verbosity=verbosity,
        )

    def notify_error(self, error: str, verbosity: str = "normal") -> bool:
        """Notify about an error."""
        return self.notify(
            title="Error",
            subtitle="Assistant encountered an issue",
            informative_text=error[:120],
            notification_type=NotificationType.ERROR,
            verbosity=verbosity,
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable notifications."""
        self.config.enabled = enabled

    def set_min_verbosity(self, verbosity: str) -> None:
        """Set minimum verbosity level for notifications."""
        self.config.min_verbosity = verbosity


# Global notification manager instance
_notification_manager: NotificationManager | None = None


def get_notification_manager(config: NotificationConfig | None = None) -> NotificationManager:
    """Get or create the global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(config)
    elif config is not None:
        _notification_manager.config = config
    return _notification_manager
