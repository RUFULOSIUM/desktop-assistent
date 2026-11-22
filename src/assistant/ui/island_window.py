"""Island Window - NSPanel controller for the Island UI."""

from __future__ import annotations

from typing import Callable, Optional

from AppKit import (
    NSApplication,
    NSMakeRect,
    NSMakeSize,
    NSPoint,
    NSRect,
    NSScreen,
    NSView,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSWindow,
    NSWindowCollectionBehavior,
    NSWindowLevel,
)
from Foundation import NSObject

from ._macos import create_island_panel, IslandWindowDelegate
from .island_view import IslandState, IslandView
from ..platform.notch import detect_notch, get_island_position


class IslandWindow:
    """
    Controller for the Island window.

    Manages the NSPanel lifecycle, positioning, and state synchronization
    with the IslandView.
    """

    def __init__(self, config_position: str = "notch"):
        self._config_position = config_position
        self._panel: Optional[NSWindow] = None
        self._view: Optional[IslandView] = None
        self._delegate = IslandWindowDelegate.alloc().init()
        self._is_visible = False
        self._animation_enabled = True

        # Callbacks to pipeline
        self._on_wake_detected: Optional[Callable[[], None]] = None
        self._on_listening: Optional[Callable[[], None]] = None
        self._on_thinking: Optional[Callable[[], None]] = None
        self._on_speaking: Optional[Callable[[str], None]] = None
        self._on_idle: Optional[Callable[[], None]] = None

        # Notch info
        self._notch_info = detect_notch()

    def initialize(self) -> None:
        """Create and configure the window and view."""
        if self._panel is not None:
            return

        # Get initial position
        pos = get_island_position(self._config_position)
        rect = NSMakeRect(pos[0], pos[1], pos[2], pos[3])

        # Create panel
        self._panel = create_island_panel(rect)
        self._panel.setDelegate_(self._delegate)

        # Create view
        self._view = IslandView.alloc().initWithFrame_(self._panel.contentView().bounds())
        self._view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self._panel.contentView().addSubview_(self._view)

        # Set up state change callback
        self._view.set_state_change_callback(self._handle_view_state_change)

        # Set up drag-and-drop callback
        self._view.set_drop_callback(self._handle_files_dropped)

        # Initially hidden for notch Macs in idle
        if self._notch_info.has_notch and self._config_position == "notch":
            self._panel.orderOut_(self._panel)
            self._is_visible = False
        else:
            self._panel.orderFrontRegardless()
            self._is_visible = True

    def _handle_view_state_change(self, state: IslandState) -> None:
        """Handle view state changes - show/hide window and bridge to pipeline callbacks."""
        # For notch Macs: hide when IDLE or HIDDEN, show otherwise
        # For non-notch Macs: always show (handled by initialize)
        is_notch_mac = self._notch_info.has_notch and self._config_position == "notch"

        if is_notch_mac:
            if state in (IslandState.IDLE, IslandState.HIDDEN):
                self.hide()
            else:
                self.show()
        else:
            # Non-notch Macs: always visible, just ensure shown
            self.show()

        # Bridge to pipeline callbacks
        if state == IslandState.ACTIVATED and self._on_wake_detected:
            self._on_wake_detected()
        elif state == IslandState.LISTENING and self._on_listening:
            self._on_listening()
        elif state == IslandState.THINKING and self._on_thinking:
            self._on_thinking()
        elif state == IslandState.SPEAKING and self._on_speaking:
            self._on_speaking("")
        elif state in (IslandState.IDLE, IslandState.HIDDEN) and self._on_idle:
            self._on_idle()

    def show(self) -> None:
        """Show the island window."""
        if self._panel and not self._is_visible:
            self._panel.orderFrontRegardless()
            self._is_visible = True

    def hide(self) -> None:
        """Hide the island window."""
        if self._panel and self._is_visible:
            self._panel.orderOut_(self._panel)
            self._is_visible = False

    def set_state(self, state: IslandState) -> None:
        """Set the island visual state."""
        if self._view:
            self._view.set_state(state)

    def set_audio_level(self, level: float) -> None:
        """Set audio level for pulsating animation."""
        if self._view:
            self._view.set_audio_level(level)

    def set_position(self, position: str) -> None:
        """Update window position (notch, menu_bar, center)."""
        self._config_position = position
        self._notch_info = detect_notch()
        self._reposition()

    def _reposition(self) -> None:
        """Reposition window based on current config."""
        if not self._panel:
            return

        pos = get_island_position(self._config_position)
        rect = NSMakeRect(pos[0], pos[1], pos[2], pos[3])
        self._panel.setFrame_display_(rect, True)

    def set_animation_enabled(self, enabled: bool) -> None:
        """Enable/disable animations."""
        self._animation_enabled = enabled
        if self._view:
            # View handles its own animation config
            pass

    def is_visible(self) -> bool:
        return self._is_visible

    def get_view(self) -> Optional[IslandView]:
        return self._view

    def shutdown(self) -> None:
        """Clean up window resources."""
        if self._panel:
            self._panel.close()
            self._panel = None
        self._view = None

    # Pipeline callback setters
    def set_callbacks(
        self,
        on_wake_detected: Optional[Callable[[], None]] = None,
        on_listening: Optional[Callable[[], None]] = None,
        on_thinking: Optional[Callable[[], None]] = None,
        on_speaking: Optional[Callable[[str], None]] = None,
        on_idle: Optional[Callable[[], None]] = None,
        on_files_dropped: Optional[Callable[[list[str]], None]] = None,
    ) -> None:
        """Set pipeline callbacks."""
        self._on_wake_detected = on_wake_detected
        self._on_listening = on_listening
        self._on_thinking = on_thinking
        self._on_speaking = on_speaking
        self._on_idle = on_idle
        self._on_files_dropped = on_files_dropped

    def _handle_files_dropped(self, file_paths: list[str]) -> None:
        """Handle files dropped on the island."""
        if self._on_files_dropped:
            self._on_files_dropped(file_paths)


def create_island_window(config_position: str = "notch") -> IslandWindow:
    """Factory function to create an island window."""
    return IslandWindow(config_position)