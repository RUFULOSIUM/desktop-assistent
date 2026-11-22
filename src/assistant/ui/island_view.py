"""Island View - The visual component with metaball animation and state machine."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from AppKit import (
    NSBezierPath,
    NSColor,
    NSCompositingOperationSourceOver,
    NSFont,
    NSGraphicsContext,
    NSMakeRect,
    NSMakeSize,
    NSPoint,
    NSRect,
    NSRoundLineCapStyle,
    NSSize,
    NSView,
    NSPasteboard,
    NSPasteboardTypeFileURL,
    NSDragOperationCopy,
    NSDraggingInfo,
)
from Foundation import NSObject, NSURL

from .animation import (
    Animator,
    AnimationState,
    Easing,
    SpringConfig,
    animate_spring,
    get_animator,
)
from ._macos import screen_to_view_coords


class IslandState(Enum):
    """Visual states of the island."""

    IDLE = "idle"
    ACTIVATED = "activated"  # Wake detected, expanding from notch
    LISTENING = "listening"  # Pulsating with audio
    THINKING = "thinking"  # Loading spinner
    ACTING = "acting"  # Second cursor / task execution
    SPEAKING = "speaking"  # Pulsating with TTS
    HIDDEN = "hidden"  # Fully hidden (idle on notch Macs)


@dataclass
class IslandConfig:
    """Configuration for island appearance."""

    # Colors
    idle_color: NSColor = NSColor.colorWithCalibratedWhite_alpha_(0.15, 0.9)
    active_color: NSColor = NSColor.colorWithCalibratedWhite_alpha_(0.25, 0.95)
    thinking_color: NSColor = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.3, 0.6, 1.0, 0.95)
    acting_color: NSColor = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.6, 0.2, 0.95)
    speaking_color: NSColor = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.8, 0.4, 0.95)

    # Dimensions
    notch_width: float = 218.0
    notch_height: float = 36.0
    max_expanded_width: float = 400.0
    max_expanded_height: float = 64.0
    corner_radius: float = 18.0

    # Animation
    expansion_duration: float = 0.35
    contraction_duration: float = 0.3
    pulsate_frequency: float = 2.0  # Hz
    pulsate_amplitude: float = 0.08  # Scale factor

    # Spring configs for different transitions
    spring_expand: SpringConfig = field(default_factory=lambda: SpringConfig(stiffness=220, damping=22))
    spring_contract: SpringConfig = field(default_factory=lambda: SpringConfig(stiffness=180, damping=20))
    spring_bounce: SpringConfig = field(default_factory=lambda: SpringConfig(stiffness=280, damping=16))


class MetaballPath:
    """Generates metaball-style blob paths."""

    @staticmethod
    def create_pill(
        rect: NSRect,
        corner_radius: float,
        left_bulge: float = 0.0,
        right_bulge: float = 0.0,
    ) -> NSBezierPath:
        """Create a pill shape with optional side bulges (metaball effect)."""
        path = NSBezierPath.bezierPath()

        x, y = rect.origin.x, rect.origin.y
        w, h = rect.size.width, rect.size.height
        r = min(corner_radius, h / 2)

        # Left bulge center
        left_cx = x + r + left_bulge * (w / 4)
        # Right bulge center
        right_cx = x + w - r - right_bulge * (w / 4)
        cy = y + h / 2

        path.moveToPoint_(NSPoint(x + r, y))

        # Top edge to right
        path.lineToPoint_(NSPoint(x + w - r, y))
        # Top-right corner
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x + w, y), NSPoint(x + w, y + r), r
        )
        # Right edge with bulge
        if right_bulge > 0.01:
            path.lineToPoint_(NSPoint(x + w, right_cx - r))
            path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                NSPoint(right_cx, cy), r + right_bulge * (h / 2), 90, 270, True
            )
        else:
            path.lineToPoint_(NSPoint(x + w, y + h - r))

        # Bottom-right corner
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x + w, y + h), NSPoint(x + w - r, y + h), r
        )
        # Bottom edge to left
        path.lineToPoint_(NSPoint(x + r, y + h))
        # Bottom-left corner
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x, y + h), NSPoint(x, y + h - r), r
        )
        # Left edge with bulge
        if left_bulge > 0.01:
            path.lineToPoint_(NSPoint(x, left_cx + r))
            path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                NSPoint(left_cx, cy), r + left_bulge * (h / 2), 270, 90, True
            )
        else:
            path.lineToPoint_(NSPoint(x, y + r))

        # Top-left corner
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x, y), NSPoint(x + r, y), r
        )

        path.closePath()
        return path

    @staticmethod
    def create_spinner(
        center: NSPoint,
        radius: float,
        line_width: float = 3.0,
        start_angle: float = 0.0,
        end_angle: float = 270.0,
    ) -> NSBezierPath:
        """Create a spinner arc path."""
        path = NSBezierPath.bezierPath()
        path.setLineWidth_(line_width)
        path.setLineCapStyle_(NSRoundLineCapStyle)
        path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
            center, radius, start_angle, end_angle, False
        )
        return path

    @staticmethod
    def create_cursor(
        rect: NSRect,
        progress: float,  # 0 = pill, 1 = cursor
    ) -> NSBezierPath:
        """Morph between pill and cursor shape."""
        path = NSBezierPath.bezierPath()

        x, y = rect.origin.x, rect.origin.y
        w, h = rect.size.width, rect.size.height
        r = min(18.0, h / 2)

        # Cursor dimensions
        cursor_w = max(w * 0.3, 40.0)
        cursor_h = h * 1.5

        # Interpolate
        curr_w = w + (cursor_w - w) * progress
        curr_h = h + (cursor_h - h) * progress
        curr_r = r * (1 - progress * 0.5)

        path.moveToPoint_(NSPoint(x + curr_r, y))

        # Top edge
        path.lineToPoint_(NSPoint(x + curr_w - curr_r, y))
        # Top-right
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x + curr_w, y), NSPoint(x + curr_w, y + curr_r), curr_r
        )
        # Right edge
        path.lineToPoint_(NSPoint(x + curr_w, y + curr_h - curr_r))
        # Bottom-right (pointy for cursor)
        if progress > 0.5:
            # Pointy bottom
            path.lineToPoint_(NSPoint(x + curr_w / 2, y + curr_h))
            path.lineToPoint_(NSPoint(x, y + curr_h - curr_r))
        else:
            # Rounded bottom
            path.appendBezierPathWithArcFromPoint_toPoint_radius_(
                NSPoint(x + curr_w, y + curr_h), NSPoint(x + curr_w - curr_r, y + curr_h), curr_r
            )
            path.lineToPoint_(NSPoint(x + curr_r, y + curr_h))
            path.appendBezierPathWithArcFromPoint_toPoint_radius_(
                NSPoint(x, y + curr_h), NSPoint(x, y + curr_h - curr_r), curr_r
            )

        # Left edge
        path.lineToPoint_(NSPoint(x, y + curr_r))
        # Top-left
        path.appendBezierPathWithArcFromPoint_toPoint_radius_(
            NSPoint(x, y), NSPoint(x + curr_r, y), curr_r
        )

        path.closePath()
        return path


class IslandView(NSView):
    """Main island view with state-driven animation."""

    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the view (called from __init__)."""
        self._state = IslandState.IDLE
        self._config = IslandConfig()
        self._animator = get_animator()

        # Animation values
        self._expansion = 0.0  # 0 = notch size, 1 = fully expanded
        self._pulsate_phase = 0.0
        self._pulsate_scale = 1.0
        self._spinner_angle = 0.0
        self._cursor_progress = 0.0
        self._left_bulge = 0.0
        self._right_bulge = 0.0

        # Audio level for pulsating (0-1)
        self._audio_level = 0.0

        # Callbacks
        self._on_state_change: Optional[Callable[[IslandState], None]] = None
        self._on_drop: Optional[Callable[[list[str]], None]] = None

        # Animation states
        self._expansion_anim: Optional[AnimationState] = None
        self._pulsate_timer = None
        self._spinner_timer = None

        # Setup
        self._setup_timers()
        self._initialize_drag_and_drop()

    def _setup_timers(self) -> None:
        """Setup animation timers using NSTimer."""
        from AppKit import NSTimer

        # Pulsate timer (for listening/speaking)
        self._pulsate_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 60.0, self, "pulsateTick:", None, True
        )

        # Spinner timer (for thinking)
        self._spinner_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 60.0, self, "spinnerTick:", None, True
        )

    @classmethod
    def create(cls) -> "IslandView":
        """Factory method to create and initialize an IslandView."""
        view = cls.alloc().init()
        if view is None:
            view = cls.alloc().initWithFrame_(NSMakeRect(0, 0, 218, 36))
        return view

    def pulsateTick_(self, timer) -> None:
        """Update pulsate animation."""
        if self._state in (IslandState.LISTENING, IslandState.SPEAKING):
            self._pulsate_phase += self._config.pulsate_frequency * 2 * math.pi / 60.0
            # Combine audio level with base pulsate
            base_pulsate = 1.0 + self._config.pulsate_amplitude * math.sin(self._pulsate_phase)
            audio_pulsate = 1.0 + self._audio_level * 0.15
            self._pulsate_scale = base_pulsate * audio_pulsate
            self.setNeedsDisplay_(True)

    def spinnerTick_(self, timer) -> None:
        """Update spinner animation."""
        if self._state == IslandState.THINKING:
            self._spinner_angle += 12.0  # degrees per frame at 60fps
            if self._spinner_angle >= 360.0:
                self._spinner_angle -= 360.0
            self.setNeedsDisplay_(True)

    def set_audio_level(self, level: float) -> None:
        """Set audio level for pulsating (0.0 to 1.0)."""
        self._audio_level = max(0.0, min(1.0, level))

    def set_state(self, state: IslandState) -> None:
        """Change the island state with animation."""
        if state == self._state:
            return

        old_state = self._state
        self._state = state

        # Stop previous animations
        if self._expansion_anim:
            self._animator.stop(self._expansion_anim)
            self._expansion_anim = None

        # Start new animations based on state
        if state == IslandState.IDLE:
            self._animate_to_idle()
        elif state == IslandState.ACTIVATED:
            self._animate_activated()
        elif state == IslandState.LISTENING:
            self._animate_listening()
        elif state == IslandState.THINKING:
            self._animate_thinking()
        elif state == IslandState.ACTING:
            self._animate_acting()
        elif state == IslandState.SPEAKING:
            self._animate_speaking()
        elif state == IslandState.HIDDEN:
            self._animate_hidden()

        if self._on_state_change:
            self._on_state_change(state)

    def _animate_to_idle(self) -> None:
        """Animate to idle state (notch size, stationary)."""
        target_expansion = 0.0
        self._expansion_anim = self._animator.animate_spring(
            target=target_expansion,
            on_update=lambda v: setattr(self, "_expansion", v),
            on_complete=lambda: setattr(self, "_expansion", target_expansion),
            initial_value=self._expansion,
            spring_config=self._config.spring_contract,
        )
        self._cursor_progress = 0.0
        self._left_bulge = 0.0
        self._right_bulge = 0.0

    def _animate_hidden(self) -> None:
        """Animate to fully hidden (for notch Macs idle)."""
        target_expansion = 0.0
        self._expansion_anim = self._animator.animate_spring(
            target=target_expansion,
            on_update=lambda v: setattr(self, "_expansion", v),
            initial_value=self._expansion,
            spring_config=self._config.spring_contract,
        )

    def _animate_activated(self) -> None:
        """Animate wake activation - expand from notch with bulge."""
        target_expansion = 0.6
        self._expansion_anim = self._animator.animate_spring(
            target=target_expansion,
            on_update=lambda v: setattr(self, "_expansion", v),
            initial_value=self._expansion,
            spring_config=self._config.spring_expand,
        )
        # Add bulge effect during expansion
        self._left_bulge = 0.5
        self._right_bulge = 0.5

    def _animate_listening(self) -> None:
        """Animate listening - expanded, pulsating."""
        target_expansion = 1.0
        self._expansion_anim = self._animator.animate_spring(
            target=target_expansion,
            on_update=lambda v: setattr(self, "_expansion", v),
            initial_value=self._expansion,
            spring_config=self._config.spring_expand,
        )
        self._left_bulge = 0.3
        self._right_bulge = 0.3
        self._pulsate_phase = 0.0

    def _animate_thinking(self) -> None:
        """Animate thinking - spinner."""
        # Keep expanded but show spinner
        self._left_bulge = 0.0
        self._right_bulge = 0.0
        self._spinner_angle = 0.0

    def _animate_acting(self) -> None:
        """Animate acting - morph to cursor."""
        self._cursor_progress = 0.0
        # Animate cursor morph
        self._expansion_anim = self._animator.animate_spring(
            target=1.0,
            on_update=lambda v: setattr(self, "_cursor_progress", v),
            initial_value=self._cursor_progress,
            spring_config=self._config.spring_bounce,
        )
        self._left_bulge = 0.0
        self._right_bulge = 0.0

    def _animate_speaking(self) -> None:
        """Animate speaking - pulsating with TTS."""
        target_expansion = 1.0
        self._expansion_anim = self._animator.animate_spring(
            target=target_expansion,
            on_update=lambda v: setattr(self, "_expansion", v),
            initial_value=self._expansion,
            spring_config=self._config.spring_expand,
        )
        self._left_bulge = 0.2
        self._right_bulge = 0.2
        self._pulsate_phase = 0.0

    def drawRect_(self, dirty_rect: NSRect) -> None:
        """Draw the island based on current state."""
        context = NSGraphicsContext.currentContext()
        if context is None:
            return

        # Get bounds
        bounds = self.bounds()
        center_x = bounds.origin.x + bounds.size.width / 2
        center_y = bounds.origin.y + bounds.size.height / 2

        # Calculate current size based on expansion
        cfg = self._config
        base_w = cfg.notch_width
        base_h = cfg.notch_height
        max_w = cfg.max_expanded_width
        max_h = cfg.max_expanded_height

        curr_w = base_w + (max_w - base_w) * self._expansion
        curr_h = base_h + (max_h - base_h) * self._expansion

        # Apply pulsate scale
        curr_w *= self._pulsate_scale
        curr_h *= self._pulsate_scale

        # Center the rect
        rect = NSMakeRect(
            center_x - curr_w / 2,
            center_y - curr_h / 2,
            curr_w,
            curr_h,
        )

        # Choose color based on state
        if self._state == IslandState.IDLE or self._state == IslandState.HIDDEN:
            color = cfg.idle_color
        elif self._state == IslandState.ACTIVATED:
            color = cfg.active_color
        elif self._state == IslandState.LISTENING:
            color = cfg.active_color
        elif self._state == IslandState.THINKING:
            color = cfg.thinking_color
        elif self._state == IslandState.ACTING:
            color = cfg.acting_color
        elif self._state == IslandState.SPEAKING:
            color = cfg.speaking_color
        else:
            color = cfg.active_color

        # Draw based on state
        if self._state == IslandState.THINKING:
            self._draw_thinking(rect, color, center_x, center_y)
        elif self._state == IslandState.ACTING:
            self._draw_acting(rect, color, center_x, center_y)
        else:
            self._draw_pill(rect, color)

    def _draw_pill(self, rect: NSRect, color: NSColor) -> None:
        """Draw pill/metaball shape."""
        path = MetaballPath.create_pill(
            rect,
            self._config.corner_radius,
            left_bulge=self._left_bulge,
            right_bulge=self._right_bulge,
        )
        color.setFill()
        path.fill()

    def _draw_thinking(
        self, rect: NSRect, color: NSColor, center_x: float, center_y: float
    ) -> None:
        """Draw thinking state - pill with spinner."""
        # Draw background pill
        path = MetaballPath.create_pill(
            rect,
            self._config.corner_radius,
            left_bulge=0.0,
            right_bulge=0.0,
        )
        # Semi-transparent background
        bg_color = color.colorWithAlphaComponent_(0.3)
        bg_color.setFill()
        path.fill()

        # Draw spinner
        spinner_radius = min(rect.size.width, rect.size.height) / 3
        spinner_center = NSPoint(center_x, center_y)
        spinner_path = MetaballPath.create_spinner(
            spinner_center,
            spinner_radius,
            line_width=3.0,
            start_angle=self._spinner_angle,
            end_angle=self._spinner_angle + 270,
        )
        color.setStroke()
        spinner_path.stroke()

    def _draw_acting(
        self, rect: NSRect, color: NSColor, center_x: float, center_y: float
    ) -> None:
        """Draw acting state - morphing to cursor."""
        # Interpolate between pill and cursor
        path = MetaballPath.create_cursor(rect, self._cursor_progress)
        color.setFill()
        path.fill()

    def set_state_change_callback(self, callback: Callable[[IslandState], None]) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def set_drop_callback(self, callback: Callable[[list[str]], None]) -> None:
        """Set callback for file drops."""
        self._on_drop = callback

    def _initialize_drag_and_drop(self) -> None:
        """Register for drag-and-drop file types."""
        self.registerForDraggedTypes_([NSPasteboardTypeFileURL])

    def draggingEntered_(self, sender: NSDraggingInfo) -> int:
        """Called when a drag enters the view."""
        if self._is_valid_drop(sender):
            return NSDragOperationCopy
        return 0

    def draggingUpdated_(self, sender: NSDraggingInfo) -> int:
        """Called periodically during a drag."""
        if self._is_valid_drop(sender):
            return NSDragOperationCopy
        return 0

    def prepareForDragOperation_(self, sender: NSDraggingInfo) -> bool:
        """Called before the drop is performed."""
        return self._is_valid_drop(sender)

    def performDragOperation_(self, sender: NSDraggingInfo) -> bool:
        """Called to perform the drop operation."""
        file_paths = self._extract_file_paths(sender)
        if file_paths and hasattr(self, '_on_drop') and self._on_drop:
            self._on_drop(file_paths)
        return len(file_paths) > 0

    def concludeDragOperation_(self, sender: NSDraggingInfo) -> None:
        """Called after the drop operation completes."""
        pass

    def _is_valid_drop(self, sender: NSDraggingInfo) -> bool:
        """Check if the drag contains valid file URLs."""
        pasteboard = sender.draggingPasteboard()
        types = pasteboard.types()
        if types is None:
            return False
        return NSPasteboardTypeFileURL in types

    def _extract_file_paths(self, sender: NSDraggingInfo) -> list[str]:
        """Extract file paths from the drag pasteboard."""
        pasteboard = sender.draggingPasteboard()
        urls = pasteboard.readObjectsForClasses_options_([NSURL], None)
        if urls is None:
            return []
        paths = []
        for url in urls:
            if url.isFileURL():
                path = url.path()
                if path:
                    paths.append(path)
        return paths

    @property
    def state(self) -> IslandState:
        return self._state

    @property
    def config(self) -> IslandConfig:
        return self._config

    @config.setter
    def config(self, value: IslandConfig) -> None:
        self._config = value
        self.setNeedsDisplay_(True)