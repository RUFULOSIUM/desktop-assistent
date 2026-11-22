"""Clipboard utilities for macOS."""

from __future__ import annotations

from AppKit import NSPasteboard, NSStringPboardType


def read_clipboard_text() -> str | None:
    """
    Read text content from the system clipboard (general pasteboard).

    Returns:
        Clipboard text content, or None if clipboard is empty or doesn't contain text.
    """
    pasteboard = NSPasteboard.generalPasteboard()
    if pasteboard is None:
        return None

    # Check if clipboard contains string data
    types = pasteboard.types()
    if not types or NSStringPboardType not in types:
        return None

    # Read the string
    content = pasteboard.stringForType_(NSStringPboardType)
    if content is None:
        return None

    return str(content)


def has_clipboard_text() -> bool:
    """Check if clipboard contains text without reading it."""
    pasteboard = NSPasteboard.generalPasteboard()
    if pasteboard is None:
        return False

    types = pasteboard.types()
    return types is not None and NSStringPboardType in types


def write_clipboard_text(text: str) -> bool:
    """
    Write text to the system clipboard.

    Args:
        text: Text to write to clipboard.

    Returns:
        True if successful, False otherwise.
    """
    pasteboard = NSPasteboard.generalPasteboard()
    if pasteboard is None:
        return False

    pasteboard.clearContents()
    return bool(pasteboard.setString_forType_(text, NSStringPboardType))
