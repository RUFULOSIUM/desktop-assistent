"""Tests for clipboard module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from assistant.platform.clipboard import (
    has_clipboard_text,
    read_clipboard_text,
    write_clipboard_text,
)


class TestClipboard:
    """Tests for clipboard functions."""

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_read_clipboard_text_success(self, mock_pasteboard_class: MagicMock) -> None:
        """Test reading text from clipboard."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.types.return_value = ["NSStringPboardType"]
        mock_pasteboard.stringForType_.return_value = "Hello, clipboard!"

        result = read_clipboard_text()

        assert result == "Hello, clipboard!"
        mock_pasteboard.stringForType_.assert_called_once()

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_read_clipboard_text_empty(self, mock_pasteboard_class: MagicMock) -> None:
        """Test reading from empty clipboard."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.types.return_value = []

        result = read_clipboard_text()

        assert result is None

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_read_clipboard_text_no_string_type(self, mock_pasteboard_class: MagicMock) -> None:
        """Test reading clipboard with no string type."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.types.return_value = ["NSFilenamesPboardType"]

        result = read_clipboard_text()

        assert result is None

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_read_clipboard_text_none_pasteboard(self, mock_pasteboard_class: MagicMock) -> None:
        """Test reading when pasteboard is None."""
        mock_pasteboard_class.generalPasteboard.return_value = None

        result = read_clipboard_text()

        assert result is None

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_has_clipboard_text_true(self, mock_pasteboard_class: MagicMock) -> None:
        """Test has_clipboard_text returns True when text available."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.types.return_value = ["NSStringPboardType"]

        result = has_clipboard_text()

        assert result is True

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_has_clipboard_text_false(self, mock_pasteboard_class: MagicMock) -> None:
        """Test has_clipboard_text returns False when no text."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.types.return_value = []

        result = has_clipboard_text()

        assert result is False

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_write_clipboard_text_success(self, mock_pasteboard_class: MagicMock) -> None:
        """Test writing text to clipboard."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.setString_forType_.return_value = True

        result = write_clipboard_text("Test text")

        assert result is True
        mock_pasteboard.clearContents.assert_called_once()
        mock_pasteboard.setString_forType_.assert_called_once_with(
            "Test text", "NSStringPboardType"
        )

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_write_clipboard_text_failure(self, mock_pasteboard_class: MagicMock) -> None:
        """Test writing text to clipboard failure."""
        mock_pasteboard = MagicMock()
        mock_pasteboard_class.generalPasteboard.return_value = mock_pasteboard
        mock_pasteboard.setString_forType_.return_value = False

        result = write_clipboard_text("Test text")

        assert result is False

    @patch("assistant.platform.clipboard.NSPasteboard")
    def test_write_clipboard_text_none_pasteboard(self, mock_pasteboard_class: MagicMock) -> None:
        """Test writing when pasteboard is None."""
        mock_pasteboard_class.generalPasteboard.return_value = None

        result = write_clipboard_text("Test text")

        assert result is False
