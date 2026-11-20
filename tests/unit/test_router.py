"""Tests for router module."""

from pathlib import Path
from unittest.mock import MagicMock

from assistant.router import (
    AGENT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    RouterClassifier,
    create_classifier,
)


def test_router_classifier_init() -> None:
    """Test RouterClassifier initialization."""
    classifier = RouterClassifier()
    assert classifier.MAX_INPUT_LENGTH == 5000
    assert not classifier._loaded


def test_router_classifier_fallback_chat() -> None:
    """Test fallback classification for chat-like input."""
    classifier = RouterClassifier()
    # Questions should be classified as chat
    assert classifier._fallback_classify("What is the weather?") is False
    assert classifier._fallback_classify("How do I do this?") is False
    assert classifier._fallback_classify("Why does this happen?") is False
    assert classifier._fallback_classify("Explain this concept") is False


def test_router_classifier_fallback_agent() -> None:
    """Test fallback classification for agent-like input."""
    classifier = RouterClassifier()
    # Commands should be classified as agent
    assert classifier._fallback_classify("Open Chrome") is True
    assert classifier._fallback_classify("Create a file") is True
    assert classifier._fallback_classify("Run the script") is True
    assert classifier._fallback_classify("Delete the folder") is True


def test_router_classifier_encode_input() -> None:
    """Test input encoding."""
    classifier = RouterClassifier()
    encoded = classifier._encode_input("test")
    assert encoded.shape == (1, 5000)
    assert encoded.dtype == "float32"
    # Check normalization
    assert encoded.max() <= 1.0
    assert encoded.min() >= 0.0


def test_router_classifier_encode_long_input() -> None:
    """Test input encoding truncates long input."""
    classifier = RouterClassifier()
    long_text = "a" * 10000
    encoded = classifier._encode_input(long_text)
    assert encoded.shape == (1, 5000)


def test_router_classifier_classify_without_model() -> None:
    """Test classification falls back when no model loaded."""
    classifier = RouterClassifier()
    # Should use fallback (no model loaded)
    result = classifier.classify("Open Chrome")
    assert result is True
    result = classifier.classify("What is the weather?")
    assert result is False


def test_router_classifier_with_mock_model() -> None:
    """Test classification with mocked Keras model."""
    classifier = RouterClassifier()
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.8]]  # > 0.5 = agent
    classifier._model = mock_model
    classifier._loaded = True

    result = classifier.classify("any text")
    assert result is True
    mock_model.predict.assert_called_once()


def test_router_classifier_with_mock_model_chat() -> None:
    """Test classification with mocked Keras model returning chat."""
    classifier = RouterClassifier()
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.3]]  # < 0.5 = chat
    classifier._model = mock_model
    classifier._loaded = True

    result = classifier.classify("any text")
    assert result is False


def test_create_classifier() -> None:
    """Test factory function."""
    classifier = create_classifier()
    assert isinstance(classifier, RouterClassifier)

    custom_path = Path("/custom/path/model.keras")
    classifier2 = create_classifier(custom_path)
    assert classifier2.model_path == custom_path


def test_prompts() -> None:
    """Test system prompts are defined."""
    assert CHAT_SYSTEM_PROMPT
    assert AGENT_SYSTEM_PROMPT
    assert "voice assistant" in CHAT_SYSTEM_PROMPT.lower()
    assert "pi-mono" in AGENT_SYSTEM_PROMPT.lower()
    assert "json" in AGENT_SYSTEM_PROMPT.lower()