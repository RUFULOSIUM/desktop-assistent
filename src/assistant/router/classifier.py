"""Keras-based router classifier for chat vs agent decision."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RouterClassifier:
    """Small Keras model to classify requests as agent (True) or chat (False).

    The model takes up to 5000 characters as Unicode codepoints and outputs
    a single boolean: True = agent task, False = chat response.
    """

    MAX_INPUT_LENGTH = 5000
    MODEL_FILENAME = "router_classifier.keras"

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or self._default_model_path()
        self._model = None
        self._loaded = False

    @staticmethod
    def _default_model_path() -> Path:
        """Get the default model path."""
        return Path(__file__).parent / "models" / RouterClassifier.MODEL_FILENAME

    def _load_model(self) -> None:
        """Load the Keras model lazily."""
        if self._loaded:
            return

        try:
            import tensorflow as tf  # type: ignore[import-untyped]

            if self.model_path.exists():
                self._model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded router classifier from {self.model_path}")
            else:
                logger.warning(
                    f"Router model not found at {self.model_path}, using fallback"
                )
                self._model = self._create_fallback_model()
            self._loaded = True
        except ImportError:
            logger.warning("TensorFlow not installed, using fallback classifier")
            self._model = None
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load router model: {e}, using fallback")
            self._model = None
            self._loaded = True

    def _create_fallback_model(self) -> Any:
        """Create a simple fallback model that uses keyword heuristics."""
        # This is a placeholder - in reality we'd train a proper model
        return None

    def _encode_input(self, text: str) -> np.ndarray:
        """Encode text as Unicode codepoints, padded/truncated to MAX_INPUT_LENGTH."""
        # Convert to Unicode codepoints
        codepoints = [ord(c) for c in text[: self.MAX_INPUT_LENGTH]]

        # Pad or truncate
        if len(codepoints) < self.MAX_INPUT_LENGTH:
            codepoints.extend([0] * (self.MAX_INPUT_LENGTH - len(codepoints)))
        else:
            codepoints = codepoints[: self.MAX_INPUT_LENGTH]

        # Normalize to [0, 1] range (Unicode max is ~1.1M, but we use 0-65535 for BMP)
        arr = np.array(codepoints, dtype=np.float32) / 65535.0
        return arr.reshape(1, -1)

    def classify(self, text: str) -> bool:
        """Classify text as agent task (True) or chat (False).

        Args:
            text: Input text to classify (max 5000 chars).

        Returns:
            True if agent task, False if chat.
        """
        self._load_model()

        if self._model is not None:
            try:
                encoded = self._encode_input(text)
                prediction = self._model.predict(encoded, verbose=0)
                return bool(prediction[0][0] > 0.5)
            except Exception as e:
                logger.error(f"Router model prediction failed: {e}")

        # Fallback: keyword-based heuristic
        return self._fallback_classify(text)

    def _fallback_classify(self, text: str) -> bool:
        """Fallback classification using keywords."""
        text_lower = text.lower()

        # Agent task indicators
        agent_keywords = [
            "open", "launch", "start", "run", "execute", "create", "make", "build",
            "delete", "remove", "install", "update", "change", "modify", "edit",
            "write", "read", "copy", "move", "rename", "find", "search", "browse",
            "navigate", "go to", "click", "type", "press", "send", "download",
            "upload", "save", "export", "import", "convert", "transform", "process",
            "schedule", "remind", "notify", "alert", "mail", "email", "message",
            "call", "dial", "contact", "calendar", "event", "meeting", "appointment",
            "file", "folder", "directory", "document", "spreadsheet", "presentation",
            "image", "video", "audio", "music", "photo", "screenshot", "record",
            "chrome", "safari", "firefox", "terminal", "finder", "vscode", "code",
            "xcode", "docker", "git", "npm", "pip", "brew", "apt", "yum",
        ]

        # Chat indicators (questions, explanations, creative tasks)
        chat_keywords = [
            "what", "why", "how", "when", "where", "who", "which", "whose",
            "explain", "describe", "define", "meaning", "definition", "concept",
            "idea", "think", "opinion", "suggest", "recommend", "advice",
            "help", "understand", "learn", "teach", "show", "tell", "write",
            "story", "poem", "joke", "riddle", "quote", "fact", "trivia",
            "translate", "summarize", "compare", "difference", "similar",
        ]

        agent_score = sum(1 for kw in agent_keywords if kw in text_lower)
        chat_score = sum(1 for kw in chat_keywords if kw in text_lower)

        # If it looks like a command (starts with verb), lean towards agent
        first_word = text_lower.split()[0] if text_lower.split() else ""
        command_verbs = {
            "open",
            "run",
            "create",
            "delete",
            "install",
            "write",
            "read",
            "copy",
            "move",
            "find",
            "search",
        }
        if first_word in command_verbs:
            agent_score += 2

        return agent_score > chat_score


def create_classifier(model_path: Path | None = None) -> RouterClassifier:
    """Factory function to create a router classifier."""
    return RouterClassifier(model_path)