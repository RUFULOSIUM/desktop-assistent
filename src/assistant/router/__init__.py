"""Router module for classifying requests as chat vs agent."""

from __future__ import annotations

from .classifier import RouterClassifier, create_classifier
from .prompts import AGENT_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT

__all__ = [
    "RouterClassifier",
    "create_classifier",
    "CHAT_SYSTEM_PROMPT",
    "AGENT_SYSTEM_PROMPT",
]