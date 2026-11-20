"""Agent module for dispatching tasks to pi-mono."""

from __future__ import annotations

from .dispatch import AgentDispatcher, DispatchResult, create_dispatcher
from .presets import get_preset, list_presets

__all__ = [
    "AgentDispatcher",
    "DispatchResult",
    "create_dispatcher",
    "get_preset",
    "list_presets",
]