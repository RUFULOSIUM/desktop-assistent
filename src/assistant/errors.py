"""Error hierarchy for the Desktop Assistant."""

from __future__ import annotations

from typing import Any


class AssistantError(Exception):
    """Base exception for all assistant errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        base = self.message
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} ({ctx_str})"
        return base


class ConfigError(AssistantError):
    """Configuration-related errors."""


class AudioError(AssistantError):
    """Audio-related errors."""

    def __init__(self, message: str, device: str | None = None, **context: Any) -> None:
        super().__init__(message, device=device, **context)
        self.device = device


class WakeError(AssistantError):
    """Wake word detection errors."""

    def __init__(self, message: str, model: str | None = None, **context: Any) -> None:
        super().__init__(message, model=model, **context)
        self.model = model


class STTError(AssistantError):
    """Speech-to-text errors."""

    def __init__(self, message: str, provider: str | None = None, **context: Any) -> None:
        super().__init__(message, provider=provider, **context)
        self.provider = provider


class LLMError(AssistantError):
    """LLM provider errors."""

    def __init__(self, message: str, model: str | None = None, **context: Any) -> None:
        super().__init__(message, model=model, **context)
        self.model = model


class AgentError(AssistantError):
    """Agent dispatch errors."""

    def __init__(self, message: str, task: str | None = None, **context: Any) -> None:
        super().__init__(message, task=task, **context)
        self.task = task


class UIError(AssistantError):
    """UI-related errors."""

    def __init__(self, message: str, state: str | None = None, **context: Any) -> None:
        super().__init__(message, state=state, **context)
        self.state = state


class PermissionError(AssistantError):
    """Permission-related errors."""

    def __init__(self, message: str, required: str | None = None, **context: Any) -> None:
        super().__init__(message, required=required, **context)
        self.required = required


class ValidationError(AssistantError):
    """Validation errors."""

    def __init__(
        self, message: str, field: str | None = None, value: Any = None, **context: Any
    ) -> None:
        super().__init__(message, field=field, value=value, **context)
        self.field = field
        self.value = value


class NotFoundError(AssistantError):
    """Resource not found errors."""

    def __init__(self, message: str, resource: str | None = None, **context: Any) -> None:
        super().__init__(message, resource=resource, **context)
        self.resource = resource


class InitializationError(AssistantError):
    """Initialization errors."""

    def __init__(self, message: str, component: str | None = None, **context: Any) -> None:
        super().__init__(message, component=component, **context)
        self.component = component
