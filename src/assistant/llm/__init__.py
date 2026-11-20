"""LLM provider abstraction for the Desktop Assistant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether this provider runs locally."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Initialize the provider (load model, connect to API, etc.)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Clean up resources."""
        ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            LLMResponse with the generated text and metadata.
        """
        ...

    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Stream completions for the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific options.

        Yields:
            Partial LLMResponse objects as tokens are generated.
        """
        ...


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMProviderNotAvailableError(LLMError):
    """Raised when a provider is not available (not installed, no API key, etc.)."""

    pass


def create_llm_provider(config: Any) -> LLMProvider:
    """Factory function to create an LLM provider based on configuration.

    Args:
        config: LLMConfig object with provider settings.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        LLMProviderNotAvailableError: If the requested provider is not available.
    """
    provider = config.provider.lower()

    if provider == "local":
        from .local import LocalLLMProvider

        return LocalLLMProvider(config)
    elif provider in ("openai", "anthropic"):
        from .remote import RemoteLLMProvider

        return RemoteLLMProvider(config)
    elif provider == "auto":
        # Try local first, then remote
        try:
            from .local import LocalLLMProvider

            local_provider = LocalLLMProvider(config)
            local_provider.start()  # Verify server is reachable
            return local_provider
        except LLMProviderNotAvailableError:
            from .remote import RemoteLLMProvider

            return RemoteLLMProvider(config)
    else:
        raise LLMProviderNotAvailableError(f"Unknown LLM provider: {provider}")