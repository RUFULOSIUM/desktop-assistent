"""Remote LLM provider implementations (OpenAI, Anthropic, etc.)."""

from __future__ import annotations

from typing import Any

import requests

from . import LLMProvider, LLMProviderNotAvailableError, LLMResponse


class RemoteLLMProvider(LLMProvider):
    """Remote LLM provider for OpenAI-compatible and Anthropic APIs."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._provider = config.provider.lower()
        self._model = config.remote_model
        self._api_key = config.api_key
        self._base_url = self._get_base_url()
        self._session = requests.Session()
        self._started = False

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._model}"

    @property
    def is_local(self) -> bool:
        return False

    def _get_base_url(self) -> str:
        if self._provider == "openai":
            return "https://api.openai.com/v1"
        elif self._provider == "anthropic":
            return "https://api.anthropic.com/v1"
        return "https://api.openai.com/v1"

    def start(self) -> None:
        """Initialize the provider."""
        if self._started:
            return

        if not self._api_key:
            raise LLMProviderNotAvailableError(
                f"API key required for {self._provider} provider. "
                "Set it in config or environment variable."
            )

        # Verify API key works
        try:
            headers = self._get_headers()
            resp = self._session.get(f"{self._base_url}/models", headers=headers, timeout=10)
            if resp.status_code == 401:
                raise LLMProviderNotAvailableError(f"Invalid API key for {self._provider}")
            resp.raise_for_status()
        except Exception as e:
            raise LLMProviderNotAvailableError(
                f"Cannot connect to {self._provider} API: {e}"
            ) from e

        self._started = True

    def _get_headers(self) -> dict[str, str]:
        if self._provider == "openai":
            return {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
        elif self._provider == "anthropic":
            return {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def stop(self) -> None:
        """Clean up resources."""
        self._session.close()
        self._started = False

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion."""
        if not self._started:
            self.start()

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        if self._provider == "openai":
            return self._complete_openai(prompt, temperature, max_tokens)
        elif self._provider == "anthropic":
            return self._complete_anthropic(prompt, temperature, max_tokens)
        else:
            return self._complete_openai(prompt, temperature, max_tokens)

    def _complete_openai(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Complete using OpenAI API."""
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(text=content, metadata=data)

    def _complete_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Complete using Anthropic API."""
        resp = self._session.post(
            f"{self._base_url}/messages",
            headers=self._get_headers(),
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [{}])[0].get("text", "")
        return LLMResponse(text=content, metadata=data)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Stream completions."""
        if not self._started:
            self.start()

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        if self._provider == "openai":
            return self._stream_openai(prompt, temperature, max_tokens)
        elif self._provider == "anthropic":
            return self._stream_anthropic(prompt, temperature, max_tokens)
        else:
            return self._stream_openai(prompt, temperature, max_tokens)

    def _stream_openai(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Stream from OpenAI."""
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                line_str = line.decode()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json

                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield LLMResponse(text=content, metadata=data)
                    except json.JSONDecodeError:
                        continue

    def _stream_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Stream from Anthropic."""
        resp = self._session.post(
            f"{self._base_url}/messages",
            headers=self._get_headers(),
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                line_str = line.decode()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    try:
                        import json

                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            content = delta.get("text", "")
                            if content:
                                yield LLMResponse(text=content, metadata=data)
                    except json.JSONDecodeError:
                        continue