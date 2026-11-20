"""Local LLM provider implementations (Ollama, llama.cpp, etc.)."""

from __future__ import annotations

import json
import shutil
from typing import Any

import requests

from . import LLMProvider, LLMProviderNotAvailableError, LLMResponse


class LocalLLMProvider(LLMProvider):
    """Local LLM provider supporting Ollama, llama.cpp server, and LM Studio."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._model = config.local_model
        self._base_url = self._detect_server()
        self._session = requests.Session()
        self._started = False

    @property
    def name(self) -> str:
        return f"local:{self._model}"

    @property
    def is_local(self) -> bool:
        return True

    def _detect_server(self) -> str:
        """Detect which local LLM server is running."""
        # Check for Ollama
        if shutil.which("ollama"):
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    return "http://localhost:11434"
            except Exception:
                pass

        # Check for llama.cpp server
        if shutil.which("llama-server"):
            try:
                resp = requests.get("http://localhost:8080/health", timeout=2)
                if resp.status_code == 200:
                    return "http://localhost:8080"
            except Exception:
                pass

        # Check for LM Studio
        if shutil.which("lms"):
            try:
                resp = requests.get("http://localhost:1234/v1/models", timeout=2)
                if resp.status_code == 200:
                    return "http://localhost:1234/v1"
            except Exception:
                pass

        # No server found
        raise LLMProviderNotAvailableError(
            "No local LLM server found. Install Ollama, llama.cpp, or LM Studio."
        )

    def start(self) -> None:
        """Initialize the provider."""
        if self._started:
            return

        # Verify server is reachable
        try:
            if "11434" in self._base_url:
                resp = self._session.get(f"{self._base_url}/api/tags", timeout=5)
            elif "8080" in self._base_url:
                resp = self._session.get(f"{self._base_url}/health", timeout=5)
            elif "1234" in self._base_url:
                resp = self._session.get(f"{self._base_url}/models", timeout=5)
            else:
                resp = self._session.get(f"{self._base_url}/models", timeout=5)

            if resp.status_code != 200:
                raise LLMProviderNotAvailableError(
                    f"Local LLM server not responding at {self._base_url}"
                )
        except Exception as e:
            raise LLMProviderNotAvailableError(
                f"Cannot connect to local LLM server at {self._base_url}: {e}"
            ) from e

        self._started = True

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

        if "11434" in self._base_url:
            return self._complete_ollama(prompt, temperature, max_tokens)
        elif "8080" in self._base_url:
            return self._complete_llamacpp(prompt, temperature, max_tokens)
        else:
            return self._complete_openai_compatible(prompt, temperature, max_tokens)

    def _complete_ollama(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Complete using Ollama API."""
        resp = self._session.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(text=data.get("response", ""), metadata=data)

    def _complete_llamacpp(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Complete using llama.cpp server API."""
        resp = self._session.post(
            f"{self._base_url}/completion",
            json={
                "prompt": prompt,
                "temperature": temperature,
                "n_predict": max_tokens,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(text=data.get("content", ""), metadata=data)

    def _complete_openai_compatible(
    self, prompt: str, temperature: float, max_tokens: int
) -> LLMResponse:
        """Complete using OpenAI-compatible API (LM Studio, vLLM, etc.)."""
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
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

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Stream completions."""
        if not self._started:
            self.start()

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        if "11434" in self._base_url:
            return self._stream_ollama(prompt, temperature, max_tokens)
        elif "8080" in self._base_url:
            return self._stream_llamacpp(prompt, temperature, max_tokens)
        else:
            return self._stream_openai_compatible(prompt, temperature, max_tokens)

    def _stream_ollama(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Stream from Ollama."""
        resp = self._session.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield LLMResponse(text=data.get("response", ""), metadata=data)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    def _stream_llamacpp(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Stream from llama.cpp."""
        resp = self._session.post(
            f"{self._base_url}/completion",
            json={
                "prompt": prompt,
                "temperature": temperature,
                "n_predict": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode().removeprefix("data: "))
                    yield LLMResponse(text=data.get("content", ""), metadata=data)
                    if data.get("stop", False):
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue

    def _stream_openai_compatible(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Stream from OpenAI-compatible API."""
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
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
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield LLMResponse(text=content, metadata=data)
                    except json.JSONDecodeError:
                        continue