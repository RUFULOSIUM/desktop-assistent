"""Tests for LLM module."""

from unittest.mock import MagicMock, patch

from assistant.config import LLMConfig
from assistant.llm import (
    LLMProviderNotAvailableError,
    LLMResponse,
    create_llm_provider,
)
from assistant.llm.local import LocalLLMProvider
from assistant.llm.remote import RemoteLLMProvider


def test_llm_provider_base() -> None:
    """Test LLMProvider abstract base class."""
    from assistant.llm import LLMProvider

    class TestProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "test"

        @property
        def is_local(self) -> bool:
            return True

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def complete(self, _prompt: str, **_kwargs) -> LLMResponse:
            return LLMResponse(text="test")

        def stream_complete(self, _prompt: str, **_kwargs):
            yield LLMResponse(text="test")

    provider = TestProvider()
    assert provider.name == "test"
    assert provider.is_local is True
    assert provider.complete("test").text == "test"


def test_create_llm_provider_local() -> None:
    """Test creating local LLM provider."""
    config = LLMConfig(provider="local", local_model="test-model")

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        provider = create_llm_provider(config)
        assert isinstance(provider, LocalLLMProvider)
        assert provider._model == "test-model"


def test_create_llm_provider_remote_openai() -> None:
    """Test creating OpenAI remote provider."""
    config = LLMConfig(provider="openai", remote_model="gpt-4", api_key="test-key")

    provider = create_llm_provider(config)
    assert isinstance(provider, RemoteLLMProvider)
    assert provider._model == "gpt-4"
    assert provider._api_key == "test-key"


def test_create_llm_provider_remote_anthropic() -> None:
    """Test creating Anthropic remote provider."""
    config = LLMConfig(provider="anthropic", remote_model="claude-3", api_key="test-key")

    provider = create_llm_provider(config)
    assert isinstance(provider, RemoteLLMProvider)
    assert provider._model == "claude-3"


def test_create_llm_provider_auto_local_available() -> None:
    """Test auto provider uses local when available."""
    config = LLMConfig(
        provider="auto", local_model="test", remote_model="gpt-4", api_key="test-key"
    )

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch("assistant.llm.local.requests.Session.post") as mock_post,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        # Mock the start() verification call
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_post.return_value = mock_post_resp

        provider = create_llm_provider(config)
        assert isinstance(provider, LocalLLMProvider)


def test_create_llm_provider_auto_fallback() -> None:
    """Test auto provider falls back to remote when local unavailable."""
    config = LLMConfig(
        provider="auto", local_model="test", remote_model="gpt-4", api_key="test-key"
    )

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch("assistant.llm.local.requests.Session.post") as mock_post,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        # Mock start() to raise connection error
        mock_post_resp = MagicMock()
        mock_post_resp.raise_for_status.side_effect = ConnectionError("Connection refused")
        mock_post.return_value = mock_post_resp

        provider = create_llm_provider(config)
        assert isinstance(provider, RemoteLLMProvider)


def test_create_llm_provider_invalid() -> None:
    """Test creating provider with invalid config raises error."""
    config = LLMConfig(provider="invalid", local_model="test")

    try:
        create_llm_provider(config)
        raise AssertionError("Should have raised LLMProviderNotAvailableError")
    except LLMProviderNotAvailableError as e:
        assert "Unknown LLM provider" in str(e)


def test_local_provider_detect_server_ollama() -> None:
    """Test Ollama server detection."""
    config = LLMConfig(provider="local", local_model="llama3")

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        provider = LocalLLMProvider(config)
        assert provider._base_url == "http://localhost:11434"


def test_local_provider_detect_server_llama_cpp() -> None:
    """Test llama.cpp server detection."""
    config = LLMConfig(provider="local", local_model="llama3")

    def _which_side_effect(x: str) -> str | None:
        return "/usr/bin/llama-server" if x == "llama-server" else None

    with (
        patch("assistant.llm.local.shutil.which", side_effect=_which_side_effect),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        provider = LocalLLMProvider(config)
        assert provider._base_url == "http://localhost:8080"


def test_local_provider_detect_server_lm_studio() -> None:
    """Test LM Studio server detection."""
    config = LLMConfig(provider="local", local_model="llama3")

    def _which_side_effect(x: str) -> str | None:
        return "/usr/bin/lms" if x == "lms" else None

    with (
        patch("assistant.llm.local.shutil.which", side_effect=_which_side_effect),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        provider = LocalLLMProvider(config)
        assert provider._base_url == "http://localhost:1234/v1"


def test_local_provider_no_server() -> None:
    """Test error when no local server found."""
    config = LLMConfig(provider="local", local_model="llama3")

    with patch("assistant.llm.local.shutil.which", return_value=None):
        try:
            LocalLLMProvider(config)
            raise AssertionError("Should have raised LLMProviderNotAvailableError")
        except LLMProviderNotAvailableError as e:
            assert "No local LLM server found" in str(e)


def test_local_provider_complete() -> None:
    """Test local provider completion."""
    config = LLMConfig(provider="local", local_model="llama3")

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch("assistant.llm.local.requests.Session.post") as mock_post,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"response": "Hello world"}
        mock_post.return_value = mock_post_resp

        provider = LocalLLMProvider(config)
        result = provider.complete("Hello")
        assert result.text == "Hello world"


def test_local_provider_stream_complete() -> None:
    """Test local provider streaming completion."""
    config = LLMConfig(provider="local", local_model="llama3")

    with (
        patch("assistant.llm.local.shutil.which", return_value="/usr/bin/ollama"),
        patch("assistant.llm.local.requests.get") as mock_get,
        patch("assistant.llm.local.requests.Session.post") as mock_post,
        patch.object(LocalLLMProvider, "start"),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        # Streaming response
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            b'{"response": "Hello"}',
            b'{"response": " world"}',
        ]
        mock_post.return_value = mock_post_resp

        provider = LocalLLMProvider(config)
        chunks = list(provider.stream_complete("Hello"))
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"


def test_remote_provider_complete_openai() -> None:
    """Test remote provider OpenAI completion."""
    config = LLMConfig(provider="openai", remote_model="gpt-4", api_key="test-key")

    with (
        patch("assistant.llm.remote.requests.Session.post") as mock_post,
        patch.object(RemoteLLMProvider, "start"),
    ):
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_post.return_value = mock_post_resp

        provider = RemoteLLMProvider(config)
        result = provider.complete("Hello")
        assert result.text == "Hello world"


def test_remote_provider_complete_anthropic() -> None:
    """Test remote provider Anthropic completion."""
    config = LLMConfig(provider="anthropic", remote_model="claude-3", api_key="test-key")

    with (
        patch("assistant.llm.remote.requests.Session.post") as mock_post,
        patch.object(RemoteLLMProvider, "start"),
    ):
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "content": [{"text": "Hello world"}]
        }
        mock_post.return_value = mock_post_resp

        provider = RemoteLLMProvider(config)
        result = provider.complete("Hello")
        assert result.text == "Hello world"