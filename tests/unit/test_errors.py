def test_error_hierarchy() -> None:
    from assistant.errors import (
        AgentError,
        AssistantError,
        AudioError,
        ConfigError,
        InitializationError,
        LLMError,
        NotFoundError,
        PermissionError,
        STTError,
        UIError,
        ValidationError,
        WakeError,
    )

    # All should inherit from AssistantError
    for err_cls in [
        ConfigError,
        AudioError,
        WakeError,
        STTError,
        LLMError,
        AgentError,
        UIError,
        PermissionError,
        ValidationError,
        NotFoundError,
        InitializationError,
    ]:
        assert issubclass(err_cls, AssistantError)

    # Base error
    err = AssistantError("base error")
    assert str(err) == "base error"


def test_error_chaining() -> None:
    from assistant.errors import AudioError, ConfigError

    try:
        try:
            raise ConfigError("config failed")
        except ConfigError as e:
            raise AudioError("audio failed") from e
    except AudioError as e:
        # Verify chaining works
        assert e.__cause__ is not None
        assert isinstance(e.__cause__, ConfigError)
        assert str(e.__cause__) == "config failed"


def test_error_context() -> None:
    from assistant.errors import ValidationError

    err = ValidationError("invalid value", field="sample_rate", value=100)
    assert err.field == "sample_rate"
    assert err.value == 100
    assert "sample_rate" in str(err)
    assert "100" in str(err)


def test_permission_error() -> None:
    from assistant.errors import PermissionError

    err = PermissionError("permission denied", required="shell")
    assert err.required == "shell"
    assert "shell" in str(err)


def test_not_found_error() -> None:
    from assistant.errors import NotFoundError

    err = NotFoundError("device not found", resource="audio_input")
    assert err.resource == "audio_input"
    assert "audio_input" in str(err)


def test_initialization_error() -> None:
    from assistant.errors import InitializationError

    err = InitializationError("failed to initialize", component="audio")
    assert err.component == "audio"
    assert "audio" in str(err)


def test_audio_error_with_device() -> None:
    from assistant.errors import AudioError

    err = AudioError("device error", device="microphone")
    assert err.device == "microphone"
    assert "microphone" in str(err)


def test_wake_error_with_model() -> None:
    from assistant.errors import WakeError

    err = WakeError("model load failed", model="hey_assistant")
    assert err.model == "hey_assistant"
    assert "hey_assistant" in str(err)


def test_stt_error_with_provider() -> None:
    from assistant.errors import STTError

    err = STTError("transcription failed", provider="whisper-api")
    assert err.provider == "whisper-api"
    assert "whisper-api" in str(err)


def test_llm_error_with_model() -> None:
    from assistant.errors import LLMError

    err = LLMError("generation failed", model="gpt-4")
    assert err.model == "gpt-4"
    assert "gpt-4" in str(err)


def test_agent_error_with_task() -> None:
    from assistant.errors import AgentError

    err = AgentError("task failed", task="open_chrome")
    assert err.task == "open_chrome"
    assert "open_chrome" in str(err)


def test_ui_error_with_state() -> None:
    from assistant.errors import UIError

    err = UIError("render failed", state="listening")
    assert err.state == "listening"
    assert "listening" in str(err)
