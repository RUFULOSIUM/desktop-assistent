"""Tests for pipeline module."""

from unittest.mock import MagicMock, patch

from assistant.config import Config
from assistant.pipeline import AssistantPipeline, PipelineResult, create_pipeline


def test_pipeline_result() -> None:
    """Test PipelineResult dataclass."""
    result = PipelineResult(success=True, response="hello", is_agent_task=False)
    assert result.success is True
    assert result.response == "hello"
    assert result.is_agent_task is False

    result2 = PipelineResult(success=False, error="failed", is_agent_task=True)
    assert result2.success is False
    assert result2.error == "failed"
    assert result2.is_agent_task is True


def test_create_pipeline() -> None:
    """Test factory function."""
    config = Config.get_default()
    pipeline = create_pipeline(config)
    assert isinstance(pipeline, AssistantPipeline)


def test_pipeline_init() -> None:
    """Test pipeline initialization."""
    config = Config.get_default()
    pipeline = AssistantPipeline(config)
    assert pipeline.config == config
    assert pipeline.is_running is False


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_initialize(
    mock_wake: MagicMock,
    _mock_dispatcher: MagicMock,
    _mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test pipeline initialization."""
    config = Config.get_default()

    # Setup mocks
    mock_stt_instance = MagicMock()
    mock_stt.return_value = mock_stt_instance

    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance

    mock_wake_instance = MagicMock()
    mock_wake.return_value = mock_wake_instance

    pipeline = AssistantPipeline(config)
    pipeline.initialize()

    assert pipeline.is_running is True
    mock_stt_instance.start.assert_called_once()
    mock_llm_instance.start.assert_called_once()
    mock_wake_instance.start.assert_called_once()


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_shutdown(
    mock_wake: MagicMock,
    _mock_dispatcher: MagicMock,
    _mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test pipeline shutdown."""
    config = Config.get_default()

    mock_stt_instance = MagicMock()
    mock_stt.return_value = mock_stt_instance
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance
    mock_wake_instance = MagicMock()
    mock_wake.return_value = mock_wake_instance

    pipeline = AssistantPipeline(config)
    pipeline.initialize()
    pipeline.shutdown()

    assert pipeline.is_running is False
    mock_stt_instance.stop.assert_called_once()
    mock_llm_instance.stop.assert_called_once()
    mock_wake_instance.stop.assert_called_once()


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_set_callbacks(
    mock_wake: MagicMock,
    mock_dispatcher: MagicMock,
    mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test setting UI callbacks."""
    config = Config.get_default()

    mock_stt.return_value = MagicMock()
    mock_llm.return_value = MagicMock()
    mock_classifier.return_value = MagicMock()
    mock_dispatcher.return_value = MagicMock()
    mock_wake.return_value = MagicMock()

    pipeline = AssistantPipeline(config)
    pipeline.initialize()

    wake_cb = MagicMock()
    listen_cb = MagicMock()
    think_cb = MagicMock()
    speak_cb = MagicMock()
    idle_cb = MagicMock()

    pipeline.set_callbacks(
        on_wake_detected=wake_cb,
        on_listening=listen_cb,
        on_thinking=think_cb,
        on_speaking=speak_cb,
        on_idle=idle_cb,
    )

    # Trigger wake callback
    pipeline._on_wake_word("test-model")
    wake_cb.assert_called_once()

    # Trigger speak callback
    pipeline._on_speaking("hello")
    speak_cb.assert_called_once_with("hello")


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_process_transcription_chat(
    mock_wake: MagicMock,
    mock_dispatcher: MagicMock,
    mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test processing transcription for chat path."""
    config = Config.get_default()

    mock_stt.return_value = MagicMock()
    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = MagicMock(text="Chat response")
    mock_llm.return_value = mock_llm_instance

    mock_classifier_instance = MagicMock()
    mock_classifier_instance.classify.return_value = False  # chat
    mock_classifier.return_value = mock_classifier_instance

    mock_dispatcher.return_value = MagicMock()
    mock_wake.return_value = MagicMock()

    pipeline = AssistantPipeline(config)
    pipeline.initialize()

    result = pipeline.process_transcription("What is the weather?")

    assert result.success is True
    assert result.response == "Chat response"
    assert result.is_agent_task is False
    mock_llm_instance.complete.assert_called_once()


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_process_transcription_agent(
    mock_wake: MagicMock,
    mock_dispatcher: MagicMock,
    mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test processing transcription for agent path."""
    config = Config.get_default()

    mock_stt.return_value = MagicMock()

    # Mock LLM to return JSON task specification
    mock_llm_instance = MagicMock()
    task_json = (
        '{"task": "Open Chrome and play a video", '
        '"permissions": ["shell", "apps"], "confirmation_required": false}'
    )
    mock_llm_instance.complete.return_value = MagicMock(text=task_json)
    mock_llm.return_value = mock_llm_instance

    mock_classifier_instance = MagicMock()
    mock_classifier_instance.classify.return_value = True  # agent
    mock_classifier.return_value = mock_classifier_instance

    mock_dispatcher_instance = MagicMock()
    mock_dispatcher_instance.dispatch.return_value = MagicMock(
        success=True, output="Agent did it", exit_code=0, error=None
    )
    mock_dispatcher.return_value = mock_dispatcher_instance

    mock_wake.return_value = MagicMock()

    pipeline = AssistantPipeline(config)
    pipeline.initialize()

    result = pipeline.process_transcription("Open Chrome and play a video")

    assert result.success is True
    assert result.response == "Agent did it"
    assert result.is_agent_task is True
    mock_dispatcher_instance.dispatch.assert_called_once()


@patch("assistant.pipeline.create_stt_provider")
@patch("assistant.pipeline.create_llm_provider")
@patch("assistant.pipeline.create_classifier")
@patch("assistant.pipeline.create_dispatcher")
@patch("assistant.pipeline.WakeWordEngine")
def test_pipeline_process_transcription_permission_denied(
    mock_wake: MagicMock,
    mock_dispatcher: MagicMock,
    mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_stt: MagicMock,
) -> None:
    """Test processing transcription with permission denied."""
    config = Config.get_default()
    config.permissions.allow_shell = False

    mock_stt.return_value = MagicMock()

    # Mock LLM to return JSON task specification requiring shell
    mock_llm_instance = MagicMock()
    task_json = (
        '{"task": "Run shell command", '
        '"permissions": ["shell"], "confirmation_required": false}'
    )
    mock_llm_instance.complete.return_value = MagicMock(text=task_json)
    mock_llm.return_value = mock_llm_instance

    mock_classifier_instance = MagicMock()
    mock_classifier_instance.classify.return_value = True  # agent
    mock_classifier.return_value = mock_classifier_instance

    mock_dispatcher_instance = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_instance

    mock_wake.return_value = MagicMock()

    pipeline = AssistantPipeline(config)
    pipeline.initialize()

    result = pipeline.process_transcription("Run shell command")

    assert result.success is False
    assert "No permissions granted" in result.error
    mock_dispatcher_instance.dispatch.assert_not_called()