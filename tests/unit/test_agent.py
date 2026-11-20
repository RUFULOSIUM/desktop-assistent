"""Tests for agent module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from assistant.agent import (
    AgentDispatcher,
    DispatchResult,
    create_dispatcher,
    get_preset,
    list_presets,
)
from assistant.agent.presets import AgentPreset


def test_dispatch_result() -> None:
    """Test DispatchResult dataclass."""
    result = DispatchResult(success=True, output="done", exit_code=0)
    assert result.success is True
    assert result.output == "done"
    assert result.exit_code == 0
    assert result.error is None

    result2 = DispatchResult(success=False, output="", error="failed", exit_code=1)
    assert result2.success is False
    assert result2.error == "failed"


def test_agent_dispatcher_init() -> None:
    """Test AgentDispatcher initialization."""
    dispatcher = AgentDispatcher(pi_binary="pi", working_dir=Path("/tmp"))
    assert dispatcher.pi_binary == "pi"
    assert dispatcher.working_dir == Path("/tmp")
    assert dispatcher.timeout == 300


def test_agent_dispatcher_find_pi() -> None:
    """Test finding pi binary."""
    dispatcher = AgentDispatcher(pi_binary="nonexistent")
    # Should not find it
    assert dispatcher._pi_path is None
    assert dispatcher.is_available() is False


def test_agent_dispatcher_is_available() -> None:
    """Test is_available when pi not found."""
    dispatcher = AgentDispatcher(pi_binary="nonexistent")
    assert dispatcher.is_available() is False


def test_agent_dispatcher_dispatch_not_found() -> None:
    """Test dispatch when pi not found."""
    dispatcher = AgentDispatcher(pi_binary="nonexistent")
    result = dispatcher.dispatch("test task")
    assert result.success is False
    assert result.exit_code == 127
    assert "not found" in result.error.lower()


def test_agent_dispatcher_dispatch_success() -> None:
    """Test successful dispatch."""
    dispatcher = AgentDispatcher(pi_binary="pi")
    dispatcher._pi_path = "/fake/pi"

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "Task done"}'
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = dispatcher.dispatch("test task", permissions=["shell", "files"])
        assert result.success is True
        assert result.output == "Task done"
        assert result.exit_code == 0


def test_agent_dispatcher_dispatch_failure() -> None:
    """Test failed dispatch."""
    dispatcher = AgentDispatcher(pi_binary="pi")
    dispatcher._pi_path = "/fake/pi"

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result

        result = dispatcher.dispatch("test task")
        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "Error occurred"


def test_agent_dispatcher_dispatch_timeout() -> None:
    """Test dispatch timeout."""
    import subprocess

    dispatcher = AgentDispatcher(pi_binary="pi", timeout=1)
    dispatcher._pi_path = "/fake/pi"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("pi", 1)

        result = dispatcher.dispatch("test task")
        assert result.success is False
        assert result.exit_code == 124
        assert "timed out" in result.error.lower()


def test_agent_dispatcher_dispatch_stream() -> None:
    """Test streaming dispatch."""
    dispatcher = AgentDispatcher(pi_binary="pi")
    dispatcher._pi_path = "/fake/pi"

    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        results = list(dispatcher.dispatch_stream("test task"))
        assert len(results) >= 2


def test_create_dispatcher() -> None:
    """Test factory function."""
    dispatcher = create_dispatcher()
    assert isinstance(dispatcher, AgentDispatcher)

    class MockConfig:
        pi_binary = "custom-pi"
        working_dir = "/custom/dir"
        timeout = 600

    dispatcher2 = create_dispatcher(MockConfig())
    assert dispatcher2.pi_binary == "custom-pi"
    assert dispatcher2.working_dir == Path("/custom/dir")
    assert dispatcher2.timeout == 600


def test_presets() -> None:
    """Test agent presets."""
    presets = list_presets()
    assert len(presets) > 0
    assert all(isinstance(p, AgentPreset) for p in presets)

    general = get_preset("general")
    assert general is not None
    assert general.name == "general"
    assert "shell" in general.permissions

    coding = get_preset("coding")
    assert coding is not None
    assert coding.confirmation_required is False

    unknown = get_preset("unknown")
    assert unknown is None


def test_agent_preset_dataclass() -> None:
    """Test AgentPreset dataclass."""
    preset = AgentPreset(
        name="test",
        description="Test preset",
        permissions=["shell"],
        confirmation_required=True,
    )
    assert preset.name == "test"
    assert preset.system_prompt_addendum == ""