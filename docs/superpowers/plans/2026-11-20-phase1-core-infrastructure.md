# Phase 1: Core Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement core infrastructure: config system, logging, error hierarchy, CLI entry point, and main app scaffold.

**Architecture:** 
- Config: TOML-based with security levels, wake word, STT/LLM providers, first-run API scanner
- Logging: Structured logging with configurable levels and outputs
- Errors: Hierarchical exception classes for different error domains
- CLI: Click-based command line interface
- App: Main application class orchestrating all components

**Tech Stack:** Python 3.12, Click, TOML, logging, pathlib

**Spec:** `docs/specification.md` (sections 5, 6), `docs/implementation-plan.md` (Phase 1), `QUESTIONS.md` (Q12, Q13, Q14, Q17, Q23)

## Global Constraints

- Python 3.12 only (pyproject.toml)
- macOS only
- 100% test coverage target (pytest)
- Apache-2.0 license
- TOML-based config with security levels
- First-run API scanner for installed providers

## Review Focus

1. **Config validation** — Ensure all required fields have defaults, validate security levels, wake words
2. **Logging configuration** — Multiple handlers (file, stdout), proper formatting, level control
3. **Error hierarchy** — Proper exception chaining, context preservation, user-friendly messages
4. **CLI usability** — Clear help, subcommands, consistent naming, proper exit codes
5. **App lifecycle** — Startup/shutdown ordering, resource cleanup, signal handling

---

### Task 1: Config System

**Files:**
- Create: `src/assistant/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: None (foundational)
- Produces: `Config` class with:
  - `load(path: Path | None = None) -> Config` — load from TOML file
  - `save(path: Path) -> None` — save to TOML file
  - `get_default() -> Config` — factory for defaults
  - Properties for all config sections: `security`, `wake`, `stt`, `llm`, `audio`, `ui`, `permissions`

- [ ] **Step 1: Write failing test for Config**

```python
# tests/unit/test_config.py
import pytest
from pathlib import Path
import tempfile


def test_config_defaults() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    assert cfg.security.level == "standard"
    assert cfg.wake.word == "hey assistant"
    assert cfg.wake.enabled is True
    assert cfg.stt.provider == "cloud"
    assert cfg.llm.provider == "local"
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.channels == 1


def test_config_load_save() -> None:
    from assistant.config import Config

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        path = Path(f.name)

    try:
        cfg = Config.get_default()
        cfg.security.level = "strict"
        cfg.wake.word = "computer"
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.security.level == "strict"
        assert loaded.wake.word == "computer"
    finally:
        path.unlink(missing_ok=True)


def test_config_validation() -> None:
    from assistant.config import Config

    # Invalid security level
    cfg = Config.get_default()
    cfg.security.level = "invalid"
    with pytest.raises(ValueError):
        cfg.validate()

    # Invalid sample rate
    cfg = Config.get_default()
    cfg.audio.sample_rate = 100
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_first_run_api_scan() -> None:
    from assistant.config import Config

    cfg = Config.get_default()
    # Should not crash
    detected = cfg.scan_installed_apis()
    assert isinstance(detected, dict)
    assert "llm" in detected
    assert "stt" in detected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_config_defaults -v`
Expected: FAIL

- [ ] **Step 3: Implement Config system (`config.py`)**

Use dataclasses for config sections, TOML for serialization, pathlib for paths. Include security levels: "standard", "strict", "paranoid". Wake word config with enabled/disabled, custom word. STT/LLM provider selection with fallback chains.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/assistant/config.py tests/unit/test_config.py
git commit -m "feat: add config system with TOML support"
```

---

### Task 2: Logging System

**Files:**
- Create: `src/assistant/logging.py`
- Test: `tests/unit/test_logging.py`

**Interfaces:**
- Consumes: `Config` (for log level, format, output)
- Produces: `setup_logging(config: Config) -> None`, `get_logger(name: str) -> logging.Logger`

- [ ] **Step 1: Write failing test for Logging**

```python
# tests/unit/test_logging.py
import pytest
import logging
from io import StringIO


def test_setup_logging() -> None:
    from assistant.logging import setup_logging, get_logger
    from assistant.config import Config

    cfg = Config.get_default()
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.info("test message")
    output = stream.getvalue()
    assert "test message" in output


def test_log_levels() -> None:
    from assistant.logging import setup_logging, get_logger
    from assistant.config import Config

    cfg = Config.get_default()
    cfg.logging.level = "DEBUG"
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    output = stream.getvalue()
    assert "debug message" in output
    assert "info message" in output
    assert "warning message" in output
    assert "error message" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_logging.py::test_setup_logging -v`
Expected: FAIL

- [ ] **Step 3: Implement Logging (`logging.py`)**

Configure root logger with handlers (stdout, optional file), formatters, level from config. Provide `get_logger` helper.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_logging.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/assistant/logging.py tests/unit/test_logging.py
git commit -m "feat: add logging system"
```

---

### Task 3: Error Hierarchy

**Files:**
- Create: `src/assistant/errors.py`
- Test: `tests/unit/test_errors.py`

**Interfaces:**
- Consumes: None (foundational)
- Produces: Exception classes:
  - `AssistantError` (base)
  - `ConfigError`, `AudioError`, `WakeError`, `STTError`, `LLMError`, `AgentError`, `UIError`, `PermissionError`
  - `ValidationError`, `NotFoundError`, `InitializationError`

- [ ] **Step 1: Write failing test for Errors**

```python
# tests/unit/test_errors.py
import pytest


def test_error_hierarchy() -> None:
    from assistant.errors import (
        AssistantError,
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
        raise ConfigError("config failed")
    except ConfigError as e:
        raise AudioError("audio failed") from e

    # Verify chaining works
    assert True
```


def test_error_context() -> None:
    from assistant.errors import ValidationError

    err = ValidationError("invalid value", field="sample_rate", value=100)
    assert err.field == "sample_rate"
    assert err.value == 100
    assert "sample_rate" in str(err)
    assert "100" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_errors.py::test_error_hierarchy -v`
Expected: FAIL

- [ ] **Step 3: Implement Errors (`errors.py`)**

Define exception hierarchy with proper `__init__`, `__str__`, context attributes. Use `__notes__` for additional context (Python 3.11+).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_errors.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/assistant/errors.py tests/unit/test_errors.py
git commit -m "feat: add error hierarchy"
```

---

### Task 4: CLI Entry Point

**Files:**
- Create: `src/assistant/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `Config`, `AudioCapture`, `VoiceActivityDetector`
- Produces: Click CLI with commands: `run`, `config`, `doctor`, `version`

- [ ] **Step 1: Write failing test for CLI**

```python
# tests/unit/test_cli.py
import pytest
from click.testing import CliRunner


def test_cli_version() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_run_help() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower()


def test_cli_config_show() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli.py::test_cli_version -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI (`cli.py`)**

Use Click for command parsing. Commands: `run` (start assistant), `config` (show/edit config), `doctor` (health check), `version`. Global options: `--config`, `--verbose`, `--log-file`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/assistant/cli.py tests/unit/test_cli.py
git commit -m "feat: add CLI entry point"
```

---

### Task 5: Main App Scaffold

**Files:**
- Create: `src/assistant/app.py`
- Create: `src/assistant/__main__.py`
- Test: `tests/unit/test_app.py`

**Interfaces:**
- Consumes: All previous components
- Produces: `Application` class with `run()`, `shutdown()`, `health_check()`

- [ ] **Step 1: Write failing test for App**

```python
# tests/unit/test_app.py
import pytest
from unittest.mock import Mock, patch


def test_app_initialization() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)
    assert app.config == cfg
    assert app.is_running is False


def test_app_start_stop() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)

    with patch.object(app, "_init_audio") as mock_audio:
        with patch.object(app, "_init_wake") as mock_wake:
            with patch.object(app, "_init_ui") as mock_ui:
                app.start()
                assert app.is_running is True
                mock_audio.assert_called_once()
                mock_wake.assert_called_once()
                mock_ui.assert_called_once()

                app.stop()
                assert app.is_running is False


def test_app_health_check() -> None:
    from assistant.app import Application
    from assistant.config import Config

    cfg = Config.get_default()
    app = Application(cfg)
    health = app.health_check()
    assert isinstance(health, dict)
    assert "config" in health
    assert "audio" in health
    assert "wake" in health
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_app.py::test_app_initialization -v`
Expected: FAIL

- [ ] **Step 3: Implement App (`app.py`, `__main__.py`)**

`Application` class orchestrates all components. `__main__.py` calls CLI entry point. Handle signals (SIGINT, SIGTERM) for graceful shutdown.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_app.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/assistant/app.py src/assistant/__main__.py tests/unit/test_app.py
git commit -m "feat: add main app scaffold"
```

---

### Task 6: Update Implementation Plan

**Files:**
- Modify: `docs/implementation-plan.md`

- [ ] **Step 1: Mark Phase 1 tasks as complete**

Update checkboxes in `docs/implementation-plan.md` for Phase 1 items.

- [ ] **Step 2: Commit**

```bash
git add docs/implementation-plan.md
git commit -m "docs: mark Phase 1 complete"
```