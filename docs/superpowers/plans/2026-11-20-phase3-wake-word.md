# Phase 3: Wake Word Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement wake word detection using openWakeWord with configurable wake word, model loading from local directory, and fallback activation methods (double-click, triple-Shift).

**Architecture:**
- Wrapper around openWakeWord `Model` class in `src/assistant/wake/openwakeword_engine.py`
- Model loading from `src/assistant/wake/models/` with auto-discovery of `.onnx` files
- Integration with existing config system (`WakeConfig`) for wake word, thresholds, and fallback settings
- Provides callback-based API for async detection from audio frames

**Tech Stack:** openWakeWord, onnxruntime, numpy, existing audio pipeline (AudioRingBuffer, VoiceActivityDetector)

**Spec:** `docs/specification.md` (sections 2, 3), `docs/implementation-plan.md` (Phase 3), `QUESTIONS.md` (Q14, Q17, Q23)

## Global Constraints

- Python 3.12 only (pyproject.toml)
- macOS only
- 100% test coverage target (pytest)
- Apache-2.0 license
- Wake word detection runs on audio callback thread; must not block
- Model files (`.onnx`) not committed to git; loaded from `src/assistant/wake/models/` at runtime
- Fallback: double-click and triple-Shift hotkeys (handled by UI layer, but wake engine exposes callback)

## Review Focus

1. **Model loading failures** — Missing model directory, no `.onnx` files found, corrupted model files; must fall back gracefully or raise clear `WakeError`
2. **Audio format mismatch** — openWakeWord expects 16kHz 16-bit PCM; verify conversion from audio pipeline frames (30ms frames = 960 bytes)
3. **Threshold tuning** — Default threshold 0.5; expose configurable per-model threshold via config; test false-accept/false-reject behavior
4. **Resource cleanup** — Model holds ONNX sessions; ensure proper shutdown on app stop; test repeated init/fini cycles
5. **Callback thread safety** — Detection callback runs on audio thread; must not allocate or lock; use lock-free queue or atomic flag for wake event

---

### Task 1: Add Wake Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: None
- Produces: `wake` optional dependency group with `openwakeword>=0.6`, `onnxruntime>=1.18`

- [ ] **Step 1: Add wake optional dependencies**

```toml
# In pyproject.toml under [project.optional-dependencies]
wake = [
    "openwakeword>=0.6",
    "onnxruntime>=1.18",
]
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add wake optional dependencies"
```

---

### Task 2: Create Wake Engine Wrapper

**Files:**
- Create: `src/assistant/wake/openwakeword_engine.py`
- Create: `src/assistant/wake/__init__.py`
- Test: `tests/unit/test_openwakeword_engine.py`

**Interfaces:**
- Consumes: `WakeConfig` from `assistant.config`, `AudioRingBuffer` from `assistant.audio.ringbuffer`
- Produces: `WakeWordEngine` class:
  - `__init__(config: WakeConfig, model_dir: Path | None = None)`
  - `start() -> None` — initialize model, begin detection
  - `stop() -> None` — cleanup model resources
  - `process_frame(audio_frame: bytes) -> bool` — process one 30ms frame, return True if wake word detected
  - `set_callback(callback: Callable[[str], None]) -> None` — set wake word detected callback (model name)
  - `is_running() -> bool`

- [ ] **Step 1: Write failing test for WakeWordEngine basic initialization**

```python
# tests/unit/test_openwakeword_engine.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


def test_wake_engine_init():
    from assistant.wake.openwakeword_engine import WakeWordEngine
    from assistant.config import WakeConfig

    config = WakeConfig(word="hey assistant", enabled=True)
    with patch("assistant.wake.openwakeword_engine.Model") as mock_model:
        engine = WakeWordEngine(config, model_dir=Path("/fake/models"))
        assert engine.config == config
        assert engine.is_running() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_openwakeword_engine.py::test_wake_engine_init -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement WakeWordEngine (`openwakeword_engine.py`)**

Key implementation details:
- Load model from `model_dir` (default: `src/assistant/wake/models/`) — find first `.onnx` file
- Use openWakeWord `Model` with `inference_framework="onnx"` (more portable)
- Configure `vad_threshold=0.5` (from openWakeWord docs)
- Frame size: openWakeWord processes 80ms (1280 samples); our audio pipeline gives 30ms (480 samples)
  - Buffer frames until 1280 samples accumulated, then call `model.predict()`
- On detection (score > threshold), call registered callback with model name
- Handle missing model directory gracefully: raise `WakeError` with clear message

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_openwakeword_engine.py::test_wake_engine_init -v`
Expected: PASS

- [ ] **Step 5: Write test for frame processing and detection**

```python
def test_wake_engine_process_frame():
    from assistant.wake.openwakeword_engine import WakeWordEngine
    from assistant.config import WakeConfig
    import numpy as np

    config = WakeConfig(word="hey assistant", enabled=True)
    callback = Mock()

    with patch("assistant.wake.openwakeword_engine.Model") as mock_model_class:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_assistant": 0.0}  # No detection
        mock_model_class.return_value = mock_model

        engine = WakeWordEngine(config, model_dir=Path("/fake/models"))
        engine.set_callback(callback)
        engine.start()

        # 30ms frame at 16kHz 16-bit mono = 960 bytes
        frame = bytes(960)
        result = engine.process_frame(frame)

        assert result is False
        assert callback.call_count == 0

        # Simulate detection
        mock_model.predict.return_value = {"hey_assistant": 0.8}
        result = engine.process_frame(frame)

        assert result is True
        callback.assert_called_once_with("hey_assistant")

        engine.stop()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_openwakeword_engine.py::test_wake_engine_process_frame -v`
Expected: PASS

- [ ] **Step 7: Write test for model loading and error handling**

```python
def test_wake_engine_model_loading_error():
    from assistant.wake.openwakeword_engine import WakeWordEngine
    from assistant.config import WakeConfig
    from assistant.errors import WakeError

    config = WakeConfig(word="hey assistant", enabled=True)

    # Non-existent model directory
    try:
        engine = WakeWordEngine(config, model_dir=Path("/nonexistent"))
        engine.start()
        raise AssertionError("Should have raised WakeError")
    except WakeError as e:
        assert "model" in str(e).lower()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/unit/test_openwakeword_engine.py::test_wake_engine_model_loading_error -v`
Expected: PASS

- [ ] **Step 9: Write test for config integration**

```python
def test_wake_engine_config_integration():
    from assistant.wake.openwakeword_engine import WakeWordEngine
    from assistant.config import WakeConfig, Config
    from pathlib import Path

    # Test with full Config
    cfg = Config.get_default()
    cfg.wake.word = "computer"
    cfg.wake.enabled = True

    with patch("assistant.wake.openwakeword_engine.Model") as mock_model:
        engine = WakeWordEngine(cfg.wake, model_dir=Path("/fake/models"))
        assert engine.config.word == "computer"
        assert engine.config.enabled is True
```

- [ ] **Step 10: Run all wake engine tests**

Run: `pytest tests/unit/test_openwakeword_engine.py -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add src/assistant/wake/openwakeword_engine.py src/assistant/wake/__init__.py tests/unit/test_openwakeword_engine.py
git commit -m "feat: add openWakeWord engine wrapper"
```

---

### Task 3: Export Wake Module

**Files:**
- Modify: `src/assistant/wake/__init__.py`

**Interfaces:**
- Consumes: `WakeWordEngine` from `assistant.wake.openwakeword_engine`
- Produces: Public exports: `WakeWordEngine`, `WakeError` (re-exported from errors)

- [ ] **Step 1: Implement `__init__.py`**

```python
from assistant.wake.openwakeword_engine import WakeWordEngine
from assistant.errors import WakeError

__all__ = ["WakeWordEngine", "WakeError"]
```

- [ ] **Step 2: Test import**

Run: `python -c "from assistant.wake import WakeWordEngine, WakeError; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/assistant/wake/__init__.py
git commit -m "feat: export wake module"
```

---

### Task 4: Integration Test with Audio Pipeline

**Files:**
- Test: `tests/integration/test_wake_pipeline.py`

**Interfaces:**
- Consumes: `WakeWordEngine`, `AudioCapture`, `AudioRingBuffer`, `VoiceActivityDetector`
- Produces: End-to-end test verifying wake word detection from audio capture

- [ ] **Step 1: Write integration test (mocked)**

```python
# tests/integration/test_wake_pipeline.py
import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.integration
def test_wake_pipeline_flow():
    from assistant.wake.openwakeword_engine import WakeWordEngine
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector
    from assistant.config import WakeConfig, AudioConfig

    wake_config = WakeConfig(word="hey assistant", enabled=True)
    audio_config = AudioConfig(sample_rate=16000, channels=1, frame_duration_ms=30)

    rb = AudioRingBuffer(capacity_frames=16000 * 2, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    wake_detected = {"detected": False, "model": None}

    def on_wake(model_name: str):
        wake_detected["detected"] = True
        wake_detected["model"] = model_name

    with patch("assistant.wake.openwakeword_engine.Model") as mock_model_class:
        mock_model = MagicMock()
        # First call: no detection, second call: detection
        mock_model.predict.side_effect = [
            {"hey_assistant": 0.0},
            {"hey_assistant": 0.8},
        ]
        mock_model_class.return_value = mock_model

        engine = WakeWordEngine(wake_config, model_dir=None)  # Will use default
        engine.set_callback(on_wake)
        engine.start()

        # Mock capture to feed frames
        with patch.object(capture, "start"):
            with patch.object(capture, "stop"):
                capture.start()

                # Simulate audio frames being written to ring buffer
                # (In real test, this would come from actual audio callback)
                frame = bytes(960)  # 30ms silence
                for _ in range(10):
                    rb.write(frame)

                # Process frames through wake engine
                # (In real implementation, this runs in a loop)
                for _ in range(2):
                    data = rb.read(960)
                    if len(data) == 960:
                        engine.process_frame(data)

                capture.stop()

        engine.stop()

    assert wake_detected["detected"] is True
    assert wake_detected["model"] == "hey_assistant"
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/test_wake_pipeline.py -v -m integration`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_wake_pipeline.py
git commit -m "test: add wake word pipeline integration test"
```

---

### Task 5: Update Implementation Plan Checklist

**Files:**
- Modify: `docs/implementation-plan.md`

- [ ] **Step 1: Mark Phase 3 tasks as complete**

Update checkboxes in `docs/implementation-plan.md` for Phase 3 items:
- [x] openWakeWord engine integration (`src/assistant/wake/openwakeword_engine.py`)
- [x] Model loading from `src/assistant/wake/models/`
- [x] Configurable wake word with fallback to double-click / triple-Shift

- [ ] **Step 2: Commit**

```bash
git add docs/implementation-plan.md
git commit -m "docs: mark Phase 3 complete"
```

---

### Task 6: Full Test Suite & Coverage Check

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/unit/test_openwakeword_engine.py tests/unit/test_config.py tests/unit/test_vad.py tests/unit/test_capture.py tests/unit/test_ringbuffer.py -v`
Expected: All PASS

- [ ] **Step 2: Check coverage for wake module**

Run: `pytest --cov=src/assistant/wake --cov-report=term-missing`
Expected: 100% coverage

- [ ] **Step 3: Check overall audio + wake coverage**

Run: `pytest --cov=src/assistant/audio --cov=src/assistant/wake --cov-report=term-missing`
Expected: 100% coverage

- [ ] **Step 4: Run linter and type checker**

Run: `ruff check src/assistant/wake tests/unit/test_openwakeword_engine.py`
Run: `mypy src/assistant/wake`
Expected: No errors

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: lint and type fixes for wake module"
```
