# Phase 2: Audio Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete audio pipeline: C ring buffer with Python C extension, VAD integration, audio capture via Core Audio, and Python wrappers.

**Architecture:** 
- Native ring buffer in C (`native/ringbuffer/`) with lock-free, thread-safe design for real-time audio callbacks
- Python C extension module (`ringbuffer`) exposing the C API
- Python wrapper (`src/assistant/audio/ringbuffer.py`) providing idiomatic Python API
- VAD using webrtcvad (`src/assistant/audio/vad.py`)
- Core Audio capture (`src/assistant/audio/capture.py`) using pyobjc to interface with AudioUnit

**Tech Stack:** C99, Python C API, pyobjc-framework-AudioToolbox, pyobjc-framework-CoreAudio, webrtcvad, numpy

**Spec:** `docs/specification.md` (sections 5, 6), `docs/implementation-plan.md` (Phase 2), `QUESTIONS.md` (Q14, Q23)

## Global Constraints

- Python 3.12 only (pyproject.toml)
- macOS only (Core Audio, AudioUnit)
- 100% test coverage target (pytest)
- Performance-critical audio path in C, no locks in audio callback
- Ring buffer must be thread-safe for single-producer (audio callback) / single-consumer (Python reader)
- Apache-2.0 license

## Review Focus

1. **Audio callback thread safety** — C ring buffer must never block or allocate in the audio callback; verify with stress test
2. **Buffer overrun/underrun handling** — What happens when producer is faster/slower than consumer; test edge cases
3. **VAD false positives/negatives** — Tune aggressiveness for speech vs. noise; test with various audio inputs
4. **Core Audio device changes** — Handle device plug/unplug, sample rate changes; test hot-plug
5. **Memory leaks in C extension** — Reference counting, proper cleanup on module unload; test with repeated init/fini

---

### Task 1: C Ring Buffer Implementation

**Files:**
- Create: `native/ringbuffer/ringbuffer.h`
- Create: `native/ringbuffer/ringbuffer.c`
- Create: `native/ringbuffer/module.c`
- Create: `native/ringbuffer/setup.py` (or add to pyproject.toml)
- Test: `tests/unit/test_ringbuffer.py`

**Interfaces:**
- Consumes: None (foundational)
- Produces: Python module `ringbuffer` with class `RingBuffer`:
  - `RingBuffer(capacity: int)` — constructor
  - `write(data: bytes) -> int` — returns bytes written
  - `read(size: int) -> bytes` — returns data read
  - `readable() -> int` — bytes available to read
  - `writable() -> int` — space available to write
  - `capacity() -> int` — total capacity
  - `clear()` — reset buffer

- [ ] **Step 1: Write failing test for RingBuffer basic operations**

```python
# tests/unit/test_ringbuffer.py
import pytest
import ringbuffer


def test_ringbuffer_basic():
    rb = ringbuffer.RingBuffer(1024)
    assert rb.capacity() == 1024
    assert rb.readable() == 0
    assert rb.writable() == 1024

    written = rb.write(b"hello")
    assert written == 5
    assert rb.readable() == 5
    assert rb.writable() == 1019

    data = rb.read(5)
    assert data == b"hello"
    assert rb.readable() == 0
    assert rb.writable() == 1024


def test_ringbuffer_wraparound():
    rb = ringbuffer.RingBuffer(16)
    rb.write(b"01234567")  # 8 bytes
    rb.read(4)  # read 4, 4 remaining
    rb.write(b"abcd")  # write 4, buffer full (8)
    assert rb.readable() == 8
    data = rb.read(8)
    assert data == b"4567abcd"


def test_ringbuffer_clear():
    rb = ringbuffer.RingBuffer(32)
    rb.write(b"test")
    rb.clear()
    assert rb.readable() == 0
    assert rb.writable() == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ringbuffer.py::test_ringbuffer_basic -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement C ring buffer header (`ringbuffer.h`)**

```c
#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t head;  // write position
    size_t tail;  // read position
    // No locks needed for single-producer/single-consumer
    // head only written by producer, tail only by consumer
} ringbuffer_t;

int ringbuffer_init(ringbuffer_t *rb, size_t capacity);
void ringbuffer_free(ringbuffer_t *rb);
size_t ringbuffer_write(ringbuffer_t *rb, const uint8_t *data, size_t len);
size_t ringbuffer_read(ringbuffer_t *rb, uint8_t *data, size_t len);
size_t ringbuffer_readable(const ringbuffer_t *rb);
size_t ringbuffer_writable(const ringbuffer_t *rb);
size_t ringbuffer_capacity(const ringbuffer_t *rb);
void ringbuffer_clear(ringbuffer_t *rb);

#endif
```

- [ ] **Step 4: Implement C ring buffer (`ringbuffer.c`)**

Lock-free circular buffer using atomic operations for head/tail (or plain size_t with memory barriers since single-producer/single-consumer on same CPU cache line is safe on x86/ARM).

- [ ] **Step 5: Implement Python C extension (`module.c`)**

Expose `RingBuffer` type with methods: `__init__`, `write`, `read`, `readable`, `writable`, `capacity`, `clear`.

- [ ] **Step 6: Add build configuration**

Add `native/ringbuffer` as a C extension in `pyproject.toml` under `[tool.setuptools.ext_modules]` or create `setup.py`.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/unit/test_ringbuffer.py::test_ringbuffer_basic -v`
Expected: PASS

- [ ] **Step 8: Run all ringbuffer tests**

Run: `pytest tests/unit/test_ringbuffer.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add native/ringbuffer/ tests/unit/test_ringbuffer.py pyproject.toml
git commit -m "feat: add C ring buffer with Python C extension"
```

---

### Task 2: Python Ring Buffer Wrapper

**Files:**
- Create: `src/assistant/audio/ringbuffer.py`
- Create: `src/assistant/audio/_ringbuffer_py.py` (fallback pure Python implementation)
- Modify: `src/assistant/audio/__init__.py`
- Test: `tests/unit/test_ringbuffer.py` (extend)

**Interfaces:**
- Consumes: `ringbuffer` C extension (Task 1)
- Produces: `AudioRingBuffer` class in `src.assistant.audio.ringbuffer`:
  - Same API as C extension but with numpy support
  - `write_frames(frames: np.ndarray) -> int` — write audio frames
  - `read_frames(num_frames: int) -> np.ndarray` — read audio frames
  - Context manager support for cleanup

- [ ] **Step 1: Write failing test for Python wrapper**

```python
# tests/unit/test_ringbuffer.py (additions)
import numpy as np


def test_audio_ringbuffer_numpy():
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=1024, num_channels=1, dtype=np.float32)
    assert rb.capacity_frames == 1024

    frames = np.zeros(100, dtype=np.float32)
    frames[:] = 0.5
    written = rb.write_frames(frames)
    assert written == 100

    read_frames = rb.read_frames(100)
    assert read_frames.shape == (100,)
    assert np.allclose(read_frames, 0.5)


def test_audio_ringbuffer_fallback():
    # Test that fallback works when C extension not available
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(1024)
    rb.write(b"test")
    assert rb.read(4) == b"test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ringbuffer.py::test_audio_ringbuffer_numpy -v`
Expected: FAIL

- [ ] **Step 3: Implement pure Python fallback (`_ringbuffer_py.py`)**

Simple list-based or bytearray-based ring buffer for environments without C extension.

- [ ] **Step 4: Implement Python wrapper (`ringbuffer.py`)**

Try importing C extension, fall back to pure Python. Add numpy-aware methods.

- [ ] **Step 5: Export from `audio/__init__.py`**

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_ringbuffer.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/assistant/audio/ringbuffer.py src/assistant/audio/_ringbuffer_py.py src/assistant/audio/__init__.py tests/unit/test_ringbuffer.py
git commit -m "feat: add Python ring buffer wrapper with numpy support"
```

---

### Task 3: VAD Integration

**Files:**
- Create: `src/assistant/audio/vad.py`
- Test: `tests/unit/test_vad.py`

**Interfaces:**
- Consumes: `webrtcvad` (optional dependency)
- Produces: `VoiceActivityDetector` class:
  - `__init__(aggressiveness: int = 3, sample_rate: int = 16000, frame_duration_ms: int = 30)`
  - `is_speech(audio_frame: bytes) -> bool` — process 10/20/30ms frame
  - `process_stream(audio_iter: Iterator[bytes]) -> Iterator[tuple[bytes, bool]]` — yield (frame, is_speech)

- [ ] **Step 1: Write failing test for VAD**

```python
# tests/unit/test_vad.py
import pytest


def test_vad_basic():
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)

    # Silence frame (all zeros)
    silence = bytes(960)  # 30ms at 16kHz mono 16-bit = 960 bytes
    assert vad.is_speech(silence) is False

    # Simple tone (not real speech, but tests API)
    # Real speech test would need actual audio data
    assert hasattr(vad, "process_stream")


def test_vad_frame_size():
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    # 30ms * 16000Hz * 2 bytes = 960 bytes
    assert vad.frame_size_bytes == 960

    vad2 = VoiceActivityDetector(sample_rate=8000, frame_duration_ms=20)
    # 20ms * 8000Hz * 2 bytes = 320 bytes
    assert vad2.frame_size_bytes == 320
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vad.py::test_vad_basic -v`
Expected: FAIL

- [ ] **Step 3: Implement VAD (`vad.py`)**

Wrap `webrtcvad.Vad` with proper frame size calculation and stream processing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vad.py -v`
Expected: PASS

- [ ] **Step 5: Add webrtcvad to optional dependencies in pyproject.toml**

- [ ] **Step 6: Commit**

```bash
git add src/assistant/audio/vad.py tests/unit/test_vad.py pyproject.toml
git commit -m "feat: add VAD integration with webrtcvad"
```

---

### Task 4: Audio Capture (Core Audio)

**Files:**
- Create: `src/assistant/audio/capture.py`
- Test: `tests/unit/test_capture.py` (new)

**Interfaces:**
- Consumes: `AudioRingBuffer` (Task 2), `VoiceActivityDetector` (Task 3), pyobjc-framework-CoreAudio, pyobjc-framework-AudioToolbox
- Produces: `AudioCapture` class:
  - `__init__(ring_buffer: AudioRingBuffer, vad: VoiceActivityDetector | None = None, device_id: int | None = None, sample_rate: int = 16000, channels: int = 1)`
  - `start() -> None` — begin capture
  - `stop() -> None` — stop capture
  - `is_running() -> bool`
  - Callback runs on Core Audio thread, writes to ring buffer (lock-free)

- [ ] **Step 1: Write failing test for AudioCapture**

```python
# tests/unit/test_capture.py
import pytest
from unittest.mock import Mock, patch


def test_audio_capture_init():
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector()

    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)
    assert capture.sample_rate == 16000
    assert capture.channels == 1
    assert capture.is_running() is False


def test_audio_capture_start_stop():
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    with patch("assistant.audio.capture.AudioUnit") as mock_au:
        capture = AudioCapture(ring_buffer=rb, sample_rate=16000, channels=1)
        capture.start()
        assert capture.is_running() is True
        capture.stop()
        assert capture.is_running() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_capture.py::test_audio_capture_init -v`
Expected: FAIL

- [ ] **Step 3: Implement AudioCapture (`capture.py`)**

Use pyobjc to:
- Find default input device or specified device
- Create AudioComponentDescription for kAudioUnitType_Output / kAudioUnitSubType_HALOutput
- Set up AURenderCallbackStruct for input callback
- In callback: convert AudioBufferList to bytes, write to ring buffer
- Handle sample rate conversion if needed (Core Audio provides 44.1/48kHz, we want 16kHz)

- [ ] **Step 4: Add pyobjc-framework-CoreAudio and pyobjc-framework-AudioToolbox to optional dependencies**

- [ ] **Step 5: Run test to verify it passes** (may need integration test on real hardware)

Run: `pytest tests/unit/test_capture.py -v`
Expected: PASS (unit tests with mocks)

- [ ] **Step 6: Commit**

```bash
git add src/assistant/audio/capture.py tests/unit/test_capture.py pyproject.toml
git commit -m "feat: add Core Audio capture with AudioUnit"
```

---

### Task 5: Integration Test & Pipeline Smoke Test

**Files:**
- Test: `tests/integration/test_audio_pipeline.py` (new)

**Interfaces:**
- Consumes: All audio components (Tasks 1-4)
- Produces: End-to-end test verifying audio flows from capture → ring buffer → VAD

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_audio_pipeline.py
import pytest
import time
import numpy as np


@pytest.mark.integration
def test_audio_pipeline_flow():
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector

    rb = AudioRingBuffer(capacity_frames=16000 * 2, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    capture.start()
    time.sleep(0.5)  # Capture some audio
    capture.stop()

    # Should have some data in buffer
    assert rb.readable() > 0

    # Read and verify VAD can process
    data = rb.read(min(rb.readable(), vad.frame_size_bytes * 10))
    frames = np.frombuffer(data, dtype=np.int16)
    # Just verify no crash
    for i in range(0, len(frames), vad.frame_size_bytes // 2):
        chunk = frames[i : i + vad.frame_size_bytes // 2].tobytes()
        if len(chunk) == vad.frame_size_bytes:
            _ = vad.is_speech(chunk)
```

- [ ] **Step 2: Run integration test** (requires audio hardware)

Run: `pytest tests/integration/test_audio_pipeline.py -v -m integration`
Expected: PASS on macOS with microphone

- [ ] **Step 3: Run full test suite for audio module**

Run: `pytest tests/unit/test_ringbuffer.py tests/unit/test_vad.py tests/unit/test_capture.py -v`
Expected: All PASS

- [ ] **Step 4: Check coverage**

Run: `pytest --cov=src/assistant/audio --cov-report=term-missing`
Expected: 100% coverage

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_audio_pipeline.py
git commit -m "test: add audio pipeline integration test"
```

---

### Task 6: Update Implementation Plan Checklist

**Files:**
- Modify: `docs/implementation-plan.md`

- [ ] **Step 1: Mark Phase 2 tasks as complete**

Update checkboxes in `docs/implementation-plan.md` for Phase 2 items.

- [ ] **Step 2: Commit**

```bash
git add docs/implementation-plan.md
git commit -m "docs: mark Phase 2 complete"
```