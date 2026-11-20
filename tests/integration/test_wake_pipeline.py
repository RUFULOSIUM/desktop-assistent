"""Integration tests for wake word pipeline."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import-not-found]

# Create a mock openwakeword module before any imports
mock_openwakeword = MagicMock()
mock_model_class = MagicMock()
mock_openwakeword.model.Model = mock_model_class
sys.modules["openwakeword"] = mock_openwakeword
sys.modules["openwakeword.model"] = mock_openwakeword.model


def create_fake_model_dir() -> Path:
    """Create a temporary directory with a fake .onnx model file."""
    tmpdir = tempfile.mkdtemp()
    model_file = Path(tmpdir) / "hey_assistant.onnx"
    model_file.write_bytes(b"fake onnx model")
    return Path(tmpdir)


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_wake_pipeline_flow() -> None:
    """Test end-to-end wake word detection from audio capture."""
    import numpy as np  # type: ignore[import-not-found]

    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector
    from assistant.config import WakeConfig
    from assistant.wake.openwakeword_engine import WakeWordEngine

    wake_config = WakeConfig(word="hey assistant", enabled=True)

    rb = AudioRingBuffer(capacity_frames=16000 * 2, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    wake_detected: dict[str, bool | str | None] = {"detected": False, "model": None}

    def on_wake(model_name: str) -> None:
        wake_detected["detected"] = True
        wake_detected["model"] = model_name

    mock_model = MagicMock()
    # First call: no detection, second call: detection
    mock_model.predict.side_effect = [
        {"hey_assistant": 0.0},
        {"hey_assistant": 0.8},
    ]
    mock_model_class.return_value = mock_model

    model_dir = create_fake_model_dir()
    try:
        engine = WakeWordEngine(wake_config, model_dir=model_dir)
        engine.set_callback(on_wake)
        engine.start()

        # Mock capture to feed frames
        with patch.object(capture, "start"), patch.object(capture, "stop"):
            capture.start()

            # Simulate audio frames being written to ring buffer
            frame = bytes(960)  # 30ms silence
            for _ in range(10):
                rb.write(frame)

            # Process frames through wake engine
            for _ in range(2):
                data = rb.read(960)
                if len(data) == 960:
                    # Need 3 frames per 80ms processing window
                    engine.process_frame(data)
                    engine.process_frame(data)
                    engine.process_frame(data)

            capture.stop()

        engine.stop()

        assert wake_detected["detected"] is True
        assert wake_detected["model"] == "hey_assistant"
    finally:
        for f in model_dir.glob("*"):
            f.unlink()
        model_dir.rmdir()
