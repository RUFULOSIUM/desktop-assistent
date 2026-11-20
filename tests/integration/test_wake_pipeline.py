"""Integration tests for wake word pipeline."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import-not-found]

# Create a mock sherpa_onnx module before any imports
mock_sherpa = MagicMock()
mock_kws = MagicMock()
mock_stream = MagicMock()

mock_kws.create_stream.return_value = mock_stream
# is_ready called per process_frame:
# 1st frame: False
# 2nd frame: True -> get_result="", then is_ready=False
# 3rd frame: True -> get_result="HEY_ASSISTANT" -> detected!
mock_kws.is_ready.side_effect = [False, True, False, True]
mock_kws.get_result.side_effect = ["", "HEY_ASSISTANT"]
mock_sherpa.KeywordSpotter.return_value = mock_kws
sys.modules["sherpa_onnx"] = mock_sherpa


def create_fake_model_dir() -> Path:
    """Create a temporary directory with fake sherpa-onnx model files."""
    tmpdir = tempfile.mkdtemp()
    model_dir = Path(tmpdir)

    # Create required model files
    tokens_content = (
        "<blk> 0\n<sos/eos> 1\nHH 2\nEY1 3\nAH0 4\nS 5\nIH1 6\nT 7\nN 8\n"
    )
    (model_dir / "tokens.txt").write_text(tokens_content)
    (model_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.onnx").write_bytes(b"fake encoder")
    (model_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx").write_bytes(b"fake decoder")
    (model_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx").write_bytes(b"fake joiner")
    (model_dir / "keywords.txt").write_text("HH EY1 AH0 S IH1 S T AH0 N T @HEY_ASSISTANT\n")

    return model_dir


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_wake_pipeline_flow() -> None:
    """Test end-to-end wake word detection from audio capture."""
    import numpy as np  # type: ignore[import-not-found]

    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector
    from assistant.config import WakeConfig
    from assistant.wake.sherpa_engine import SherpaWakeEngine

    wake_config = WakeConfig(word="HEY_ASSISTANT", enabled=True)

    rb = AudioRingBuffer(capacity_frames=16000 * 2, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    wake_detected: dict[str, bool | str | None] = {"detected": False, "model": None}

    def on_wake(model_name: str) -> None:
        wake_detected["detected"] = True
        wake_detected["model"] = model_name

    model_dir = create_fake_model_dir()
    try:
        engine = SherpaWakeEngine(wake_config, model_dir=model_dir)
        engine.set_callback(on_wake)
        engine.start()

        # Mock capture to feed frames
        with patch.object(capture, "start"), patch.object(capture, "stop"):
            capture.start()

            # Simulate audio frames being written to ring buffer
            frame = bytes(1920)  # 960 samples (60ms at 16kHz)
            for _ in range(10):
                rb.write(frame)

            # Process frames through wake engine (need 3 for mock sequence)
            for _ in range(3):
                data = rb.read(1920)
                if len(data) == 1920:
                    engine.process_frame(data)

            capture.stop()

        engine.stop()

        assert wake_detected["detected"] is True
        assert wake_detected["model"] == "HEY_ASSISTANT"
    finally:
        for f in model_dir.glob("*"):
            f.unlink()
        model_dir.rmdir()