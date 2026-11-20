# mypy: ignore-errors
"""Integration tests for the audio pipeline."""

import time

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_audio_pipeline_flow() -> None:
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


def test_audio_pipeline_with_vad() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector

    rb = AudioRingBuffer(capacity_frames=16000 * 2, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    capture.start()
    time.sleep(0.3)
    capture.stop()

    # Should have some data
    assert rb.readable() > 0

    # Test VAD on captured data
    data = rb.read(min(rb.readable(), vad.frame_size_bytes * 5))
    frames = np.frombuffer(data, dtype=np.int16)

    speech_frames = 0
    total_frames = 0
    for i in range(0, len(frames), vad.frame_size_bytes // 2):
        chunk = frames[i : i + vad.frame_size_bytes // 2].tobytes()
        if len(chunk) == vad.frame_size_bytes:
            total_frames += 1
            if vad.is_speech(chunk):
                speech_frames += 1

    # Just verify the pipeline works without errors
    assert total_frames > 0


def test_audio_pipeline_multiple_captures() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector

    rb = AudioRingBuffer(capacity_frames=16000 * 4, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)
    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)

    # Multiple start/stop cycles
    for _ in range(3):
        capture.start()
        time.sleep(0.1)
        capture.stop()
        assert rb.readable() > 0
        # Clear buffer for next iteration
        rb.clear()

    assert rb.readable() == 0
