from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore[import-not-found]


def test_audio_capture_init() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer
    from assistant.audio.vad import VoiceActivityDetector

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)
    vad = VoiceActivityDetector()

    capture = AudioCapture(ring_buffer=rb, vad=vad, sample_rate=16000, channels=1)
    assert capture.sample_rate == 16000
    assert capture.channels == 1
    assert capture.is_running() is False


def test_audio_capture_start_stop() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    # Mock the entire AudioUnit module import
    with patch.dict("sys.modules", {"AudioUnit": MagicMock(), "CoreAudio": MagicMock()}):
        from importlib import reload

        import assistant.audio.capture as capture_module

        reload(capture_module)

        capture = capture_module.AudioCapture(ring_buffer=rb, sample_rate=16000, channels=1)
        capture.start()
        assert capture.is_running() is True
        capture.stop()
        assert capture.is_running() is False


def test_audio_capture_default_params() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    capture = AudioCapture(ring_buffer=rb)
    assert capture.sample_rate == 16000
    assert capture.channels == 1
    assert capture.vad is None


def test_audio_capture_without_vad() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    capture = AudioCapture(ring_buffer=rb, vad=None)
    assert capture.vad is None


def test_audio_capture_double_start() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    with patch.dict("sys.modules", {"AudioUnit": MagicMock(), "CoreAudio": MagicMock()}):
        from importlib import reload

        import assistant.audio.capture as capture_module

        reload(capture_module)

        capture = capture_module.AudioCapture(ring_buffer=rb)
        capture.start()
        assert capture.is_running() is True
        # Starting again should be idempotent
        capture.start()
        assert capture.is_running() is True


def test_audio_capture_stop_without_start() -> None:
    from assistant.audio.capture import AudioCapture
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=48000, num_channels=1, dtype=np.int16)

    capture = AudioCapture(ring_buffer=rb)
    assert capture.is_running() is False
    # Stopping without starting should be safe
    capture.stop()
    assert capture.is_running() is False
