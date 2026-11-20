def test_vad_basic() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)

    # Silence frame (all zeros)
    silence = bytes(960)  # 30ms at 16kHz mono 16-bit = 960 bytes
    assert vad.is_speech(silence) is False

    # Simple tone (not real speech, but tests API)
    # Real speech test would need actual audio data
    assert hasattr(vad, "process_stream")


def test_vad_frame_size() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    # 30ms * 16000Hz * 2 bytes = 960 bytes
    assert vad.frame_size_bytes == 960

    vad2 = VoiceActivityDetector(sample_rate=8000, frame_duration_ms=20)
    # 20ms * 8000Hz * 2 bytes = 320 bytes
    assert vad2.frame_size_bytes == 320


def test_vad_invalid_aggressiveness() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    try:
        VoiceActivityDetector(aggressiveness=4)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "aggressiveness must be between 0 and 3" in str(e)

    try:
        VoiceActivityDetector(aggressiveness=-1)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "aggressiveness must be between 0 and 3" in str(e)


def test_vad_invalid_frame_duration() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    try:
        VoiceActivityDetector(frame_duration_ms=40)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "frame_duration_ms must be one of" in str(e)
        assert "10" in str(e)
        assert "20" in str(e)
        assert "30" in str(e)

    try:
        VoiceActivityDetector(frame_duration_ms=5)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "frame_duration_ms must be one of" in str(e)


def test_vad_invalid_sample_rate() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    try:
        VoiceActivityDetector(sample_rate=44100)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "sample_rate must be one of" in str(e)
        assert "8000" in str(e)
        assert "16000" in str(e)
        assert "32000" in str(e)
        assert "48000" in str(e)


def test_vad_is_speech_wrong_frame_size() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    # Wrong frame size
    wrong_frame = bytes(100)

    try:
        vad.is_speech(wrong_frame)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Expected frame size 960 bytes" in str(e)
        assert "got 100 bytes" in str(e)


def test_vad_process_stream() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)

    # Create an iterator of silence frames
    silence = bytes(960)
    frames = [silence] * 5

    results = list(vad.process_stream(iter(frames)))
    assert len(results) == 5
    for frame, is_speech in results:
        assert frame == silence
        assert is_speech is False


def test_vad_process_stream_empty() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(aggressiveness=3, sample_rate=16000, frame_duration_ms=30)

    results = list(vad.process_stream(iter([])))
    assert len(results) == 0


def test_vad_properties() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(aggressiveness=2, sample_rate=8000, frame_duration_ms=20)

    assert vad.aggressiveness == 2
    assert vad.sample_rate == 8000
    assert vad.frame_duration_ms == 20
    assert vad.frame_size_bytes == 320


def test_vad_all_aggressiveness_levels() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    for level in range(4):
        vad = VoiceActivityDetector(aggressiveness=level)
        assert vad.aggressiveness == level


def test_vad_all_sample_rates() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    for rate in (8000, 16000, 32000, 48000):
        vad = VoiceActivityDetector(sample_rate=rate)
        assert vad.sample_rate == rate


def test_vad_all_frame_durations() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    for duration in (10, 20, 30):
        vad = VoiceActivityDetector(frame_duration_ms=duration)
        assert vad.frame_duration_ms == duration


def test_vad_frame_size_calculation() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    # 10ms frames
    vad10 = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=10)
    assert vad10.frame_size_bytes == 320  # 16000 * 10 * 2 / 1000

    # 20ms frames
    vad20 = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=20)
    assert vad20.frame_size_bytes == 640  # 16000 * 20 * 2 / 1000

    # 30ms frames
    vad30 = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    assert vad30.frame_size_bytes == 960  # 16000 * 30 * 2 / 1000


def test_vad_is_speech_with_different_rates() -> None:
    from assistant.audio.vad import VoiceActivityDetector

    # Test with different sample rates
    for rate in (8000, 16000, 32000, 48000):
        vad = VoiceActivityDetector(sample_rate=rate, frame_duration_ms=30)
        frame_size = rate * 30 * 2 // 1000
        silence = bytes(frame_size)
        assert vad.is_speech(silence) is False
