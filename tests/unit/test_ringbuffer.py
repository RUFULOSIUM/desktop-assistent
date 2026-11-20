import numpy as np  # type: ignore[import-not-found]
import ringbuffer  # type: ignore[import-not-found]


def test_ringbuffer_basic() -> None:
    rb = ringbuffer.RingBuffer(1024)
    assert rb.capacity() == 1024
    assert rb.readable() == 0
    # One byte reserved to distinguish full/empty
    assert rb.writable() == 1023

    written = rb.write(b"hello")
    assert written == 5
    assert rb.readable() == 5
    assert rb.writable() == 1018

    data = rb.read(5)
    assert data == b"hello"
    assert rb.readable() == 0
    assert rb.writable() == 1023


def test_ringbuffer_wraparound() -> None:
    rb = ringbuffer.RingBuffer(16)
    rb.write(b"01234567")  # 8 bytes
    rb.read(4)  # read 4, 4 remaining
    rb.write(b"abcd")  # write 4, buffer has 8
    assert rb.readable() == 8
    data = rb.read(8)
    assert data == b"4567abcd"


def test_ringbuffer_clear() -> None:
    rb = ringbuffer.RingBuffer(32)
    rb.write(b"test")
    rb.clear()
    assert rb.readable() == 0
    assert rb.writable() == 31  # one byte reserved


def test_ringbuffer_full() -> None:
    rb = ringbuffer.RingBuffer(16)
    # Fill buffer completely (15 bytes max due to reserved slot)
    written = rb.write(b"0123456789abcde")  # 15 bytes
    assert written == 15
    assert rb.readable() == 15
    assert rb.writable() == 0
    # Try to write more - should return 0
    written = rb.write(b"f")
    assert written == 0


def test_ringbuffer_read_partial() -> None:
    rb = ringbuffer.RingBuffer(32)
    rb.write(b"hello world")
    data = rb.read(5)
    assert data == b"hello"
    assert rb.readable() == 6
    data = rb.read(10)  # read more than available
    assert data == b" world"
    assert rb.readable() == 0


def test_audio_ringbuffer_numpy() -> None:
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


def test_audio_ringbuffer_stereo() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=512, num_channels=2, dtype=np.float32)
    assert rb.capacity_frames == 512
    assert rb.num_channels == 2

    # Write stereo frames
    frames = np.zeros((100, 2), dtype=np.float32)
    frames[:, 0] = 0.5  # left channel
    frames[:, 1] = -0.5  # right channel
    written = rb.write_frames(frames)
    assert written == 100

    read_frames = rb.read_frames(100)
    assert read_frames.shape == (100, 2)
    assert np.allclose(read_frames[:, 0], 0.5)
    assert np.allclose(read_frames[:, 1], -0.5)


def test_audio_ringbuffer_int16() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=1024, num_channels=1, dtype=np.int16)
    frames = np.zeros(100, dtype=np.int16)
    frames[:] = 1000
    written = rb.write_frames(frames)
    assert written == 100

    read_frames = rb.read_frames(100)
    assert read_frames.dtype == np.int16
    assert np.allclose(read_frames, 1000)


def test_audio_ringbuffer_wraparound() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=16, num_channels=1, dtype=np.float32)
    frames = np.arange(8, dtype=np.float32)
    rb.write_frames(frames)
    rb.read_frames(4)
    frames2 = np.arange(4, 8, dtype=np.float32)
    rb.write_frames(frames2)
    assert rb.readable() == 8 * 4  # 8 frames * 4 bytes
    read_frames = rb.read_frames(8)
    expected = np.concatenate([np.arange(4, 8), np.arange(4, 8)])
    assert np.allclose(read_frames, expected)


def test_audio_ringbuffer_clear() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=32, num_channels=1, dtype=np.float32)
    frames = np.ones(10, dtype=np.float32)
    rb.write_frames(frames)
    rb.clear()
    assert rb.readable() == 0
    read_frames = rb.read_frames(10)
    assert len(read_frames) == 0


def test_audio_ringbuffer_read_partial() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=32, num_channels=1, dtype=np.float32)
    frames = np.arange(10, dtype=np.float32)
    rb.write_frames(frames)
    read_frames = rb.read_frames(5)
    assert len(read_frames) == 5
    assert np.allclose(read_frames, np.arange(5))
    read_frames = rb.read_frames(10)  # read more than available
    assert len(read_frames) == 5
    assert np.allclose(read_frames, np.arange(5, 10))


def test_audio_ringbuffer_context_manager() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    with AudioRingBuffer(capacity_frames=100, num_channels=1, dtype=np.float32) as rb:
        frames = np.ones(10, dtype=np.float32)
        rb.write_frames(frames)
        assert rb.readable() == 40  # 10 frames * 4 bytes
    # After context exit, buffer should be cleared
    # Note: __exit__ calls clear()


# mypy: disable-error-code="attr-defined"
def test_audio_ringbuffer_write_read_error_shape() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=100, num_channels=2, dtype=np.float32)
    frames = np.zeros((10, 3), dtype=np.float32)  # Wrong number of channels
    try:
        rb.write_frames(frames)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Expected frames with shape" in str(e)


def test_audio_ringbuffer_zero_frames() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=100, num_channels=1, dtype=np.float32)
    read_frames = rb.read_frames(0)
    assert len(read_frames) == 0
    # For mono, shape is (0,) after squeeze
    assert read_frames.shape == (0,)

    # For stereo, shape is (0, 2)
    rb2 = AudioRingBuffer(capacity_frames=100, num_channels=2, dtype=np.float32)
    read_frames2 = rb2.read_frames(0)
    assert len(read_frames2) == 0
    assert read_frames2.shape == (0, 2)


def test_audio_ringbuffer_dtype_property() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=100, num_channels=1, dtype=np.float32)
    assert rb.dtype == np.dtype(np.float32)

    rb2 = AudioRingBuffer(capacity_frames=100, num_channels=2, dtype=np.int16)
    assert rb2.dtype == np.dtype(np.int16)


def test_audio_ringbuffer_fallback() -> None:
    # Test that fallback works when C extension not available
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(1024)
    rb.write(b"test")
    assert rb.read(4) == b"test"


def test_pyringbuffer_basic() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(1024)
    assert rb.capacity() == 1024
    assert rb.readable() == 0
    assert rb.writable() == 1023

    written = rb.write(b"hello")
    assert written == 5
    assert rb.readable() == 5
    assert rb.writable() == 1018

    data = rb.read(5)
    assert data == b"hello"
    assert rb.readable() == 0
    assert rb.writable() == 1023


def test_pyringbuffer_wraparound() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(16)
    rb.write(b"01234567")  # 8 bytes
    rb.read(4)  # read 4, 4 remaining
    rb.write(b"abcd")  # write 4, buffer has 8
    assert rb.readable() == 8
    data = rb.read(8)
    assert data == b"4567abcd"


def test_pyringbuffer_clear() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(32)
    rb.write(b"test")
    rb.clear()
    assert rb.readable() == 0
    assert rb.writable() == 31


def test_pyringbuffer_full() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(16)
    written = rb.write(b"0123456789abcde")  # 15 bytes
    assert written == 15
    assert rb.readable() == 15
    assert rb.writable() == 0
    written = rb.write(b"f")
    assert written == 0


def test_pyringbuffer_read_partial() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(32)
    rb.write(b"hello world")
    data = rb.read(5)
    assert data == b"hello"
    assert rb.readable() == 6
    data = rb.read(10)  # read more than available
    assert data == b" world"
    assert rb.readable() == 0


def test_pyringbuffer_write_empty() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(32)
    written = rb.write(b"")
    assert written == 0
    assert rb.readable() == 0


def test_pyringbuffer_read_empty() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(32)
    data = rb.read(0)
    assert data == b""
    data = rb.read(5)
    assert data == b""


def test_pyringbuffer_invalid_capacity() -> None:
    from assistant.audio._ringbuffer_py import PyRingBuffer

    try:
        PyRingBuffer(0)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "capacity must be positive" in str(e)

    try:
        PyRingBuffer(-1)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "capacity must be positive" in str(e)


def test_pyringbuffer_wraparound_write_read() -> None:
    """Test PyRingBuffer wraparound for both write and read (head < tail)."""
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(16)
    # Write 8 bytes, read 6, write 10 more (wraparound write: 8+10=18 > 16)
    rb.write(b"01234567")
    rb.read(6)  # tail=6, head=8
    rb.write(b"abcdefghij")  # 10 bytes, first_chunk=8, so wraps
    assert rb.readable() == 12
    # Now read with wraparound (head < tail)
    data = rb.read(12)
    assert data == b"67abcdefghij"


def test_pyringbuffer_read_wraparound() -> None:
    """Test PyRingBuffer read with wraparound (head < tail)."""
    from assistant.audio._ringbuffer_py import PyRingBuffer

    rb = PyRingBuffer(16)
    # Fill buffer to create head < tail scenario
    rb.write(b"0123456789abcdef")  # 15 bytes (max)
    rb.read(10)  # tail=10, head=15 (head < tail)
    # Now read should wrap around
    data = rb.read(5)
    assert data == b"abcde"


def test_audio_ringbuffer_context_manager_exit() -> None:
    from assistant.audio.ringbuffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_frames=100, num_channels=1, dtype=np.float32)
    frames = np.ones(10, dtype=np.float32)
    rb.write_frames(frames)
    assert rb.readable() == 40
    rb.__exit__(None, None, None)
    assert rb.readable() == 0
