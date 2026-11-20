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
