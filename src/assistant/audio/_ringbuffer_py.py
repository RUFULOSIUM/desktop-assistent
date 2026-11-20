"""Pure Python ring buffer fallback for when C extension is not available."""

from __future__ import annotations


class PyRingBuffer:
    """Pure Python ring buffer implementation."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer = bytearray(capacity)
        self._capacity = capacity
        self._head = 0  # write position
        self._tail = 0  # read position

    def capacity(self) -> int:
        return self._capacity

    def readable(self) -> int:
        if self._head >= self._tail:
            return self._head - self._tail
        return self._capacity - (self._tail - self._head)

    def writable(self) -> int:
        return self._capacity - 1 - self.readable()

    def clear(self) -> None:
        self._head = 0
        self._tail = 0

    def write(self, data: bytes) -> int:
        if not data:
            return 0

        writable = self.writable()
        if len(data) > writable:
            data = data[:writable]
        if not data:
            return 0

        first_chunk = self._capacity - self._head
        if len(data) <= first_chunk:
            self._buffer[self._head : self._head + len(data)] = data
            self._head = (self._head + len(data)) % self._capacity
        else:
            self._buffer[self._head :] = data[:first_chunk]
            self._buffer[: len(data) - first_chunk] = data[first_chunk:]
            self._head = len(data) - first_chunk
        return len(data)

    def read(self, size: int) -> bytes:
        if size <= 0:
            return b""

        readable = self.readable()
        if size > readable:
            size = readable
        if size == 0:
            return b""

        first_chunk = self._capacity - self._tail
        if size <= first_chunk:
            result = bytes(self._buffer[self._tail : self._tail + size])
            self._tail = (self._tail + size) % self._capacity
        else:
            result = bytes(self._buffer[self._tail :]) + bytes(self._buffer[: size - first_chunk])
            self._tail = size - first_chunk
        return result
