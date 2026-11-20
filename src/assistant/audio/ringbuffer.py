"""Python ring buffer wrapper with numpy support.

This module provides a high-level API for audio ring buffers,
with automatic fallback to pure Python implementation if the
C extension is not available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import ringbuffer as _c_ringbuffer  # type: ignore[import-not-found]

    _HAS_C_EXTENSION = True
except ImportError:
    _HAS_C_EXTENSION = False

from ._ringbuffer_py import PyRingBuffer


class _BaseRingBuffer:
    """Base class for ring buffer implementations."""

    def __init__(self, capacity: int) -> None:
        if _HAS_C_EXTENSION:
            self._rb = _c_ringbuffer.RingBuffer(capacity)
        else:
            self._rb = PyRingBuffer(capacity)

    def capacity(self) -> int:
        return self._rb.capacity()  # type: ignore[no-any-return]

    def readable(self) -> int:
        return self._rb.readable()  # type: ignore[no-any-return]

    def writable(self) -> int:
        return self._rb.writable()  # type: ignore[no-any-return]

    def clear(self) -> None:
        self._rb.clear()

    def write(self, data: bytes) -> int:
        return self._rb.write(data)  # type: ignore[no-any-return]

    def read(self, size: int) -> bytes:
        return self._rb.read(size)  # type: ignore[no-any-return]

    def __enter__(self) -> _BaseRingBuffer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.clear()


class AudioRingBuffer(_BaseRingBuffer):
    """Audio ring buffer with numpy support.

    Args:
        capacity_frames: Number of audio frames the buffer can hold.
        num_channels: Number of audio channels (1 for mono, 2 for stereo).
        dtype: NumPy dtype for audio data (e.g., np.float32, np.int16).
    """

    def __init__(
        self,
        capacity_frames: int,
        num_channels: int = 1,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self._num_channels = num_channels
        self._dtype = np.dtype(dtype)
        self._bytes_per_frame = num_channels * self._dtype.itemsize
        capacity_bytes = capacity_frames * self._bytes_per_frame
        super().__init__(capacity_bytes)
        self._capacity_frames = capacity_frames

    @property
    def capacity_frames(self) -> int:
        return self._capacity_frames

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def write_frames(self, frames: NDArray[np.generic]) -> int:
        """Write audio frames to the buffer.

        Args:
            frames: NumPy array of shape (num_frames,) for mono or
                (num_frames, num_channels) for multi-channel.

        Returns:
            Number of frames written.
        """
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        elif frames.ndim != 2 or frames.shape[1] != self._num_channels:
            raise ValueError(
                f"Expected frames with shape (N, {self._num_channels}), got {frames.shape}"
            )

        # Convert to bytes
        frames = frames.astype(self._dtype, copy=False)
        data = frames.tobytes()

        bytes_written = self.write(data)
        return int(bytes_written // self._bytes_per_frame)

    def read_frames(self, num_frames: int) -> NDArray[np.generic]:
        """Read audio frames from the buffer.

        Args:
            num_frames: Number of frames to read.

        Returns:
            NumPy array of shape (num_frames,) for mono or
            (num_frames, num_channels) for multi-channel.
        """
        if num_frames <= 0:
            frames = np.zeros((0, self._num_channels), dtype=self._dtype)
        else:
            bytes_to_read = num_frames * self._bytes_per_frame
            data = self.read(bytes_to_read)

            if not data:
                frames = np.zeros((0, self._num_channels), dtype=self._dtype)
            else:
                frames = np.frombuffer(data, dtype=self._dtype)
                frames = frames.reshape(-1, self._num_channels)

        if self._num_channels == 1:
            frames = frames.squeeze(axis=1)

        return frames
