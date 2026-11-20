"""Voice Activity Detection using webrtcvad."""

from __future__ import annotations

from collections.abc import Iterator

try:
    import webrtcvad  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    webrtcvad = None


class VoiceActivityDetector:
    """Voice Activity Detector using WebRTC VAD.

    Args:
        aggressiveness: VAD aggressiveness level (0-3). Higher = more aggressive.
        sample_rate: Audio sample rate in Hz (8000, 16000, 32000, 48000).
        frame_duration_ms: Frame duration in milliseconds (10, 20, or 30).
    """

    _VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)
    _VALID_FRAME_DURATIONS = (10, 20, 30)

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ) -> None:
        if webrtcvad is None:
            raise ImportError("webrtcvad is required for VoiceActivityDetector")

        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be between 0 and 3")

        if sample_rate not in self._VALID_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {self._VALID_SAMPLE_RATES}")

        if frame_duration_ms not in self._VALID_FRAME_DURATIONS:
            raise ValueError(f"frame_duration_ms must be one of {self._VALID_FRAME_DURATIONS}")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._aggressiveness = aggressiveness
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_size_bytes = (
            sample_rate * frame_duration_ms * 2 // 1000
        )  # 2 bytes per sample (16-bit)

    @property
    def frame_size_bytes(self) -> int:
        """Size of one audio frame in bytes."""
        return self._frame_size_bytes

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        """Frame duration in milliseconds."""
        return self._frame_duration_ms

    @property
    def aggressiveness(self) -> int:
        """VAD aggressiveness level."""
        return self._aggressiveness

    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if an audio frame contains speech.

        Args:
            audio_frame: Audio frame as bytes (16-bit PCM, mono).

        Returns:
            True if speech is detected, False otherwise.

        Raises:
            ValueError: If frame size doesn't match expected size.
        """
        if len(audio_frame) != self._frame_size_bytes:
            raise ValueError(
                f"Expected frame size {self._frame_size_bytes} bytes, got {len(audio_frame)} bytes"
            )
        return self._vad.is_speech(audio_frame, self._sample_rate)  # type: ignore[no-any-return]

    def process_stream(self, audio_iter: Iterator[bytes]) -> Iterator[tuple[bytes, bool]]:
        """Process a stream of audio frames.

        Args:
            audio_iter: Iterator yielding audio frames as bytes.

        Yields:
            Tuples of (audio_frame, is_speech) for each frame.
        """
        for frame in audio_iter:
            yield frame, self.is_speech(frame)
