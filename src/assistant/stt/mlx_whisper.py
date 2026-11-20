"""MLX Whisper STT implementation for Apple Silicon."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from assistant.config import STTConfig
from assistant.errors import STTError

logger = logging.getLogger(__name__)


class MLXWhisperSTT:
    """Local STT using MLX Whisper (optimized for Apple Silicon)."""

    # Model size mapping for MLX Whisper
    MODEL_SIZES = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v3",
        "large-v2": "large-v2",
        "large-v3": "large-v3",
    }

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model: bool | None = None
        self._model_size = self.MODEL_SIZES.get(config.local_model, config.local_model)
        self._mlx_whisper: Any = None

    @property
    def name(self) -> str:
        return f"mlx-whisper:{self._model_size}"

    def start(self) -> None:
        """Load the MLX Whisper model."""
        try:
            import mlx_whisper  # type: ignore[import-not-found]

            self._mlx_whisper = mlx_whisper
        except ImportError as e:
            raise STTError(
                "mlx-whisper not installed. Install with: pip install 'desktop-assistant[stt]'",
                provider=self.name,
            ) from e

        try:
            logger.info(f"Loading MLX Whisper model: {self._model_size}")
            # MLX Whisper loads model on first transcribe call
            # We just verify it's available
            self._model = True
            logger.info(f"MLX Whisper ready: {self._model_size}")
        except Exception as e:
            raise STTError(f"Failed to initialize MLX Whisper: {e}", provider=self.name) from e

    def stop(self) -> None:
        """Release the model."""
        self._model = None
        self._mlx_whisper = None

    def transcribe(self, audio: NDArray[np.generic], sample_rate: int) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, -1.0 to 1.0).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Transcribed text.
        """
        if self._mlx_whisper is None:
            self.start()

        try:
            # Resample if needed (MLX Whisper expects 16kHz)
            if sample_rate != 16000:
                audio = self._resample(audio, sample_rate, 16000)

            # Convert float32 to int16 for MLX Whisper
            audio_int16 = (audio * 32767).astype(np.int16)

            # Transcribe using MLX Whisper
            assert self._mlx_whisper is not None
            result = self._mlx_whisper.transcribe(
                audio_int16,
                path_or_hf_repo=f"mlx-community/whisper-{self._model_size}",
                language=self.config.language if self.config.language != "auto" else None,
            )

            text = str(result.get("text", "")).strip()
            logger.debug(f"Transcribed: {text}")
            return text

        except Exception as e:
            raise STTError(f"Transcription failed: {e}", provider=self.name) from e

    def transcribe_stream(
        self, audio_chunks: Iterator[NDArray[np.generic]], sample_rate: int
    ) -> Iterator[str]:
        """Stream transcribe audio chunks.

        Note: MLX Whisper doesn't support true streaming, so we buffer
        and transcribe when we have enough audio.
        """
        if self._mlx_whisper is None:
            self.start()

        buffer: list[NDArray[np.generic]] = []
        buffer_duration = 0.0
        min_buffer_seconds = 2.0  # Minimum audio before transcribing

        for chunk in audio_chunks:
            buffer.append(chunk)
            buffer_duration += len(chunk) / sample_rate

            if buffer_duration >= min_buffer_seconds:
                audio = np.concatenate(buffer)
                text = self.transcribe(audio, sample_rate)
                if text:
                    yield text
                buffer = []
                buffer_duration = 0.0

        # Flush remaining buffer
        if buffer:
            audio = np.concatenate(buffer)
            text = self.transcribe(audio, sample_rate)
            if text:
                yield text

    def _resample(
        self, audio: NDArray[np.generic], orig_sr: int, target_sr: int
    ) -> NDArray[np.float32]:
        """Resample audio using simple linear interpolation."""
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        try:
            from scipy import signal  # type: ignore[import-untyped]

            resampled = signal.resample(audio, int(len(audio) * target_sr / orig_sr))
            return resampled.astype(np.float32)  # type: ignore[no-any-return]
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.arange(new_length, dtype=np.float64) / ratio
            interpolated = np.interp(indices, np.arange(len(audio), dtype=np.float64), audio)  # type: ignore[arg-type]
            return interpolated.astype(np.float32)
