"""Speech-to-Text abstraction and provider implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from assistant.config import STTConfig
from assistant.errors import STTError

logger = logging.getLogger(__name__)


class STTProvider(ABC):
    """Abstract base class for STT providers."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model = None

    @abstractmethod
    def transcribe(self, audio: NDArray[np.generic], sample_rate: int) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Transcribed text.

        Raises:
            STTError: If transcription fails.
        """
        pass

    @abstractmethod
    def transcribe_stream(
        self, audio_chunks: Iterator[NDArray[np.generic]], sample_rate: int
    ) -> Iterator[str]:
        """Stream transcribe audio chunks.

        Args:
            audio_chunks: Iterator of audio chunks as numpy arrays.
            sample_rate: Sample rate of the audio in Hz.

        Yields:
            Partial transcription results.

        Raises:
            STTError: If transcription fails.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Initialize the provider."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether the provider runs locally (on-device)."""
        pass


def create_stt_provider(config: STTConfig) -> STTProvider:
    """Factory function to create an STT provider based on config.

    Args:
        config: STT configuration.

    Returns:
        STTProvider instance.

    Raises:
        STTError: If the provider cannot be created.
    """
    if config.provider == "cloud":
        return CloudSTTProvider(config)
    elif config.provider == "local":
        return LocalSTTProvider(config)
    elif config.provider == "auto":
        return AutoSTTProvider(config)
    else:
        raise STTError(f"Unknown STT provider: {config.provider}", provider=config.provider)


class CloudSTTProvider(STTProvider):
    """Cloud-based STT provider (e.g., OpenAI Whisper API)."""

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"cloud:{self.config.cloud_service}"

    @property
    def is_local(self) -> bool:
        return False

    def start(self) -> None:
        if self.config.cloud_service == "whisper-api":
            try:
                from openai import OpenAI  # type: ignore[import-not-found]

                self._client = OpenAI()
            except ImportError as e:
                raise STTError(
                    "openai package not installed. Install with: "
                    "pip install 'desktop-assistant[stt]'",
                    provider=self.name,
                ) from e
        else:
            raise STTError(
                f"Unsupported cloud STT service: {self.config.cloud_service}",
                provider=self.name,
            )

    def stop(self) -> None:
        self._client = None

    def transcribe(self, audio: NDArray[np.generic], _sample_rate: int) -> str:
        if self._client is None:
            self.start()

        try:
            # Convert to bytes (16-bit PCM)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            import io

            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            response = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=self.config.language if self.config.language != "auto" else None,
            )
            return str(response.text).strip()
        except Exception as e:
            raise STTError(f"Cloud transcription failed: {e}", provider=self.name) from e

    def transcribe_stream(
        self, audio_chunks: Iterator[NDArray[np.generic]], sample_rate: int
    ) -> Iterator[str]:
        # Cloud providers typically don't support streaming well
        # Collect all chunks and transcribe once
        chunks = list(audio_chunks)
        if not chunks:
            return
        audio = np.concatenate(chunks)
        yield self.transcribe(audio, sample_rate)


class LocalSTTProvider(STTProvider):
    """Local (on-device) STT provider using faster-whisper or MLX Whisper."""

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._engine: Any = None

    @property
    def name(self) -> str:
        return f"local:{self.config.local_model}"

    @property
    def is_local(self) -> bool:
        return True

    def start(self) -> None:
        # Try faster-whisper first, then MLX Whisper
        try:
            from assistant.stt.faster_whisper import FasterWhisperSTT

            self._engine = FasterWhisperSTT(self.config)
            self._engine.start()
        except Exception as e:
            logger.warning(f"faster-whisper unavailable: {e}, trying MLX Whisper")
            try:
                from assistant.stt.mlx_whisper import MLXWhisperSTT

                self._engine = MLXWhisperSTT(self.config)
                self._engine.start()
            except Exception as e2:
                raise STTError(
                    f"No local STT engine available: faster-whisper ({e}), mlx-whisper ({e2})",
                    provider=self.name,
                ) from e2

    def stop(self) -> None:
        if self._engine:
            self._engine.stop()
            self._engine = None

    def transcribe(self, audio: NDArray[np.generic], sample_rate: int) -> str:
        if self._engine is None:
            self.start()
        assert self._engine is not None
        return self._engine.transcribe(audio, sample_rate)  # type: ignore[no-any-return]

    def transcribe_stream(
        self, audio_chunks: Iterator[NDArray[np.generic]], sample_rate: int
    ) -> Iterator[str]:
        if self._engine is None:
            self.start()
        assert self._engine is not None
        yield from self._engine.transcribe_stream(audio_chunks, sample_rate)


class AutoSTTProvider(STTProvider):
    """Auto-select STT provider with fallback chain."""

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._providers: list[STTProvider] = []
        self._current_provider: STTProvider | None = None

    @property
    def name(self) -> str:
        return "auto"

    @property
    def is_local(self) -> bool:
        return self._current_provider.is_local if self._current_provider else True

    def start(self) -> None:
        # Build fallback chain from config
        for provider_name in self.config.fallback_chain:
            if provider_name == "cloud":
                self._providers.append(CloudSTTProvider(self.config))
            elif provider_name == "local":
                self._providers.append(LocalSTTProvider(self.config))

        if not self._providers:
            raise STTError("No providers in fallback chain", provider=self.name)

        # Try each provider until one works
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                provider.start()
                self._current_provider = provider
                logger.info(f"Auto STT selected provider: {provider.name}")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}")

        raise STTError(
            f"All STT providers failed. Last error: {last_error}", provider=self.name
        ) from last_error

    def stop(self) -> None:
        for provider in self._providers:
            provider.stop()
        self._current_provider = None

    def transcribe(self, audio: NDArray[np.generic], sample_rate: int) -> str:
        if self._current_provider is None:
            self.start()
        assert self._current_provider is not None
        return self._current_provider.transcribe(audio, sample_rate)

    def transcribe_stream(
        self, audio_chunks: Iterator[NDArray[np.generic]], sample_rate: int
    ) -> Iterator[str]:
        if self._current_provider is None:
            self.start()
        assert self._current_provider is not None
        yield from self._current_provider.transcribe_stream(audio_chunks, sample_rate)
