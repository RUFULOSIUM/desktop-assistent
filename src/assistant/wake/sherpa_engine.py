"""Wake word detection using sherpa-onnx (completely free, Apache 2.0)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from assistant.config import WakeConfig
from assistant.errors import WakeError

logger = logging.getLogger(__name__)


class SherpaWakeEngine:
    """Wake word detection engine using sherpa-onnx KeywordSpotter.

    Args:
        config: Wake word configuration.
        model_dir: Directory containing sherpa-onnx keyword spotter model files.
            Expected files: tokens.txt, encoder.onnx, decoder.onnx, joiner.onnx,
            and a keywords.txt file.
            Defaults to `src/assistant/wake/models/` relative to this file.
    """

    # sherpa-onnx expects 16kHz audio
    SAMPLE_RATE = 16000
    # Process in chunks of 960 samples (60ms at 16kHz) for smooth detection
    CHUNK_SIZE = 960

    def __init__(
        self,
        config: WakeConfig,
        model_dir: Path | None = None,
    ) -> None:
        self.config = config
        self._model_dir = model_dir or (Path(__file__).parent / "models")
        self._kws: Any = None
        self._stream: Any = None
        self._callback: Callable[[str], None] | None = None
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = np.array([], dtype=np.float32)

    def is_running(self) -> bool:
        """Check if the engine is running."""
        with self._lock:
            return self._running

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback to be called when wake word is detected.

        Args:
            callback: Function taking the detected keyword as argument.
        """
        self._callback = callback

    def start(self) -> None:
        """Initialize and start the wake word engine."""
        with self._lock:
            if self._running:
                return

            if not self.config.enabled:
                logger.info("Wake word detection disabled in config")
                self._running = True
                return

            # Find model files (support versioned filenames)
            encoder_files = list(self._model_dir.glob("encoder*.onnx"))
            decoder_files = list(self._model_dir.glob("decoder*.onnx"))
            joiner_files = list(self._model_dir.glob("joiner*.onnx"))

            if not encoder_files:
                raise WakeError(
                    f"No encoder model found in {self._model_dir}. "
                    "Download sherpa-onnx keyword spotter models from "
                    "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html"
                )
            if not decoder_files:
                raise WakeError(f"No decoder model found in {self._model_dir}")
            if not joiner_files:
                raise WakeError(f"No joiner model found in {self._model_dir}")

            # Prefer non-int8 (fp32) models for better accuracy
            encoder_path = next(
                (f for f in encoder_files if "int8" not in f.name), encoder_files[0]
            )
            decoder_path = next(
                (f for f in decoder_files if "int8" not in f.name), decoder_files[0]
            )
            joiner_path = next(
                (f for f in joiner_files if "int8" not in f.name), joiner_files[0]
            )

            if not (self._model_dir / "tokens.txt").exists():
                raise WakeError(f"tokens.txt not found in {self._model_dir}")
            if not (self._model_dir / "keywords.txt").exists():
                raise WakeError(f"keywords.txt not found in {self._model_dir}")

            try:
                import sherpa_onnx  # type: ignore
            except ImportError as e:
                raise WakeError(
                    "sherpa-onnx not installed. Install with: "
                    "pip install 'desktop-assistant[wake]'"
                ) from e

            logger.info(f"Loading sherpa-onnx keyword spotter model from {self._model_dir}")

            try:
                # Read keywords from file to verify wake word is in list
                keywords_file = self._model_dir / "keywords.txt"
                keywords_content = keywords_file.read_text(encoding="utf-8")
                if self.config.word not in keywords_content:
                    logger.warning(
                        f"Wake word '{self.config.word}' not found in keywords.txt. "
                        "Add it to the file for detection to work."
                    )

                self._kws = sherpa_onnx.KeywordSpotter(
                    tokens=str(self._model_dir / "tokens.txt"),
                    encoder=str(encoder_path),
                    decoder=str(decoder_path),
                    joiner=str(joiner_path),
                    num_threads=2,
                    keywords_file=str(keywords_file),
                    provider="cpu",
                )
            except Exception as e:
                raise WakeError(f"Failed to load keyword spotter model: {e}") from e

            self._stream = self._kws.create_stream()
            self._audio_buffer = np.array([], dtype=np.float32)
            self._running = True
            logger.info("Wake word engine started (sherpa-onnx)")

    def stop(self) -> None:
        """Stop the wake word engine and release resources."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stream = None
            self._kws = None
            self._audio_buffer = np.array([], dtype=np.float32)
            logger.info("Wake word engine stopped")

    def process_frame(self, audio_frame: bytes) -> bool:
        """Process a single audio frame for wake word detection.

        Args:
            audio_frame: Audio frame as bytes (16-bit PCM, 16kHz mono).
                         Expected size: 1920 bytes (960 samples / 60ms at 16kHz).

        Returns:
            True if wake word detected, False otherwise.
        """
        with self._lock:
            if (
                not self._running
                or not self.config.enabled
                or self._kws is None
                or self._stream is None
            ):
                return False

        # Convert bytes to float32 array (-1.0 to 1.0)
        audio_int16 = np.frombuffer(audio_frame, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Accumulate audio
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_float32])

        # Process when we have enough data
        if len(self._audio_buffer) < self.CHUNK_SIZE:
            return False

        detected = False
        while len(self._audio_buffer) >= self.CHUNK_SIZE:
            chunk = self._audio_buffer[: self.CHUNK_SIZE]
            self._audio_buffer = self._audio_buffer[self.CHUNK_SIZE :]

            try:
                self._stream.accept_waveform(self.SAMPLE_RATE, chunk)

                while self._kws.is_ready(self._stream):
                    self._kws.decode_stream(self._stream)
                    result = self._kws.get_result(self._stream)
                    if result:
                        detected = True
                        logger.info(f"Wake word detected: {result}")
                        if self._callback:
                            try:
                                self._callback(result)
                            except Exception as e:
                                logger.error(f"Wake word callback error: {e}")
                        # Reset stream after detection
                        self._kws.reset_stream(self._stream)
                        break

            except Exception as e:
                logger.error(f"Wake word prediction error: {e}")

        return detected