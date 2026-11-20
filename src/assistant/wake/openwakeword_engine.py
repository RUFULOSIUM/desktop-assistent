"""Wake word detection using openWakeWord."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from assistant.config import WakeConfig
from assistant.errors import WakeError

logger = logging.getLogger(__name__)


class WakeWordEngine:
    """Wake word detection engine using openWakeWord.

    Args:
        config: Wake word configuration.
        model_dir: Directory containing .onnx model files. Defaults to
            `src/assistant/wake/models/` relative to this file.
    """

    # openWakeWord processes 80ms frames (1280 samples at 16kHz)
    FRAME_SIZE_SAMPLES = 1280
    FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 16-bit = 2 bytes per sample

    def __init__(
        self,
        config: WakeConfig,
        model_dir: Path | None = None,
    ) -> None:
        self.config = config
        self._model_dir = model_dir or (Path(__file__).parent / "models")
        self._model = None
        self._callback: Callable[[str], None] | None = None
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = bytearray()
        self._threshold = 0.5  # Default threshold from openWakeWord docs

    def is_running(self) -> bool:
        """Check if the engine is running."""
        with self._lock:
            return self._running

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback to be called when wake word is detected.

        Args:
            callback: Function taking the detected model name as argument.
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

            # Find model file
            model_files = list(self._model_dir.glob("*.onnx"))
            if not model_files:
                raise WakeError(
                    f"No .onnx model files found in {self._model_dir}. "
                    "Download a model or train one using openWakeWord."
                )

            model_path = model_files[0]
            logger.info(f"Loading wake word model: {model_path.name}")

            try:
                # Import openWakeWord lazily
                from openwakeword.model import Model  # type: ignore[import-not-found]

                self._model = Model(
                    wakeword_models=[str(model_path)],
                    inference_framework="onnx",
                    vad_threshold=0.5,  # Use Silero VAD to reduce false positives
                )
            except ImportError as e:
                raise WakeError(
                    "openWakeWord not installed. Install with: "
                    "pip install 'desktop-assistant[wake]'"
                ) from e
            except Exception as e:
                raise WakeError(f"Failed to load wake word model: {e}") from e

            self._audio_buffer.clear()
            self._running = True
            logger.info("Wake word engine started")

    def stop(self) -> None:
        """Stop the wake word engine and release resources."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._model = None
            self._audio_buffer.clear()
            logger.info("Wake word engine stopped")

    def process_frame(self, audio_frame: bytes) -> bool:
        """Process a single audio frame for wake word detection.

        Args:
            audio_frame: Audio frame as bytes (16-bit PCM, 16kHz mono).
                         Expected size: 960 bytes (30ms frame).

        Returns:
            True if wake word detected, False otherwise.
        """
        with self._lock:
            if not self._running or not self.config.enabled or self._model is None:
                return False

        # Accumulate audio until we have enough for openWakeWord (1280 samples = 80ms)
        self._audio_buffer.extend(audio_frame)

        if len(self._audio_buffer) < self.FRAME_SIZE_BYTES:
            return False

        # Process complete 80ms frames
        detected = False
        while len(self._audio_buffer) >= self.FRAME_SIZE_BYTES:
            frame = bytes(self._audio_buffer[: self.FRAME_SIZE_BYTES])
            del self._audio_buffer[: self.FRAME_SIZE_BYTES]

            try:
                # Convert to numpy array (int16)
                audio_data = np.frombuffer(frame, dtype=np.int16)

                # Get predictions
                predictions = self._model.predict(audio_data)

                # Check if any model exceeded threshold
                for model_name, score in predictions.items():
                    if score >= self._threshold:
                        detected = True
                        logger.info(f"Wake word detected: {model_name} (score={score:.3f})")
                        if self._callback:
                            try:
                                self._callback(model_name)
                            except Exception as e:
                                logger.error(f"Wake word callback error: {e}")
                        break

            except Exception as e:
                logger.error(f"Wake word prediction error: {e}")

        return detected
