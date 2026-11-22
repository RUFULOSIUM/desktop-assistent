"""Main pipeline connecting wake → STT → Router → LLM/Agent → UI."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .agent import AgentDispatcher, DispatchResult, create_dispatcher
from .audio import SoundEvent, get_sound_manager
from .audio.capture import AudioCapture
from .audio.ringbuffer import AudioRingBuffer
from .audio.vad import VoiceActivityDetector
from .config import Config
from .llm import LLMProvider, create_llm_provider
from .notify import get_notification_manager
from .router import AGENT_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, RouterClassifier, create_classifier
from .stt import STTProvider, create_stt_provider
from .tts import TTSProvider, create_tts_provider
from .wake import WakeWordEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    success: bool
    response: str | None = None
    error: str | None = None
    is_agent_task: bool = False
    agent_result: DispatchResult | None = None


class AssistantPipeline:
    """Main pipeline orchestrating all components."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._stt: STTProvider | None = None
        self._llm: LLMProvider | None = None
        self._router: RouterClassifier | None = None
        self._agent: AgentDispatcher | None = None
        self._wake: WakeWordEngine | None = None
        self._tts: TTSProvider | None = None
        self._running = False
        self._lock = threading.Lock()

        # Audio capture components (initialized on demand)
        self._capture: AudioCapture | None = None
        self._ring_buffer: AudioRingBuffer | None = None
        self._vad: VoiceActivityDetector | None = None
        self._listen_thread: threading.Thread | None = None
        self._stop_listening = threading.Event()

        # Sound and notification managers
        self._sound_manager = None
        self._notification_manager = None

        # Callbacks
        self._on_wake_detected: Callable[[], None] | None = None
        self._on_listening: Callable[[], None] | None = None
        self._on_thinking: Callable[[], None] | None = None
        self._on_speaking: Callable[[str], None] | None = None
        self._on_idle: Callable[[], None] | None = None

        # Dropped file context (cleared after use)
        self._dropped_files: list[str] | None = None

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        with self._lock:
            if self._running:
                return

            # Initialize sound manager
            if self.config.ui.sound_effects:
                from .audio import SoundConfig

                sound_config = SoundConfig(
                    enabled=self.config.sound.enabled,
                    volume=self.config.sound.volume,
                    wake_detected=self.config.sound.wake_detected,
                    listening_start=self.config.sound.listening_start,
                    listening_end=self.config.sound.listening_end,
                    thinking_start=self.config.sound.thinking_start,
                    speaking_start=self.config.sound.speaking_start,
                    task_complete=self.config.sound.task_complete,
                    error=self.config.sound.error,
                    permission_needed=self.config.sound.permission_needed,
                )
                self._sound_manager = get_sound_manager(sound_config)

            # Initialize notification manager
            if self.config.ui.notifications:
                from .notify import NotificationConfig

                notif_config = NotificationConfig(
                    enabled=self.config.ui.notifications,
                    min_verbosity=self.config.ui.verbosity,
                )
                self._notification_manager = get_notification_manager(notif_config)

            # Initialize STT
            self._stt = create_stt_provider(self.config.stt)
            self._stt.start()

            # Initialize LLM
            self._llm = create_llm_provider(self.config.llm)
            self._llm.start()

            # Initialize TTS
            self._tts = create_tts_provider(self.config)
            self._tts.start()

            # Initialize Router
            self._router = create_classifier()

            # Initialize Agent
            self._agent = create_dispatcher()

            # Initialize Wake Word
            self._wake = WakeWordEngine(self.config.wake)
            self._wake.set_callback(self._on_wake_word)
            self._wake.start()

            self._running = True
            logger.info("Pipeline initialized")

    def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        with self._lock:
            if not self._running:
                return

            # Stop listening if active
            self._stop_listening.set()
            if self._listen_thread and self._listen_thread.is_alive():
                self._listen_thread.join(timeout=2.0)

            if self._capture:
                self._capture.stop()
                self._capture = None

            if self._wake:
                self._wake.stop()
            if self._stt:
                self._stt.stop()
            if self._llm:
                self._llm.stop()
            if self._tts:
                self._tts.stop()

            self._running = False
            logger.info("Pipeline shutdown")

    def set_callbacks(
        self,
        on_wake_detected: Callable[[], None] | None = None,
        on_listening: Callable[[], None] | None = None,
        on_thinking: Callable[[], None] | None = None,
        on_speaking: Callable[[str], None] | None = None,
        on_idle: Callable[[], None] | None = None,
    ) -> None:
        """Set UI callbacks for state changes."""
        self._on_wake_detected = on_wake_detected
        self._on_listening = on_listening
        self._on_thinking = on_thinking
        self._on_speaking = on_speaking
        self._on_idle = on_idle

    def set_dropped_files(self, file_paths: list[str]) -> None:
        """Set dropped file paths as context for next transcription."""
        self._dropped_files = file_paths
        logger.info(f"Set dropped file context: {file_paths}")

    def _get_and_clear_dropped_files(self) -> list[str] | None:
        """Get and clear the dropped file context."""
        files = self._dropped_files
        self._dropped_files = None
        return files

    def _on_wake_word(self, model_name: str) -> None:
        """Internal wake word callback."""
        logger.info(f"Wake word detected: {model_name}")
        # Play wake detected sound
        if self._sound_manager:
            self._sound_manager.play(SoundEvent.WAKE_DETECTED)
        if self._on_wake_detected:
            self._on_wake_detected()
        # Start listening
        self._start_listening()

    def _start_listening(self) -> None:
        """Start audio capture, VAD, and STT in a background thread."""
        # Play listening start sound
        if self._sound_manager:
            self._sound_manager.play(SoundEvent.LISTENING_START)
        if self._on_listening:
            self._on_listening()

        # Reset stop event
        self._stop_listening.clear()

        # Initialize audio components if not already done
        if self._ring_buffer is None:
            self._ring_buffer = AudioRingBuffer(
                capacity_frames=self.config.audio.sample_rate * 10,  # 10 seconds
                num_channels=self.config.audio.channels,
                dtype=np.int16,
            )
        if self._vad is None:
            try:
                self._vad = VoiceActivityDetector(
                    aggressiveness=self.config.audio.vad_aggressiveness,
                    sample_rate=self.config.audio.sample_rate,
                    frame_duration_ms=self.config.audio.frame_duration_ms,
                )
            except ImportError:
                logger.warning("webrtcvad not available, VAD disabled")
                self._vad = None

        if self._capture is None:
            try:
                self._capture = AudioCapture(
                    ring_buffer=self._ring_buffer,
                    vad=self._vad,
                    device_id=self.config.audio.input_device,
                    sample_rate=self.config.audio.sample_rate,
                    channels=self.config.audio.channels,
                )
            except ImportError as e:
                logger.warning(f"AudioCapture not available: {e}")
                self._capture = None

        # Start capture if available
        if self._capture:
            try:
                self._capture.start()
            except Exception as e:
                logger.error(f"Failed to start audio capture: {e}")
                self._capture = None

        # Start listening thread only if capture is available
        if self._capture and self._vad:
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
            logger.info("Started listening thread")
        else:
            logger.warning("Audio capture or VAD not available, listening disabled")
            if self._on_idle:
                self._on_idle()

    def _listen_loop(self) -> None:
        """Background loop: capture audio -> VAD -> accumulate speech -> STT."""
        assert self._ring_buffer is not None
        assert self._vad is not None
        assert self._stt is not None

        frame_size = self._vad.frame_size_bytes
        speech_frames: list[bytes] = []
        silence_frames = 0
        max_silence_frames = int(1.0 * 1000 / self._vad.frame_duration_ms)  # 1 second of silence
        min_speech_frames = int(0.3 * 1000 / self._vad.frame_duration_ms)  # 300ms minimum speech

        while not self._stop_listening.is_set():
            # Read available audio
            readable = self._ring_buffer.readable()
            if readable < frame_size:
                time.sleep(0.01)
                continue

            # Read one frame
            frame_data = self._ring_buffer.read(frame_size)
            if len(frame_data) != frame_size:
                continue

            # VAD
            try:
                is_speech = self._vad.is_speech(frame_data)
            except ValueError:
                continue

            if is_speech:
                speech_frames.append(frame_data)
                silence_frames = 0
            else:
                if speech_frames:
                    silence_frames += 1

            # Check if speech segment ended
            if speech_frames and silence_frames >= max_silence_frames:
                if len(speech_frames) >= min_speech_frames:
                    # Play listening end sound
                    if self._sound_manager:
                        self._sound_manager.play(SoundEvent.LISTENING_END)
                    # Process the accumulated speech
                    self._process_speech_segment(speech_frames)
                speech_frames = []
                silence_frames = 0

    def _process_speech_segment(self, speech_frames: list[bytes]) -> None:
        """Transcribe speech frames and process through pipeline."""
        assert self._stt is not None

        # Concatenate frames
        audio_data = b"".join(speech_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Get dropped file context
        dropped_files = self._get_and_clear_dropped_files()

        # Transcribe
        try:
            logger.info(f"Transcribing {len(audio_data)} bytes of audio")
            text = self._stt.transcribe(audio_array, self.config.audio.sample_rate)
            logger.info(f"Transcription: {text}")

            if text.strip():
                # Process through pipeline with dropped file context
                self.process_transcription(text, dropped_files)
            else:
                logger.info("Empty transcription, returning to listening")
                self._start_listening()  # Resume listening
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Play error sound
            if self._sound_manager:
                self._sound_manager.play(SoundEvent.ERROR)
            if self._notification_manager:
                self._notification_manager.notify_error(str(e), self.config.ui.verbosity)
            if self._on_idle:
                self._on_idle()

    def process_transcription(
        self, text: str, dropped_files: list[str] | None = None
    ) -> PipelineResult:
        """Process a transcription through the pipeline.

        Args:
            text: Transcribed text from STT.
            dropped_files: Optional list of file paths dropped on the island.

        Returns:
            PipelineResult with the response.
        """
        if not text.strip():
            return PipelineResult(success=False, error="Empty transcription")

        # Build context from dropped files
        context = ""
        if dropped_files:
            context = (
                "\n\n[Attached files:\n" + "\n".join(f"  - {f}" for f in dropped_files) + "\n]"
            )

        logger.info(f"Processing transcription: {text}{context}")

        assert self._router is not None, "Router not initialized"
        assert self._llm is not None, "LLM not initialized"
        assert self._agent is not None, "Agent not initialized"

        # Route: chat vs agent
        is_agent = self._router.classify(text)
        logger.info(f"Router decision: {'agent' if is_agent else 'chat'}")

        # Play thinking sound
        if self._sound_manager:
            self._sound_manager.play(SoundEvent.THINKING_START)

        if self._on_thinking:
            self._on_thinking()

        try:
            if is_agent:
                return self._handle_agent_task(text, dropped_files)
            else:
                return self._handle_chat(text, dropped_files)
        finally:
            if self._on_idle:
                self._on_idle()

    def _handle_chat(self, text: str, dropped_files: list[str] | None = None) -> PipelineResult:
        """Handle chat path - direct LLM response."""
        assert self._llm is not None

        # Build prompt with file context if available
        context = ""
        if dropped_files:
            context = (
                "\n\n[Attached files:\n" + "\n".join(f"  - {f}" for f in dropped_files) + "\n]"
            )

        prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {text}{context}\nAssistant:"
        response = self._llm.complete(prompt, temperature=0.7, max_tokens=512)

        # Synthesize and play speech
        self._speak(response.text)

        # Play task complete sound and notify
        if self._sound_manager:
            self._sound_manager.play(SoundEvent.TASK_COMPLETE)
        if self._notification_manager:
            self._notification_manager.notify_task_complete(
                response.text[:60], self.config.ui.verbosity
            )

        return PipelineResult(
            success=True,
            response=response.text,
            is_agent_task=False,
        )

    def _handle_agent_task(
        self, text: str, dropped_files: list[str] | None = None
    ) -> PipelineResult:
        """Handle agent path - dispatch to pi-mono."""
        assert self._llm is not None
        assert self._agent is not None

        # Build task prompt for pi-mono with file context
        context = ""
        if dropped_files:
            context = (
                "\n\n[Attached files:\n" + "\n".join(f"  - {f}" for f in dropped_files) + "\n]"
            )

        prompt = f"{AGENT_SYSTEM_PROMPT}\n\nUser request: {text}{context}"
        llm_response = self._llm.complete(prompt, temperature=0.3, max_tokens=1024)

        try:
            import json

            task_data = json.loads(llm_response.text)
            task = task_data.get("task", text)
            permissions = task_data.get("permissions", ["shell", "files"])
            confirmation_required = task_data.get("confirmation_required", True)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't output valid JSON
            task = text
            permissions = ["shell", "files"]
            confirmation_required = True

        # Check permissions against config
        allowed_permissions = self._get_allowed_permissions()
        permissions = [p for p in permissions if p in allowed_permissions]

        if not permissions:
            # Play error sound and notify
            if self._sound_manager:
                self._sound_manager.play(SoundEvent.ERROR)
            if self._notification_manager:
                self._notification_manager.notify_error(
                    "No permissions granted for this task", self.config.ui.verbosity
                )
            return PipelineResult(
                success=False,
                error="No permissions granted for this task",
                is_agent_task=True,
            )

        # Dispatch to agent
        result = self._agent.dispatch(
            task=task,
            permissions=permissions,
            _confirmation_required=confirmation_required,
        )

        if result.success:
            response_text = result.output or "Task completed successfully."
            # Play task complete sound and notify
            if self._sound_manager:
                self._sound_manager.play(SoundEvent.TASK_COMPLETE)
            if self._notification_manager:
                self._notification_manager.notify_task_complete(
                    response_text[:60], self.config.ui.verbosity
                )
        else:
            response_text = f"Task failed: {result.error or 'Unknown error'}"
            # Play error sound and notify
            if self._sound_manager:
                self._sound_manager.play(SoundEvent.ERROR)
            if self._notification_manager:
                self._notification_manager.notify_error(response_text, self.config.ui.verbosity)

        # Synthesize and play speech
        self._speak(response_text)

        return PipelineResult(
            success=result.success,
            response=response_text,
            is_agent_task=True,
            agent_result=result,
        )

    def _speak(self, text: str) -> None:
        """Synthesize and play speech using TTS provider."""
        if not text.strip():
            return

        # Play speaking start sound
        if self._sound_manager:
            self._sound_manager.play(SoundEvent.SPEAKING_START)

        if self._on_speaking:
            self._on_speaking(text)

        if self._tts:
            try:
                logger.info(f"Synthesizing speech: {text[:50]}...")
                audio_data = self._tts.synthesize(text)
                # In a real implementation, we'd play the audio here
                # For now, just log that we would play it
                logger.info(f"Speech synthesized ({len(audio_data)} bytes), would play now")
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}")
                # Play error sound
                if self._sound_manager:
                    self._sound_manager.play(SoundEvent.ERROR)

    def _get_allowed_permissions(self) -> list[str]:
        """Get permissions allowed by security config."""
        allowed = []
        perms = self.config.permissions

        if perms.allow_shell:
            allowed.append("shell")
        if perms.allow_files:
            allowed.append("files")
        if perms.allow_apps:
            allowed.append("apps")
        if perms.allow_web:
            allowed.append("web")
        if perms.allow_clipboard:
            allowed.append("clipboard")

        return allowed

    @property
    def is_running(self) -> bool:
        return self._running


def create_pipeline(config: Config) -> AssistantPipeline:
    """Factory function to create the assistant pipeline."""
    return AssistantPipeline(config)
