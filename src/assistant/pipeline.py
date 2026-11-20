"""Main pipeline connecting wake → STT → Router → LLM/Agent → UI."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .agent import AgentDispatcher, DispatchResult, create_dispatcher
from .config import Config
from .llm import LLMProvider, create_llm_provider
from .router import AGENT_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, RouterClassifier, create_classifier
from .stt import STTProvider, create_stt_provider
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
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_wake_detected: Callable[[], None] | None = None
        self._on_listening: Callable[[], None] | None = None
        self._on_thinking: Callable[[], None] | None = None
        self._on_speaking: Callable[[str], None] | None = None
        self._on_idle: Callable[[], None] | None = None

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        with self._lock:
            if self._running:
                return

            # Initialize STT
            self._stt = create_stt_provider(self.config.stt)
            self._stt.start()

            # Initialize LLM
            self._llm = create_llm_provider(self.config.llm)
            self._llm.start()

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

            if self._wake:
                self._wake.stop()
            if self._stt:
                self._stt.stop()
            if self._llm:
                self._llm.stop()

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

    def _on_wake_word(self, model_name: str) -> None:
        """Internal wake word callback."""
        logger.info(f"Wake word detected: {model_name}")
        if self._on_wake_detected:
            self._on_wake_detected()
        # Start listening
        self._start_listening()

    def _start_listening(self) -> None:
        """Start audio capture and STT."""
        if self._on_listening:
            self._on_listening()

        # In a real implementation, this would:
        # 1. Start audio capture
        # 2. Run VAD
        # 3. Stream to STT
        # 4. Get transcription
        # For now, we simulate with a callback
        pass

    def process_transcription(self, text: str) -> PipelineResult:
        """Process a transcription through the pipeline.

        Args:
            text: Transcribed text from STT.

        Returns:
            PipelineResult with the response.
        """
        if not text.strip():
            return PipelineResult(success=False, error="Empty transcription")

        logger.info(f"Processing transcription: {text}")

        assert self._router is not None, "Router not initialized"
        assert self._llm is not None, "LLM not initialized"
        assert self._agent is not None, "Agent not initialized"

        # Route: chat vs agent
        is_agent = self._router.classify(text)
        logger.info(f"Router decision: {'agent' if is_agent else 'chat'}")

        if self._on_thinking:
            self._on_thinking()

        try:
            if is_agent:
                return self._handle_agent_task(text)
            else:
                return self._handle_chat(text)
        finally:
            if self._on_idle:
                self._on_idle()

    def _handle_chat(self, text: str) -> PipelineResult:
        """Handle chat path - direct LLM response."""
        assert self._llm is not None
        prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"
        response = self._llm.complete(prompt, temperature=0.7, max_tokens=512)

        if self._on_speaking:
            self._on_speaking(response.text)

        return PipelineResult(
            success=True,
            response=response.text,
            is_agent_task=False,
        )

    def _handle_agent_task(self, text: str) -> PipelineResult:
        """Handle agent path - dispatch to pi-mono."""
        assert self._llm is not None
        assert self._agent is not None
        # Build task prompt for pi-mono
        prompt = f"{AGENT_SYSTEM_PROMPT}\n\nUser request: {text}"
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
        else:
            response_text = f"Task failed: {result.error or 'Unknown error'}"

        if self._on_speaking:
            self._on_speaking(response_text)

        return PipelineResult(
            success=result.success,
            response=response_text,
            is_agent_task=True,
            agent_result=result,
        )

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