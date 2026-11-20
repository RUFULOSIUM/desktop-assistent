"""Agent dispatch to pi-mono CLI."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result of an agent dispatch."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int | None = None


class AgentDispatcher:
    """Dispatches tasks to pi-mono CLI."""

    def __init__(
        self,
        pi_binary: str = "pi",
        working_dir: Path | None = None,
        timeout: int = 300,
    ) -> None:
        self.pi_binary = pi_binary
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout
        self._pi_path = self._find_pi()

    def _find_pi(self) -> str | None:
        """Find the pi binary."""
        # Check if pi_binary is a full path
        if Path(self.pi_binary).is_absolute() and Path(self.pi_binary).exists():
            return self.pi_binary

        # Check in PATH
        found = shutil.which(self.pi_binary)
        if found:
            return found

        # Check common install locations
        common_paths = [
            Path.home() / ".local" / "bin" / "pi",
            Path("/usr/local/bin/pi"),
            Path("/opt/homebrew/bin/pi"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    def is_available(self) -> bool:
        """Check if pi-mono is available."""
        return self._pi_path is not None

    def dispatch(
        self,
        task: str,
        permissions: list[str] | None = None,
        _confirmation_required: bool = True,
        extra_args: list[str] | None = None,
    ) -> DispatchResult:
        """Dispatch a task to pi-mono.

        Args:
            task: Task description for the agent.
            permissions: List of permission scopes required.
            confirmation_required: Whether to ask for confirmation before executing.
            extra_args: Additional CLI arguments to pass to pi.

        Returns:
            DispatchResult with success status and output.
        """
        if not self._pi_path:
            return DispatchResult(
                success=False,
                output="",
                error="pi-mono not found. Install with: curl -fsSL https://pi.dev/install.sh | sh",
                exit_code=127,
            )

        # Build the pi command
        cmd = [self._pi_path]

        # Add JSON mode for structured output
        cmd.extend(["--json"])

        # Add working directory
        cmd.extend(["--cwd", str(self.working_dir)])

        # Add permissions as flags (pi-mono handles permissions internally)
        if permissions:
            for perm in permissions:
                cmd.extend(["--allow", perm])

        # Add extra args
        if extra_args:
            cmd.extend(extra_args)

        # Add the task
        cmd.append(task)

        logger.info(f"Dispatching to pi-mono: {' '.join(cmd[:3])} ... {task[:50]}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            if result.returncode == 0:
                # Try to parse JSON output
                try:
                    output_data = json.loads(result.stdout)
                    return DispatchResult(
                        success=True,
                        output=output_data.get("result", result.stdout),
                        exit_code=0,
                    )
                except json.JSONDecodeError:
                    return DispatchResult(
                        success=True,
                        output=result.stdout,
                        exit_code=0,
                    )
            else:
                return DispatchResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    exit_code=result.returncode,
                )

        except subprocess.TimeoutExpired:
            return DispatchResult(
                success=False,
                output="",
                error=f"pi-mono timed out after {self.timeout} seconds",
                exit_code=124,
            )
        except Exception as e:
            logger.exception("Dispatch failed")
            return DispatchResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
            )

    def dispatch_stream(
        self,
        task: str,
        permissions: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> Any:
        """Dispatch a task and stream output.

        Yields:
            Lines of output as they arrive.
        """
        if not self._pi_path:
            yield DispatchResult(
                success=False,
                output="",
                error="pi-mono not found",
                exit_code=127,
            )
            return

        cmd = [self.pi_binary]
        if permissions:
            for perm in permissions:
                cmd.extend(["--allow", perm])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(task)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.working_dir,
            )

            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                yield line.rstrip()

            process.wait()
            if process.returncode != 0:
                yield DispatchResult(
                    success=False,
                    output="",
                    error=f"pi-mono exited with code {process.returncode}",
                    exit_code=process.returncode,
                )

        except Exception as e:
            yield DispatchResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
            )


def create_dispatcher(config: Any | None = None) -> AgentDispatcher:
    """Factory function to create an agent dispatcher."""
    if config:
        return AgentDispatcher(
            pi_binary=getattr(config, "pi_binary", "pi"),
            working_dir=Path(getattr(config, "working_dir", ".")),
            timeout=getattr(config, "timeout", 300),
        )
    return AgentDispatcher()