"""Agent presets for common task patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentPreset:
    """Preset configuration for an agent task type."""

    name: str
    description: str
    permissions: list[str]
    confirmation_required: bool
    system_prompt_addendum: str = ""


PRESETS: dict[str, AgentPreset] = {
    "general": AgentPreset(
        name="general",
        description="General purpose task execution",
        permissions=["shell", "files", "apps", "web"],
        confirmation_required=True,
    ),
    "coding": AgentPreset(
        name="coding",
        description="Code writing, editing, and debugging tasks",
        permissions=["shell", "files", "web"],
        confirmation_required=False,
        system_prompt_addendum="Focus on writing clean, tested code. Run tests after changes.",
    ),
    "research": AgentPreset(
        name="research",
        description="Web research and information gathering",
        permissions=["web", "files"],
        confirmation_required=False,
        system_prompt_addendum="Browse multiple sources, cite URLs, provide summaries.",
    ),
    "file_ops": AgentPreset(
        name="file_ops",
        description="File and directory operations",
        permissions=["files", "shell"],
        confirmation_required=True,
        system_prompt_addendum="Be careful with deletions. Confirm before removing files.",
    ),
    "app_control": AgentPreset(
        name="app_control",
        description="Control macOS applications",
        permissions=["apps", "shell"],
        confirmation_required=True,
        system_prompt_addendum="Use AppleScript for app automation. Ask before closing apps.",
    ),
    "system_admin": AgentPreset(
        name="system_admin",
        description="System administration tasks",
        permissions=["shell", "files"],
        confirmation_required=True,
        system_prompt_addendum=(
            "High risk. Always confirm before executing. "
            "Avoid sudo unless necessary."
        ),
    ),
    "quick": AgentPreset(
        name="quick",
        description="Quick, low-risk tasks",
        permissions=["shell", "files", "web"],
        confirmation_required=False,
        system_prompt_addendum="Execute quickly without confirmation for simple tasks.",
    ),
}


def get_preset(name: str) -> AgentPreset | None:
    """Get a preset by name."""
    return PRESETS.get(name)


def list_presets() -> list[AgentPreset]:
    """List all available presets."""
    return list(PRESETS.values())