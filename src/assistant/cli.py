"""CLI entry point for the Desktop Assistant."""

from __future__ import annotations

from pathlib import Path

import click  # type: ignore[import-not-found]

from . import __version__
from .config import Config


@click.group()  # type: ignore[untyped-decorator]
@click.option(  # type: ignore[untyped-decorator]
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file.",
)
@click.option(  # type: ignore[untyped-decorator]
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (use -v, -vv, -vvv).",
)
@click.option(  # type: ignore[untyped-decorator]
    "--log-file",
    type=click.Path(),
    help="Path to log file.",
)
@click.version_option(version=__version__, prog_name="desktop-assistant")  # type: ignore[untyped-decorator]
@click.pass_context  # type: ignore[untyped-decorator]
def main(
    ctx: click.Context,
    config: str | None,
    verbose: int,
    log_file: str | None,
) -> None:
    """Desktop Assistant - Voice-first macOS assistant living at the camera notch."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None
    ctx.obj["verbose"] = verbose
    ctx.obj["log_file"] = Path(log_file) if log_file else None


@main.command()  # type: ignore[untyped-decorator]
@click.option(  # type: ignore[untyped-decorator]
    "--no-ui",
    is_flag=True,
    help="Run without UI (headless mode).",
)
@click.pass_context  # type: ignore[untyped-decorator]
def run(_ctx: click.Context, no_ui: bool) -> None:  # noqa: ARG001
    """Start the assistant."""
    click.echo("Starting Desktop Assistant...")
    click.echo("This feature is not yet implemented.")


@main.group()  # type: ignore[untyped-decorator]
def config() -> None:
    """Configuration management."""


@config.command("show")  # type: ignore[untyped-decorator]
@click.pass_context  # type: ignore[untyped-decorator]
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    config_path = ctx.obj.get("config_path")
    cfg = Config.load(config_path) if config_path else Config.load()
    click.echo(f"Security level: {cfg.security.level}")
    click.echo(f"Wake word: {cfg.wake.word}")
    click.echo(f"STT provider: {cfg.stt.provider}")
    click.echo(f"LLM provider: {cfg.llm.provider}")
    click.echo(f"Audio sample rate: {cfg.audio.sample_rate} Hz")


@config.command("edit")  # type: ignore[untyped-decorator]
@click.pass_context  # type: ignore[untyped-decorator]
def config_edit(ctx: click.Context) -> None:
    """Edit configuration in default editor."""
    config_path = ctx.obj.get("config_path")
    cfg = Config.load(config_path) if config_path else Config.get_default()
    cfg.save()
    click.echo("Configuration saved.")


@main.command()  # type: ignore[untyped-decorator]
def doctor() -> None:
    """Run health checks."""
    click.echo("Running health checks...")
    click.echo("This feature is not yet implemented.")
