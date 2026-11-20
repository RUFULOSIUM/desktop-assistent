import tempfile

from click.testing import CliRunner  # type: ignore[import-not-found]


def test_cli_version() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_run_help() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower()


def test_cli_config_show() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0


def test_cli_config_edit() -> None:
    from assistant.cli import main

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        from pathlib import Path

        path = Path(f.name)
    try:
        result = runner.invoke(main, ["-c", str(path), "config", "edit"])
        assert result.exit_code == 0
    finally:
        path.unlink(missing_ok=True)


def test_cli_doctor() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0


def test_cli_verbose_flag() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["-v", "run", "--help"])
    assert result.exit_code == 0


def test_cli_log_file_option() -> None:
    from assistant.cli import main

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        from pathlib import Path

        path = Path(f.name)
    try:
        result = runner.invoke(main, ["--log-file", str(path), "run", "--help"])
        assert result.exit_code == 0
    finally:
        path.unlink(missing_ok=True)


def test_cli_config_option() -> None:
    from assistant.cli import main

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        from pathlib import Path

        path = Path(f.name)
    try:
        result = runner.invoke(main, ["-c", str(path), "config", "show"])
        assert result.exit_code == 0
    finally:
        path.unlink(missing_ok=True)


def test_cli_run_no_ui_flag() -> None:
    from assistant.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--no-ui"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output.lower()


def test_cli_config_show_output() -> None:
    from assistant.cli import main
    from assistant.config import Config

    runner = CliRunner()
    cfg = Config.get_default()
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        from pathlib import Path

        path = Path(f.name)
    cfg.save(path)
    try:
        runner = CliRunner()
        result = runner.invoke(main, ["-c", str(path), "config", "show"])
        assert result.exit_code == 0
        assert "standard" in result.output
        assert "HEY_ASSISTANT" in result.output
    finally:
        path.unlink(missing_ok=True)
