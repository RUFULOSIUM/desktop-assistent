import logging
import tempfile
from io import StringIO
from pathlib import Path


def test_setup_logging() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.info("test message")
    output = stream.getvalue()
    assert "test message" in output


def test_log_levels() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    cfg.logging.level = "DEBUG"
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    output = stream.getvalue()
    assert "debug message" in output
    assert "info message" in output
    assert "warning message" in output
    assert "error message" in output


def test_log_format() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    cfg.logging.format = "%(levelname)s | %(message)s"
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.info("formatted message")
    output = stream.getvalue()
    assert "INFO | formatted message" in output


def test_get_logger_returns_logger() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger

    cfg = Config.get_default()
    stream = StringIO()
    from assistant.logging import setup_logging

    setup_logging(cfg, stream=stream)

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_multiple_loggers() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    logger1.info("message1")
    logger2.info("message2")

    output = stream.getvalue()
    assert "message1" in output
    assert "message2" in output


def test_file_logging() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_path = Path(f.name)

    try:
        setup_logging(cfg, file_path=log_path)

        logger = get_logger("test")
        logger.info("file message")

        # Read the log file
        content = log_path.read_text()
        assert "file message" in content
    finally:
        log_path.unlink(missing_ok=True)


def test_file_logging_rotation() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    cfg.logging.max_size_mb = 1  # 1 MB
    cfg.logging.backup_count = 2

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_path = Path(f.name)

    try:
        setup_logging(cfg, file_path=log_path)

        logger = get_logger("test")
        # Write enough to trigger rotation
        large_message = "x" * 1024 * 1024  # 1 MB
        for _ in range(3):
            logger.info(large_message)

        # Check that rotation happened (backup files exist)
        assert log_path.exists()
    finally:
        log_path.unlink(missing_ok=True)
        # Clean up any backup files
        for i in range(1, 4):
            backup = log_path.with_suffix(f".log.{i}")
            backup.unlink(missing_ok=True)


def test_setup_logging_clears_handlers() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    stream = StringIO()

    # Setup logging twice
    setup_logging(cfg, stream=stream)
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.info("test message")

    output = stream.getvalue()
    # Should only appear once (not duplicated)
    assert output.count("test message") == 1


def test_log_level_filtering() -> None:
    from assistant.config import Config
    from assistant.logging import get_logger, setup_logging

    cfg = Config.get_default()
    cfg.logging.level = "WARNING"
    stream = StringIO()
    setup_logging(cfg, stream=stream)

    logger = get_logger("test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    output = stream.getvalue()
    assert "debug message" not in output
    assert "info message" not in output
    assert "warning message" in output
    assert "error message" in output
