import logging

from stedin_load_forecast.logging_config import configure_logging


def test_configure_logging_sets_root_level():
    configure_logging(level=logging.DEBUG, log_file=None)
    assert logging.getLogger().level == logging.DEBUG

    configure_logging(level=logging.INFO, log_file=None)
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_with_no_log_file_only_logs_to_console():
    configure_logging(level=logging.INFO, log_file=None)

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_log_file_captures_warning_and_above_but_not_info(tmp_path):
    log_file = tmp_path / "pipeline.log"
    configure_logging(level=logging.INFO, log_file=log_file, file_level=logging.WARNING)
    logger = logging.getLogger("test.logfile")

    logger.info("routine progress message")
    logger.warning("something looked off")
    logger.error("something failed")

    for handler in logging.getLogger().handlers:
        handler.flush()
    contents = log_file.read_text()

    assert "routine progress message" not in contents
    assert "something looked off" in contents
    assert "WARNING" in contents
    assert "something failed" in contents
    assert "ERROR" in contents


def test_log_file_directory_is_created_if_missing(tmp_path):
    log_file = tmp_path / "nested" / "pipeline.log"

    configure_logging(level=logging.INFO, log_file=log_file)

    assert log_file.parent.exists()
    assert log_file.exists()
