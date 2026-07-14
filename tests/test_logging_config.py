import logging

from stedin_load_forecast.logging_config import configure_logging


def test_configure_logging_sets_root_level():
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG

    configure_logging(level=logging.INFO)
    assert logging.getLogger().level == logging.INFO
