"""Shared logging setup for the pipeline.

Call configure_logging() once, at the entry point (the CLI script or a
notebook's first cell) — library modules just use logging.getLogger(__name__).
"""

import logging
from pathlib import Path

DEFAULT_LOG_FILE = Path("logs/pipeline.log")

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = DEFAULT_LOG_FILE,
    file_level: int = logging.WARNING,
) -> None:
    """Configure console + (optional) file logging.

    The console shows the full run narrative at `level` (default INFO).
    The log file only captures `file_level` and above (default WARNING) —
    it's meant as an at-a-glance record of problems (WARNING/ERROR/
    CRITICAL), not a full run transcript. Pass log_file=None to disable
    file logging (e.g. in tests, to avoid writing into the repo).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
