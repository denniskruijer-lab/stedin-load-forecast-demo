"""Shared logging setup for the pipeline.

Call configure_logging() once, at the entry point (the CLI script or a
notebook's first cell) — library modules never configure logging
themselves, they just log through `logging.getLogger(__name__)`. That
split (libraries log, only the entry point configures *how* those logs
are shown) is standard practice: it means this package behaves whether
it's run as a CLI, imported into someone else's script, or eventually
wrapped in an Airflow task — the caller decides what happens to the logs,
not the library.

Two outputs, two different jobs:

- **Console** (`StreamHandler`, no level of its own — inherits `level`,
  default INFO): the full run narrative, for watching a run happen.
- **File** (`FileHandler`, fixed at `file_level`, default WARNING): only
  problems. Kept separate from the console stream so that scanning for
  "did anything go wrong" doesn't mean grepping through hundreds of
  routine INFO lines — the file *is* the alert list.

Deliberately not a `RotatingFileHandler` or anything log-aggregation-aware:
this is a single, short-lived CLI run, not a long-running service, so
rotation/shipping would be complexity with no corresponding problem to
solve here. If this pipeline ever became a scheduled job, that's the
first thing to add — see the README's "Production next steps" section.
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
    """Set up console + (optional) file logging for the whole process.

    Safe to call more than once (e.g. from multiple tests in the same
    process) — `force=True` tells `logging.basicConfig` to discard any
    handlers a previous call installed, rather than silently no-op'ing.
    Without it, only the *first* call in a process would ever take
    effect, which is exactly the kind of bug that's invisible until two
    tests run in the same session and the second one's log level change
    appears to do nothing.

    Args:
        level: Minimum severity shown on the console. Default INFO shows
            the full run narrative; DEBUG would add step-by-step detail.
        log_file: Where to write the WARNING+ log file. Pass None to
            disable file output entirely — used by most tests, so running
            the test suite doesn't scatter log files through the repo.
        file_level: Minimum severity written to log_file. WARNING by
            default, so the file only ever contains things worth a
            human's attention.
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
