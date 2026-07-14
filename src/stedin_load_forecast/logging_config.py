"""Shared logging setup for the pipeline.

Call configure_logging() once, at the entry point (the CLI script or a
notebook's first cell) — library modules just use logging.getLogger(__name__).
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
