"""Logging setup: readable on a console, greppable in a file.

One call to :func:`setup_logging` at process start configures both sinks. A
run-scoped file under logs/ means a failed build leaves an artefact you can read
afterwards instead of scrollback you have already lost.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FMT = "%H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", *, logs_dir: Path | None = None,
                  run_name: str = "pipeline") -> Path | None:
    """Configure root logging once. Returns the log file path, if any.

    Repeat calls are no-ops, so library modules may call this defensively
    without stacking duplicate handlers onto the root logger.
    """
    global _configured
    if _configured:
        return None

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    log_path: Path | None = None
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{run_name}_{stamp}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # file always captures DEBUG
        fh.setFormatter(logging.Formatter(_FILE_FMT))
        root.addHandler(fh)

    # urllib is noisy at DEBUG and says nothing useful here.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
