"""Filesystem helpers: atomic writes, JSON/CSV round-trips, hashing.

Atomicity matters more than it looks. The pipeline writes JSON that the Next.js
dashboard reads; a half-written file from an interrupted run would render as a
broken page rather than an obvious failure. Writing to a temp file in the same
directory and then replacing is cheap insurance.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .logging import get_logger

log = get_logger(__name__)


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write bytes so readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX and on Windows for same volume
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def write_json(path: Path, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Serialise to JSON, coercing numpy/pandas scalars that json rejects."""
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, default=_json_default,
                      ensure_ascii=False)
    log.debug("write_json %s (%d bytes)", path.name, len(text))
    return atomic_write_text(path, text)


def _json_default(o: Any) -> Any:
    """Fallback encoder for numpy scalars, timestamps and Paths."""
    # numpy scalars expose .item(); pandas NaT/Timestamp expose isoformat().
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return o.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if o is pd.NaT:
        return None
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame, **kwargs: Any) -> Path:
    """Write a DataFrame to CSV atomically, index suppressed by default."""
    kwargs.setdefault("index", False)
    csv_text = df.to_csv(**kwargs)
    log.debug("write_csv %s (%d rows x %d cols)", path.name, len(df), df.shape[1])
    return atomic_write_text(path, csv_text)


def write_parquet_or_csv(path_no_ext: Path, df: pd.DataFrame) -> Path:
    """Prefer Parquet, fall back to CSV when no engine is installed.

    Parquet preserves dtypes across pipeline stages, which matters for the
    multi-million-row transaction fact. But pyarrow is not in requirements.txt,
    so this degrades rather than demanding it.
    """
    try:
        target = path_no_ext.with_suffix(".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)
        return target
    except (ImportError, ValueError) as exc:
        log.debug("parquet unavailable (%s); falling back to csv", exc)
        return write_csv(path_no_ext.with_suffix(".csv"), df)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Streaming hash, for provenance records on large raw downloads."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
