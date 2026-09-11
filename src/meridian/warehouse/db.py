"""SQLite connection management.

Foreign keys are off by default in SQLite -- a legacy compatibility decision --
so every connection this module hands out turns them on explicitly. A star
schema whose foreign keys are not enforced is a set of loosely associated
tables, and orphaned facts would silently vanish from the analysis.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.exceptions import WarehouseError
from ..common.logging import get_logger

log = get_logger(__name__)


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and sane pragmas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path.as_posix()}?mode=ro" if read_only else str(path)
    conn = sqlite3.connect(uri, uri=read_only, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # SQLite ships with foreign keys disabled for backwards compatibility.
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        # WAL survives an interrupted write better than the rollback journal,
        # and NORMAL synchronous is the usual pairing for an analytical build.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64 MB page cache
    return conn


@contextmanager
def session(path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """Connection context manager that commits on success, rolls back on error."""
    conn = connect(path, read_only=read_only)
    try:
        yield conn
        if not read_only:
            conn.commit()
    except Exception:
        if not read_only:
            conn.rollback()
        raise
    finally:
        conn.close()


def query(path: Path, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    """Run a read-only query and return the result as a DataFrame."""
    with session(path, read_only=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def query_conn(conn: sqlite3.Connection, sql: str,
               params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def execute_script(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a multi-statement script."""
    try:
        conn.executescript(sql)
    except sqlite3.Error as exc:
        raise WarehouseError(f"script failed: {exc}") from exc


def table_counts(path: Path) -> dict[str, int]:
    """Row count for every table in the database."""
    with session(path, read_only=True) as conn:
        names = [
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            n: conn.execute(f"SELECT COUNT(*) AS c FROM {n}").fetchone()["c"]
            for n in names
        }


def foreign_key_violations(path: Path) -> pd.DataFrame:
    """Rows whose foreign keys do not resolve.

    ``PRAGMA foreign_key_check`` reports violations that predate enforcement or
    slipped in while it was off, so this is run as a post-load assertion rather
    than trusted to the constraint alone.
    """
    with session(path, read_only=True) as conn:
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    return pd.DataFrame(
        [dict(r) for r in rows],
        columns=["table", "rowid", "parent", "fkid"],
    )


def integrity_check(path: Path) -> str:
    with session(path, read_only=True) as conn:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
