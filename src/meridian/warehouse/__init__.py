"""Warehouse: SQLite star schema, loader, and analytical query layer."""

from .db import (
    connect,
    execute_script,
    foreign_key_violations,
    integrity_check,
    query,
    query_conn,
    session,
    table_counts,
)
from .loader import SCHEMA_PATH, build_warehouse
from .query_layer import (
    QUERIES,
    QUERY_DIR,
    QuerySpec,
    catalogue,
    load_sql,
    run_all,
    run_query,
)

__all__ = [
    "connect", "session", "query", "query_conn", "execute_script",
    "table_counts", "foreign_key_violations", "integrity_check",
    "build_warehouse", "SCHEMA_PATH",
    "QUERIES", "QUERY_DIR", "QuerySpec", "run_query", "run_all",
    "load_sql", "catalogue",
]
