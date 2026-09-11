"""Ingestion layer: real public data sources, with provenance."""

from .base import SourceResult, coerce_numeric
from .registry import REGISTRY, SourceSpec, ingest_all, manifest_table

__all__ = [
    "SourceResult", "coerce_numeric",
    "REGISTRY", "SourceSpec", "ingest_all", "manifest_table",
]
