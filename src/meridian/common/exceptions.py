"""Exception hierarchy for the Meridian pipeline.

Every failure mode gets a named type so callers can distinguish "the network is
down" (retry) from "the payload was not what the API promised" (the contract
changed) from "synthetic data drifted outside its calibration envelope" (fail
the build loudly).
"""

from __future__ import annotations


class MeridianError(Exception):
    """Base class for every error raised by this package."""


class IngestionError(MeridianError):
    """Raised when an upstream source cannot be retrieved or parsed."""


class HTTPError(IngestionError):
    """A non-retryable HTTP failure, or retries exhausted.

    ``status`` is None for transport-level failures (DNS, connection reset,
    timeout) where no HTTP response was ever received.
    """

    def __init__(self, message: str, *, url: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


class SchemaError(IngestionError):
    """An upstream payload did not match the shape this code expects.

    Raised in preference to letting a KeyError escape from inside a parser: the
    useful information is "World Bank changed its envelope", not "list index out
    of range".
    """


class CalibrationError(MeridianError):
    """Synthetic output fell outside the envelope derived from real data.

    A build gate, not a warning. If generated portfolio ratios sit outside the
    real FDIC peer IQR the numbers are not defensible and the build must stop.
    """


class ValidationError(MeridianError):
    """A data-quality rule failed at a severity configured to halt the build."""


class WarehouseError(MeridianError):
    """Schema, load, or query execution failure in the SQLite warehouse."""


class LLMError(MeridianError):
    """The narrative provider was unreachable or returned an unusable response.

    Never fatal: narrative is enrichment, and reporting degrades to template
    text when this is raised.
    """


class ConfigError(MeridianError):
    """Environment or settings are missing/invalid."""
