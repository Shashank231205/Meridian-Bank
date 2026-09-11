"""HTTP client: custom User-Agent, exponential backoff, on-disk cache.

Three behaviours earn this module its existence.

**User-Agent.** Frankfurter (and several other public endpoints) reject the
default ``Python-urllib/3.x`` UA with HTTP 403. Verified directly: default UA
returns 403, a descriptive UA returns 200. Since every request in the pipeline
goes through :func:`get_bytes`, setting the UA once here fixes it everywhere.

**Caching.** Raw payloads are written to data/raw/ keyed by URL hash. After the
first successful fetch the entire pipeline reruns offline, which makes the build
reproducible and keeps us far inside every free tier's rate limit.

**Retries.** Public data APIs fail transiently. Retry on 429/5xx and transport
errors with exponential backoff plus jitter; never retry other 4xx, because a
404 will still be a 404 on the third attempt.

Only the standard library is used -- ``requests`` is not a dependency.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .exceptions import HTTPError
from .io import atomic_write_bytes, human_bytes
from .logging import get_logger

log = get_logger(__name__)

# A descriptive UA that identifies the project and a contact surface. Public
# data providers block anonymous default agents; they rarely block honest ones.
USER_AGENT = (
    "MeridianBank-Analytics/1.0 "
    "(+https://github.com/Shashank231205/Meridian-Bank; portfolio research project)"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "*/*",
    "Accept-Encoding": "gzip, identity",
    "Accept-Language": "en",
}

# Retry on rate limiting and server-side faults only.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class Response:
    """A fetched payload plus enough provenance to cite it later."""

    url: str
    status: int
    body: bytes
    headers: Mapping[str, str]
    from_cache: bool
    fetched_at: datetime

    @property
    def text(self) -> str:
        """Decode using the charset the server declared, defaulting to UTF-8."""
        ctype = self.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        return self.body.decode(charset, errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


def build_url(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Append query params, dropping any whose value is None or empty."""
    if not params:
        return url
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    if not clean:
        return url
    sep = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{sep}{urllib.parse.urlencode(clean, doseq=True)}"


def cache_key(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Stable cache filename component for a URL+params pair.

    Includes a readable host/path prefix so a human can tell what is in
    data/raw/ by listing it, plus a hash for uniqueness and filesystem safety.
    """
    full = build_url(url, params)
    digest = hashlib.sha256(full.encode()).hexdigest()[:16]
    parsed = urllib.parse.urlparse(full)
    stem = f"{parsed.netloc}{parsed.path}".replace("/", "_").replace(":", "_")
    stem = "".join(c for c in stem if c.isalnum() or c in "._-")[:60].strip("._-")
    return f"{stem or 'resource'}__{digest}"


def _cache_is_fresh(path: Path, ttl_days: int) -> bool:
    if not path.is_file():
        return False
    if ttl_days <= 0:
        return True  # 0 or negative means "never expire"
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=ttl_days)


def _meta_path(cache_path: Path) -> Path:
    """Sidecar path for a cache entry.

    Appends rather than using ``Path.with_suffix``, which replaces everything
    after the last dot. Cache keys embed dotted API paths (indicator codes such
    as ``FR.INR.LEND``), so with_suffix would map every indicator under
    ``FR.INR.*`` onto one shared sidecar and cross-contaminate their headers.
    """
    return cache_path.with_name(cache_path.name + ".meta.json")


def _maybe_gunzip(body: bytes, headers: Mapping[str, str]) -> bytes:
    """Transparently decompress gzip responses.

    urllib does not do this for us, and we advertise gzip to be polite to the
    providers. Some servers set the header without actually compressing, so the
    magic number is checked rather than trusted.
    """
    declared = headers.get("Content-Encoding", "").lower() == "gzip"
    if declared or body[:2] == GZIP_MAGIC:
        try:
            return gzip.decompress(body)
        except (OSError, EOFError):
            return body
    return body


def _read_cache(cache_path: Path, full_url: str, *, reason: str = "CACHE") -> Response:
    """Build a Response from an on-disk cache entry."""
    body = cache_path.read_bytes()
    meta_path = _meta_path(cache_path)
    headers: dict[str, str] = {}
    if meta_path.is_file():
        try:
            headers = json.loads(meta_path.read_text(encoding="utf-8")).get("headers", {})
        except (json.JSONDecodeError, AttributeError):
            headers = {}
    log.info("%s %s (%s)", reason, _short(full_url), human_bytes(len(body)))
    return Response(
        url=full_url, status=200, body=body, headers=headers, from_cache=True,
        fetched_at=datetime.fromtimestamp(cache_path.stat().st_mtime),
    )


def get_bytes(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
    timeout: float = 30.0,
    max_retries: int = 4,
    force_refresh: bool = False,
) -> Response:
    """Fetch a URL with caching and retries. Raises :class:`HTTPError`."""
    full_url = build_url(url, params)
    cache_path: Path | None = None

    if cache_dir is not None:
        cache_path = cache_dir / cache_key(url, params)
        if not force_refresh and _cache_is_fresh(cache_path, ttl_days):
            return _read_cache(cache_path, full_url)

    req_headers = {**DEFAULT_HEADERS, **(headers or {})}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(full_url, headers=req_headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                resp_headers = {k.title(): v for k, v in resp.headers.items()}
                body = _maybe_gunzip(raw, resp_headers)
                status = int(resp.status)

            log.info("GET %s -> %d (%s)", _short(full_url), status, human_bytes(len(body)))

            if cache_path is not None:
                atomic_write_bytes(cache_path, body)
                atomic_write_bytes(
                    _meta_path(cache_path),
                    json.dumps({
                        "url": full_url,
                        "status": status,
                        "headers": resp_headers,
                        "fetched_at": datetime.now().isoformat(timespec="seconds"),
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "bytes": len(body),
                    }, indent=2).encode(),
                )

            return Response(url=full_url, status=status, body=body,
                            headers=resp_headers, from_cache=False,
                            fetched_at=datetime.now())

        except urllib.error.HTTPError as exc:
            last_error = exc
            status = int(exc.code)
            if status not in RETRY_STATUS or attempt == max_retries:
                # Serve a stale cache entry rather than failing the build: an
                # expired payload beats no payload for a rerunnable pipeline.
                if cache_path is not None and cache_path.is_file():
                    log.warning("HTTP %d on %s; serving STALE cache", status, _short(full_url))
                    return _read_cache(cache_path, full_url, reason="STALE")
                raise HTTPError(f"HTTP {status} for {full_url}: {exc.reason}",
                                url=full_url, status=status) from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            _sleep_backoff(attempt, retry_after, f"HTTP {status}", full_url)

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == max_retries:
                if cache_path is not None and cache_path.is_file():
                    log.warning("transport error on %s; serving STALE cache", _short(full_url))
                    return _read_cache(cache_path, full_url, reason="STALE")
                raise HTTPError(f"transport failure for {full_url}: {exc}",
                                url=full_url, status=None) from exc
            _sleep_backoff(attempt, None, type(exc).__name__, full_url)

    raise HTTPError(f"retries exhausted for {full_url}: {last_error}", url=full_url)


def get_json(url: str, **kwargs: Any) -> Any:
    """Fetch and parse a JSON document."""
    return get_bytes(url, **kwargs).json()


def get_text(url: str, **kwargs: Any) -> str:
    """Fetch and decode a text document."""
    return get_bytes(url, **kwargs).text


def _sleep_backoff(attempt: int, retry_after: str | None, reason: str, url: str) -> None:
    """Exponential backoff with jitter, honouring Retry-After when sent.

    Jitter matters when several ingestion modules retry against the same host:
    without it they resynchronise and hammer in lockstep.
    """
    if retry_after:
        try:
            delay = min(float(retry_after), 60.0)
        except ValueError:
            delay = 2.0 ** attempt
    else:
        delay = 2.0 ** attempt
    delay += random.uniform(0, 0.5 * delay)
    log.warning("%s on %s; retry %d in %.1fs", reason, _short(url), attempt, delay)
    time.sleep(delay)


def _short(url: str, limit: int = 90) -> str:
    return url if len(url) <= limit else url[: limit - 3] + "..."
