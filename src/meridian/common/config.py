"""Centralised runtime configuration, sourced from the environment.

Values come from ``.env`` (loaded by :func:`load_dotenv`) with the defaults in
:class:`Settings` as the fallback. The pipeline is designed to run end to end
with no keys at all, so every key-bearing field is optional and the code paths
that need one degrade rather than raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigError

# Repository root: this file is src/meridian/common/config.py, so up four.
ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a .env file into ``os.environ``.

    Written by hand rather than taking a python-dotenv dependency: the parsing
    needed here is a dozen lines, and requirements.txt is deliberately two
    packages long. Supports ``#`` comments, blank lines, ``export`` prefixes and
    quoted values. Missing file is not an error -- the pipeline runs keyless.
    """
    path = path or ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        # Strip matching surrounding quotes, then any trailing inline comment
        # on unquoted values only (a '#' inside quotes is data, not a comment).
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one pipeline run."""

    # Determinism. A fixed seed makes synthetic output byte-identical between
    # runs, which is what lets the generator be tested at all.
    seed: int = 20260911
    n_customers: int = 25_000

    # HTTP. Raw payloads are cached under data/raw/ so that after the first
    # fetch the whole pipeline reruns offline.
    http_cache_ttl_days: int = 7
    http_timeout_s: float = 30.0
    http_max_retries: int = 4

    # Narrative layer: "ollama" (local) | "groq" (cloud) | "none".
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:4b"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # Optional enrichment keys; empty means "skip that source".
    fred_api_key: str = ""
    data_gov_in_api_key: str = ""
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""

    log_level: str = "INFO"

    # Filesystem layout. Derived, not configurable -- the tree is part of the
    # repository contract and scripts/docs reference these paths by name.
    root: Path = field(default=ROOT, repr=False)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def warehouse_dir(self) -> Path:
        return self.data_dir / "warehouse"

    @property
    def db_path(self) -> Path:
        return self.warehouse_dir / "meridian.db"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def frontend_data_dir(self) -> Path:
        """Where the pipeline writes JSON for the Next.js dashboard to read."""
        return self.root / "frontend" / "public" / "data"

    def ensure_dirs(self) -> None:
        """Create every output directory this run may write to."""
        for p in (
            self.raw_dir, self.interim_dir, self.processed_dir,
            self.synthetic_dir, self.warehouse_dir, self.figures_dir,
            self.tables_dir, self.reports_dir, self.logs_dir,
            self.frontend_data_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def get_settings(*, reload_env: bool = True) -> Settings:
    """Build a :class:`Settings` from the current environment."""
    if reload_env:
        load_dotenv()
    return Settings(
        seed=_env_int("MERIDIAN_SEED", 20260911),
        n_customers=_env_int("MERIDIAN_N_CUSTOMERS", 25_000),
        http_cache_ttl_days=_env_int("HTTP_CACHE_TTL_DAYS", 7),
        llm_provider=_env_str("LLM_PROVIDER", "ollama").lower(),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=_env_str("OLLAMA_MODEL", "qwen3:4b"),
        groq_api_key=_env_str("GROQ_API_KEY", ""),
        groq_base_url=_env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        groq_model=_env_str("GROQ_MODEL", "llama-3.3-70b-versatile"),
        fred_api_key=_env_str("FRED_API_KEY", ""),
        data_gov_in_api_key=_env_str("DATA_GOV_IN_API_KEY", ""),
        alpha_vantage_api_key=_env_str("ALPHA_VANTAGE_API_KEY", ""),
        finnhub_api_key=_env_str("FINNHUB_API_KEY", ""),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
