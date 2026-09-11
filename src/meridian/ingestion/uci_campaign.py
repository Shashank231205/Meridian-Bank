"""UCI Bank Marketing -- 41,188 rows of real retail bank campaign outcomes.

This is the project's anchor in reality. It is genuine direct-marketing data
from a Portuguese retail bank: demographics, contact channel, campaign history,
macroeconomic context at time of contact, and the binary subscription outcome.
The entire campaign analytics module runs on it, so those findings are real
findings, not artefacts of a generator.

Three handling notes.

**Nested archive.** The download is a zip containing another zip:
``bank-additional.zip`` holds ``bank-additional/bank-additional-full.csv``.
Single-level extraction yields a zip file where a CSV was expected. macOS
resource forks (``__MACOSX/``) also appear as entries and must be filtered.

**Semicolon delimiter.** The CSVs use ``;``. Reading with the default comma
gives one wide column and a silently empty analysis.

**Target leakage in ``duration``.** Call duration is recorded *after* the call
and is near-collinear with the outcome -- a zero-second call cannot be a
subscription. The dataset's own documentation says it should be discarded for
any realistic predictive model. It is dropped by default here, in
:func:`load_campaign`, rather than left for a modeller to notice: including it
produces an AUC that looks excellent and means nothing. The raw column is still
available via ``drop_leaky=False`` so the leakage can be demonstrated.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from ..common.exceptions import SchemaError
from ..common.http import get_bytes
from ..common.logging import get_logger
from .base import SourceResult

log = get_logger(__name__)

# UCI moved to a versioned static path; this is the current location.
ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# Recorded only after the call concludes: knowing it means knowing the outcome.
LEAKY_COLUMNS = ("duration",)

# The richer 2014 release, with macro context columns joined in.
PREFERRED_MEMBER = "bank-additional/bank-additional-full.csv"
FALLBACK_MEMBER = "bank-full.csv"


def _inner_csv_bytes(blob: bytes, member_hint: str) -> tuple[bytes, str]:
    """Walk the nested archive and return the CSV bytes plus its member name."""
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        names = [n for n in outer.namelist() if not n.startswith("__MACOSX/")]
        log.debug("outer archive members: %s", names)

        # Direct hit: the CSV sits at the top level.
        for name in names:
            if name.endswith(".csv") and member_hint in name:
                return outer.read(name), name

        # Otherwise descend into each nested zip.
        for name in names:
            if not name.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                inner_names = [n for n in inner.namelist() if not n.startswith("__MACOSX/")]
                log.debug("inner archive %s members: %s", name, inner_names)
                for cand in inner_names:
                    if cand.endswith(".csv") and member_hint in cand:
                        return inner.read(cand), f"{name}!{cand}"

        # Last resort: the largest CSV anywhere in the archive.
        best: tuple[int, str, bytes] | None = None
        for name in names:
            if name.endswith(".csv"):
                data = outer.read(name)
                if best is None or len(data) > best[0]:
                    best = (len(data), name, data)
        if best is not None:
            return best[2], best[1]

    raise SchemaError(f"no CSV matching {member_hint!r} found in the UCI archive")


def load_campaign(
    *,
    member: str = PREFERRED_MEMBER,
    drop_leaky: bool = True,
    cache_dir: Path | None = None,
    ttl_days: int = 30,
) -> SourceResult:
    """Download, unpack and parse the campaign dataset.

    Args:
        member: which CSV to pull from the archive.
        drop_leaky: drop post-hoc columns that leak the target. Leave True for
            anything that will be modelled.
    """
    resp = get_bytes(ZIP_URL, cache_dir=cache_dir, ttl_days=ttl_days)
    csv_bytes, member_name = _inner_csv_bytes(resp.body, member)

    df = pd.read_csv(io.BytesIO(csv_bytes), sep=";", quotechar='"')
    if df.shape[1] == 1:
        raise SchemaError(
            "UCI CSV parsed to a single column -- delimiter is ';', not ','"
        )

    df.columns = [c.strip().strip('"').replace(".", "_") for c in df.columns]

    if "y" not in df.columns:
        raise SchemaError(f"UCI CSV has no target column 'y'; got {list(df.columns)}")
    df["subscribed"] = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)

    dropped: list[str] = []
    if drop_leaky:
        dropped = [c for c in LEAKY_COLUMNS if c in df.columns]
        df = df.drop(columns=dropped)

    conv = float(df["subscribed"].mean())
    log.info(
        "uci campaign: %d rows x %d cols from %s | conversion %.2f%% | dropped %s",
        len(df), df.shape[1], member_name, conv * 100, dropped or "nothing",
    )

    return SourceResult(
        name="uci_bank_marketing", df=df, source_url=resp.url,
        fetched_at=resp.fetched_at, sha256=resp.sha256, from_cache=resp.from_cache,
        notes={
            "member": member_name,
            "conversion_rate": conv,
            "n_positive": int(df["subscribed"].sum()),
            "leaky_columns_dropped": dropped,
            "leakage_note": (
                "'duration' is recorded after the call and is near-collinear with "
                "the outcome; retained models that include it are not predictive."
            ),
        },
    )


def campaign_summary(df: pd.DataFrame) -> dict[str, object]:
    """Headline figures used by the calibration and campaign analytics layers."""
    out: dict[str, object] = {
        "n_contacts": int(len(df)),
        "conversion_rate": float(df["subscribed"].mean()),
        "n_subscribed": int(df["subscribed"].sum()),
    }
    if "age" in df:
        out["age_mean"] = float(df["age"].mean())
        out["age_p25"], out["age_p75"] = (float(df["age"].quantile(q)) for q in (0.25, 0.75))
    if "job" in df:
        out["n_job_categories"] = int(df["job"].nunique())
    # The 2014 release carries real macro context; the 2012 one carries balance.
    for col in ("euribor3m", "emp_var_rate", "cons_price_idx", "balance"):
        if col in df:
            out[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return out
