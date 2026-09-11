"""Tests for the shared infrastructure layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meridian.common import http
from meridian.common.config import Settings, load_dotenv
from meridian.common.io import atomic_write_text, read_json, write_json


class TestBuildUrl:
    def test_no_params_is_unchanged(self):
        assert http.build_url("https://x.test/a") == "https://x.test/a"

    def test_params_are_appended(self):
        assert http.build_url("https://x.test/a", {"b": 1}) == "https://x.test/a?b=1"

    def test_existing_query_uses_ampersand(self):
        got = http.build_url("https://x.test/a?x=1", {"y": 2})
        assert got == "https://x.test/a?x=1&y=2"

    def test_none_and_empty_values_are_dropped(self):
        # Optional API keys are passed as "" when absent; they must not be sent.
        got = http.build_url("https://x.test/a", {"key": "", "other": None, "keep": 3})
        assert got == "https://x.test/a?keep=3"


class TestCacheKey:
    def test_is_stable(self):
        a = http.cache_key("https://x.test/a", {"p": 1})
        b = http.cache_key("https://x.test/a", {"p": 1})
        assert a == b

    def test_differs_by_params(self):
        assert http.cache_key("https://x.test/a", {"p": 1}) != http.cache_key(
            "https://x.test/a", {"p": 2}
        )

    def test_is_filesystem_safe(self):
        key = http.cache_key("https://api.x.test/v1/a/b?q=1&r=2")
        assert not set(key) & set(r'/\:*?"<>|')


class TestUserAgent:
    def test_is_not_the_urllib_default(self):
        # The whole reason this module exists: Frankfurter 403s the default UA.
        assert "urllib" not in http.USER_AGENT.lower()
        assert "Meridian" in http.USER_AGENT

    def test_is_sent_by_default(self):
        assert http.DEFAULT_HEADERS["User-Agent"] == http.USER_AGENT


class TestGunzip:
    def test_passes_through_plain_bytes(self):
        assert http._maybe_gunzip(b"plain", {}) == b"plain"

    def test_decompresses_by_magic_number_without_header(self):
        import gzip
        blob = gzip.compress(b"hello")
        assert http._maybe_gunzip(blob, {}) == b"hello"

    def test_survives_a_lying_content_encoding_header(self):
        # Some servers declare gzip and send plain bytes; must not explode.
        assert http._maybe_gunzip(b"notgzip", {"Content-Encoding": "gzip"}) == b"notgzip"


class TestRetryPolicy:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_statuses_retry(self, status):
        assert status in http.RETRY_STATUS

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_do_not_retry(self, status):
        # A 404 will still be a 404 on the third attempt.
        assert status not in http.RETRY_STATUS


class TestIO:
    def test_json_round_trip(self, tmp_path: Path):
        obj = {"a": 1, "b": [1, 2], "c": "x"}
        p = write_json(tmp_path / "o.json", obj)
        assert read_json(p) == obj

    def test_numpy_scalars_are_coerced(self, tmp_path: Path):
        import numpy as np
        p = write_json(tmp_path / "n.json", {"i": np.int64(5), "f": np.float64(1.5)})
        assert read_json(p) == {"i": 5, "f": 1.5}

    def test_write_is_atomic_and_leaves_no_temp_files(self, tmp_path: Path):
        atomic_write_text(tmp_path / "a.txt", "content")
        assert (tmp_path / "a.txt").read_text() == "content"
        assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]

    def test_overwrite_replaces_content(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        atomic_write_text(p, "first")
        atomic_write_text(p, "second")
        assert p.read_text() == "second"


class TestDotenv:
    def test_parses_comments_quotes_and_export(self, tmp_path: Path, monkeypatch):
        (tmp_path / ".env").write_text(
            "\n".join([
                "# a comment",
                "",
                "PLAIN=value",
                'QUOTED="quoted value"',
                "export EXPORTED=e",
                "INLINE=v # trailing comment",
                "EMPTY=",
            ]),
            encoding="utf-8",
        )
        monkeypatch.delenv("PLAIN", raising=False)
        got = load_dotenv(tmp_path / ".env")
        assert got["PLAIN"] == "value"
        assert got["QUOTED"] == "quoted value"
        assert got["EXPORTED"] == "e"
        assert got["INLINE"] == "v"
        assert got["EMPTY"] == ""

    def test_missing_file_is_not_an_error(self, tmp_path: Path):
        # The pipeline is designed to run with no keys at all.
        assert load_dotenv(tmp_path / "nope.env") == {}


class TestSettings:
    def test_paths_hang_off_root(self, tmp_path: Path):
        s = Settings(root=tmp_path)
        assert s.db_path == tmp_path / "data" / "warehouse" / "meridian.db"
        assert s.frontend_data_dir == tmp_path / "frontend" / "public" / "data"

    def test_ensure_dirs_creates_the_tree(self, tmp_path: Path):
        s = Settings(root=tmp_path)
        s.ensure_dirs()
        assert s.raw_dir.is_dir() and s.tables_dir.is_dir() and s.frontend_data_dir.is_dir()

    def test_seed_default_is_fixed(self):
        # Byte-identical synthetic data across runs depends on this.
        assert Settings().seed == 20260911


class TestCacheBehaviour:
    def test_fresh_cache_is_served_without_network(self, tmp_path: Path):
        url = "https://x.test/resource"
        key = http.cache_key(url)
        (tmp_path / key).write_bytes(b'{"cached": true}')
        (tmp_path / key).with_suffix(".meta.json").write_text(
            json.dumps({"headers": {"Content-Type": "application/json"}})
        )
        # No network is reachable at x.test; a hit proves nothing was requested.
        r = http.get_bytes(url, cache_dir=tmp_path, ttl_days=7)
        assert r.from_cache and r.json() == {"cached": True}


@pytest.mark.network
class TestLiveFrankfurter:
    """Phase 0 exit criterion, verified against the live endpoint."""

    def test_custom_ua_gets_200(self, tmp_path: Path):
        r = http.get_bytes(
            "https://api.frankfurter.app/latest",
            params={"from": "EUR", "to": "INR"},
            cache_dir=tmp_path,
        )
        assert r.status == 200
        assert "INR" in r.json()["rates"]
