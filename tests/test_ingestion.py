"""Tests for the ingestion layer.

The unit tests here target the documented gotchas specifically: each one is a
failure mode that produces wrong numbers rather than an exception, so each gets
a test that would fail if the handling were removed.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from meridian.common.exceptions import SchemaError
from meridian.ingestion import fdic, uci_campaign, worldbank
from meridian.ingestion.base import SourceResult, coerce_numeric


class TestWorldBankEnvelope:
    """The response is [meta, rows] on success and [error] on failure."""

    def test_unwraps_the_two_element_success_shape(self):
        payload = [{"page": 1, "total": 2}, [{"date": "2022", "value": 8.5}]]
        assert worldbank._unwrap(payload, url="x") == [{"date": "2022", "value": 8.5}]

    def test_one_element_error_shape_raises_with_provider_message(self):
        payload = [{"message": [{"id": "120", "key": "Invalid value",
                                 "value": "The provided parameter is not valid"}]}]
        with pytest.raises(SchemaError, match="not valid"):
            worldbank._unwrap(payload, url="x")

    def test_null_rows_element_is_an_empty_series_not_a_crash(self):
        assert worldbank._unwrap([{"page": 1}, None], url="x") == []

    def test_non_list_payload_raises(self):
        with pytest.raises(SchemaError):
            worldbank._unwrap({"unexpected": "object"}, url="x")


class TestWorldBankNullYears:
    """Recent years exist as rows with value=null; taking rows[0] gives None."""

    @staticmethod
    def _series() -> pd.DataFrame:
        # Mirrors the real FR.INR.LEND shape: 2023-25 present but null.
        return pd.DataFrame({
            "indicator": ["FR.INR.LEND"] * 5,
            "indicator_label": ["Lending interest rate (%)"] * 5,
            "year": pd.array([2021, 2022, 2023, 2024, 2025], dtype="Int64"),
            "value": [8.698333, 8.567143, None, None, None],
        })

    def test_latest_value_skips_null_years(self):
        obs = worldbank.latest_value(self._series())
        assert obs is not None
        assert obs.year == 2022
        assert obs.value == pytest.approx(8.567143, abs=1e-5)

    def test_naive_most_recent_row_would_be_null(self):
        # Documents precisely what latest_value() is protecting against.
        df = self._series()
        assert pd.isna(df.loc[df["year"].idxmax(), "value"])

    def test_all_null_series_returns_none(self):
        df = self._series()
        df["value"] = None
        assert worldbank.latest_value(df) is None

    def test_empty_frame_returns_none(self):
        assert worldbank.latest_value(pd.DataFrame()) is None


class TestFdicUnits:
    """NIM is dollars (thousands); NIMY is percent. They are not interchangeable."""

    def test_nimy_is_classified_as_a_percentage(self):
        assert "NIMY" in fdic.PERCENT_FIELDS

    def test_nim_is_classified_as_dollars(self):
        assert "NIM" in fdic.DOLLAR_FIELDS

    def test_the_two_are_never_both_percentages(self):
        assert not (fdic.PERCENT_FIELDS & fdic.DOLLAR_FIELDS)

    def test_unit_guard_warns_on_dollar_scale_in_a_percent_column(self, caplog):
        # Real CERT 628 values: NIM 46,983,000 vs NIMY 3.02.
        df = pd.DataFrame({"NIMY": [46_983_000.0, 47_100_000.0]})
        with caplog.at_level("WARNING"):
            fdic.assert_units_sane(df)
        assert "dollar-scale" in caplog.text

    def test_unit_guard_is_quiet_on_plausible_percentages(self, caplog):
        df = pd.DataFrame({"NIMY": [3.02, 2.94, 3.11]})
        with caplog.at_level("WARNING"):
            fdic.assert_units_sane(df)
        assert "dollar-scale" not in caplog.text

    def test_rows_extractor_raises_on_missing_data_key(self):
        with pytest.raises(SchemaError, match="no 'data' key"):
            fdic._rows({"meta": {}}, url="x")

    def test_rows_extractor_unwraps_nested_data_objects(self):
        payload = {"data": [{"data": {"CERT": 628, "NIMY": 3.02}}]}
        assert fdic._rows(payload, url="x") == [{"CERT": 628, "NIMY": 3.02}]


class TestPeerIqr:
    def test_returns_ordered_quartiles(self):
        df = pd.DataFrame({
            "CERT": range(1, 9),
            "ROA": [0.5, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0],
            "report_date": pd.to_datetime(["2026-06-30"] * 8),
        })
        q1, med, q3 = fdic.peer_iqr(df, "ROA")
        assert q1 < med < q3

    def test_uses_only_the_latest_quarter_per_institution(self):
        # A bank with more history must not dominate the distribution.
        df = pd.DataFrame({
            "CERT": [1, 1, 1, 2],
            "ROA": [99.0, 99.0, 1.0, 1.2],
            "report_date": pd.to_datetime(
                ["2025-03-31", "2025-06-30", "2026-06-30", "2026-06-30"]
            ),
        })
        _, med, _ = fdic.peer_iqr(df, "ROA")
        assert med == pytest.approx(1.1)

    def test_missing_field_raises(self):
        with pytest.raises(SchemaError):
            fdic.peer_iqr(pd.DataFrame({"ROA": [1.0]}), "NOSUCH")

    def test_all_null_field_raises(self):
        with pytest.raises(SchemaError, match="entirely null"):
            fdic.peer_iqr(pd.DataFrame({"CERT": [1], "ROA": [None]}), "ROA")


class TestUciArchive:
    """The download is a zip inside a zip, with macOS resource forks present."""

    @staticmethod
    def _nested_zip(csv_text: str) -> bytes:
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("bank-additional/bank-additional-full.csv", csv_text)
            z.writestr("__MACOSX/._junk", "resource fork")
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as z:
            z.writestr("bank-additional.zip", inner.getvalue())
            z.writestr("__MACOSX/._outer", "resource fork")
        return outer.getvalue()

    def test_descends_into_the_inner_archive(self):
        blob = self._nested_zip("age;y\n30;no\n")
        data, name = uci_campaign._inner_csv_bytes(blob, uci_campaign.PREFERRED_MEMBER)
        assert b"age;y" in data
        assert "bank-additional-full.csv" in name

    def test_finds_a_top_level_csv_without_descending(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("bank-full.csv", "age;y\n30;no\n")
        data, name = uci_campaign._inner_csv_bytes(buf.getvalue(), "bank-full.csv")
        assert b"age;y" in data and name == "bank-full.csv"

    def test_raises_when_no_csv_exists(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("readme.txt", "nothing here")
        with pytest.raises(SchemaError, match="no CSV"):
            uci_campaign._inner_csv_bytes(buf.getvalue(), "bank-additional-full.csv")


class TestLeakage:
    def test_duration_is_registered_as_leaky(self):
        # Recorded after the call; a 0-second call cannot be a subscription.
        assert "duration" in uci_campaign.LEAKY_COLUMNS


class TestCoerceNumeric:
    def test_strips_thousands_separators(self):
        got = coerce_numeric(pd.Series(["1,234", "5,678"]))
        assert got.tolist() == [1234, 5678]

    def test_maps_placeholder_strings_to_nan(self):
        got = coerce_numeric(pd.Series(["", "N/A", "-", "null", "7"]))
        assert got.isna().sum() == 4 and got.iloc[-1] == 7

    def test_already_numeric_passes_through(self):
        s = pd.Series([1.0, 2.0])
        assert coerce_numeric(s).equals(s)


class TestSourceResult:
    def test_provenance_carries_url_time_and_shape(self):
        from datetime import datetime
        res = SourceResult(
            name="t", df=pd.DataFrame({"a": [1, 2]}), source_url="https://x.test",
            fetched_at=datetime(2026, 9, 11, 12, 0), sha256="abc",
        )
        p = res.provenance()
        assert p["n_rows"] == 2 and p["source_url"] == "https://x.test"
        assert p["fetched_at"].startswith("2026-09-11")

    def test_save_writes_csv_and_sidecar(self, tmp_path):
        from datetime import datetime
        res = SourceResult(name="t", df=pd.DataFrame({"a": [1]}),
                           source_url="u", fetched_at=datetime.now())
        res.save(tmp_path)
        assert (tmp_path / "t.csv").is_file()
        assert (tmp_path / "t.provenance.json").is_file()


@pytest.mark.network
class TestLiveSources:
    """Exercised against the real endpoints; excluded from CI."""

    def test_worldbank_lending_rate_latest_is_2022(self, tmp_path):
        res = worldbank.fetch_indicator("FR.INR.LEND", cache_dir=tmp_path)
        obs = worldbank.latest_value(res.df)
        assert obs is not None and obs.year >= 2022
        assert 4.0 < obs.value < 15.0  # a plausible Indian lending rate

    def test_uci_has_41188_rows_at_11_percent_conversion(self, tmp_path):
        res = uci_campaign.load_campaign(cache_dir=tmp_path)
        assert len(res.df) == 41_188
        assert res.notes["conversion_rate"] == pytest.approx(0.1127, abs=0.001)
        assert "duration" not in res.df.columns

    def test_fdic_nim_and_nimy_differ_by_orders_of_magnitude(self, tmp_path):
        res = fdic.fetch_financials(628, limit=4, cache_dir=tmp_path)
        row = res.df.dropna(subset=["NIM", "NIMY"]).iloc[-1]
        assert row["NIM"] > 1_000_000     # dollars, thousands
        assert 0 < row["NIMY"] < 20       # percent
