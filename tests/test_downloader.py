import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from csi500_alpha.config import AppConfig, DateSettings
from csi500_alpha.data.client import FetchResult
from csi500_alpha.data.downloader import SmokeDownloader, build_download_plan
from csi500_alpha.errors import DataQualityError


class FakeTushareClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], bool]] = []
        self.cache_tags: list[str | None] = []

    def fetch(
        self,
        api_name: str,
        *,
        params: dict[str, Any],
        fields: tuple[str, ...],
        force: bool = False,
        cache_tag: str | None = None,
    ) -> FetchResult:
        self.calls.append((api_name, params, force))
        self.cache_tags.append(cache_tag)
        frame = self._frame(api_name, params, fields)
        return FetchResult(
            frame=frame,
            request_key=f"{api_name}-{len(self.calls)}",
            cache_hit=False,
        )

    @staticmethod
    def _frame(
        api_name: str,
        params: dict[str, Any],
        fields: tuple[str, ...],
    ) -> pd.DataFrame:
        instruments = ("000001.SZ", "000002.SZ")
        if api_name == "trade_cal":
            rows = [
                ("SSE", "20250102", 1, "20241231"),
                ("SSE", "20250103", 1, "20250102"),
            ]
        elif api_name == "index_weight":
            rows = [
                ("000905.SH", instruments[0], "20250102", 50.0),
                ("000905.SH", instruments[1], "20250102", 50.0),
            ]
        elif api_name == "index_daily":
            rows = [
                ("000905.SH", date, 100.0, 101.0, 99.0, 100.5, 100.0, 1.0, 1.0)
                for date in ("20250102", "20250103")
            ]
        elif api_name == "stock_basic" and params["list_status"] == "L":
            rows = [
                (code, code[:6], code, "主板", "SZSE", "L", "20200101", None)
                for code in instruments
            ]
        elif api_name == "index_classify":
            rows = [("801010.SI", "农林牧渔", "", "L1", "110000", "1", "SW2021")]
        elif api_name == "index_member_all" and "ts_code" in params and params["is_new"] == "Y":
            code = str(params["ts_code"])
            rows = [
                (
                    "801010.SI",
                    "农林牧渔",
                    "",
                    "",
                    "",
                    "",
                    code,
                    code,
                    "20200101",
                    None,
                    "Y",
                )
            ]
        elif api_name == "index_member_all" and params["is_new"] == "Y":
            rows = [
                (
                    "801010.SI",
                    "农林牧渔",
                    "",
                    "",
                    "",
                    "",
                    code,
                    code,
                    "20200101",
                    None,
                    "Y",
                )
                for code in instruments
            ]
        elif api_name == "daily":
            date = str(params["trade_date"])
            rows = [
                (code, date, 10.0, 10.5, 9.5, 10.1, 10.0, 100.0, 1_000.0) for code in instruments
            ]
        elif api_name == "adj_factor":
            date = str(params["trade_date"])
            rows = [(code, date, 1.0) for code in instruments]
        elif api_name == "stk_limit":
            date = str(params["trade_date"])
            rows = [(code, date, 11.0, 9.0) for code in instruments]
        elif api_name == "daily_basic":
            date = str(params["trade_date"])
            rows = [(code, date, 1.0, 1.0, 1.5, 1_000.0, 800.0) for code in instruments]
        elif api_name == "suspend_d" and params["trade_date"] == "20250103":
            rows = [(instruments[0], "20250103", "09:30-15:00", "S")]
        else:
            rows = []
        return pd.DataFrame(rows, columns=list(fields))


def test_downloader_materializes_all_optional_silver_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="fake"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=True,
            include_suspensions=True,
            include_instrument_master=True,
            include_industry=True,
            industry_taxonomies=("SW2021",),
        ),
    )
    client = FakeTushareClient()

    summary = SmokeDownloader(config, client).run(force=True)  # type: ignore[arg-type]

    expected = {
        "calendar",
        "benchmark_weights",
        "index_bars",
        "stock_bars",
        "adjustments",
        "price_limits",
        "daily_characteristics",
        "suspensions",
        "instrument_master",
        "industry_classification",
        "industry_membership",
    }
    assert set(summary.paths) == expected
    assert all(path.exists() for path in summary.paths.values())
    assert summary.rows["stock_bars"] == 4
    assert summary.rows["suspensions"] == 1
    assert summary.network_requests == len(client.calls)
    assert all(force for _, _, force in client.calls)
    assert [partition.partition_id for partition in summary.partitions] == ["2025"]
    assert summary.partitions[0].status == "downloaded"
    assert summary.progress_path.exists()
    assert summary.snapshot_path.exists()


def test_downloader_supplements_only_missing_industry_instruments(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="industry-gap"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=False,
            include_instrument_master=False,
            include_industry=True,
            industry_taxonomies=("SW2021",),
            supplement_industry_by_instrument=True,
        ),
    )

    class IncompleteBulkIndustryClient(FakeTushareClient):
        @staticmethod
        def _frame(
            api_name: str,
            params: dict[str, Any],
            fields: tuple[str, ...],
        ) -> pd.DataFrame:
            frame = FakeTushareClient._frame(api_name, params, fields)
            if api_name == "index_member_all" and "l1_code" in params and params["is_new"] == "Y":
                return frame[frame["ts_code"] == "000001.SZ"].reset_index(drop=True)
            return frame

    client = IncompleteBulkIndustryClient()
    summary = SmokeDownloader(config, client).run(force=True)  # type: ignore[arg-type]

    supplement_calls = [
        params
        for api_name, params, _ in client.calls
        if api_name == "index_member_all" and "ts_code" in params
    ]
    assert {params["ts_code"] for params in supplement_calls} == {"000002.SZ"}
    assert {params["is_new"] for params in supplement_calls} == {"Y", "N"}
    snapshot = json.loads(summary.snapshot_path.read_text(encoding="utf-8"))
    audit = snapshot["industry_supplement"]
    assert audit["requested_instruments"] == 1
    assert audit["baseline"]["minimum_coverage"] == 0.5
    assert audit["final"]["minimum_coverage"] == 1.0
    assert audit["remaining_missing_instruments"] == 0


def test_download_plan_partitions_full_history_and_applies_account_limit() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "full.yaml")

    plan = build_download_plan(config)

    assert [partition.partition_id for partition in plan.partitions] == [
        str(year) for year in range(2016, 2026)
    ]
    assert plan.daily_apis == (
        "daily",
        "adj_factor",
        "stk_limit",
        "daily_basic",
        "suspend_d",
    )
    assert plan.estimated_requests > 12_000
    assert plan.effective_min_request_interval_seconds >= 0.31


def test_downloader_reuses_validated_annual_partition(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="resume"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=True,
            include_suspensions=True,
            include_instrument_master=True,
            include_industry=True,
            industry_taxonomies=("SW2021",),
        ),
    )
    first_client = FakeTushareClient()
    first = SmokeDownloader(config, first_client).run(force=True)  # type: ignore[arg-type]
    assert first.partitions[0].status == "downloaded"

    second_client = FakeTushareClient()
    second = SmokeDownloader(config, second_client).run()  # type: ignore[arg-type]

    assert second.partitions[0].status == "reused"
    daily_apis = {"daily", "adj_factor", "stk_limit", "daily_basic", "suspend_d"}
    assert not any(api_name in daily_apis for api_name, _, _ in second_client.calls)
    assert second.rows == first.rows


def test_refresh_reference_does_not_force_daily_requests(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="refresh"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=True,
            include_suspensions=True,
            include_instrument_master=True,
            include_industry=True,
            industry_taxonomies=("SW2021",),
            reference_cache_tag="refresh-20250103",
        ),
    )
    client = FakeTushareClient()

    summary = SmokeDownloader(config, client).run(  # type: ignore[arg-type]
        refresh_reference=True
    )

    mutable_reference = {"stock_basic", "index_classify", "index_member_all"}
    daily_or_ranged = {
        "trade_cal",
        "index_weight",
        "index_daily",
        "daily",
        "adj_factor",
        "stk_limit",
        "daily_basic",
        "suspend_d",
    }
    assert all(
        force
        for api_name, _, force in client.calls
        if api_name in mutable_reference
    )
    assert all(
        not force
        for api_name, _, force in client.calls
        if api_name in daily_or_ranged
    )
    assert all(
        cache_tag == "refresh-20250103"
        for (api_name, _, _), cache_tag in zip(client.calls, client.cache_tags, strict=True)
        if api_name in mutable_reference
    )
    assert all(
        cache_tag is None
        for (api_name, _, _), cache_tag in zip(client.calls, client.cache_tags, strict=True)
        if api_name in daily_or_ranged
    )
    snapshot = json.loads(summary.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["request_policy"] == {
        "force_all": False,
        "refresh_mutable_reference": True,
        "reference_cache_tag": "refresh-20250103",
    }


def test_downloader_records_failed_partition(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="failed"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
    )

    class FailingClient(FakeTushareClient):
        def fetch(
            self,
            api_name: str,
            *,
            params: dict[str, Any],
            fields: tuple[str, ...],
            force: bool = False,
            cache_tag: str | None = None,
        ) -> FetchResult:
            if api_name == "daily" and params.get("trade_date") == "20250103":
                raise RuntimeError("synthetic partition failure")
            return super().fetch(
                api_name,
                params=params,
                fields=fields,
                force=force,
                cache_tag=cache_tag,
            )

    with pytest.raises(RuntimeError, match="synthetic partition failure"):
        SmokeDownloader(config, FailingClient()).run(force=True)  # type: ignore[arg-type]

    progress_path = config.paths.quality_root / "download-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["partitions"]["2025"]["status"] == "failed"
    assert progress["partitions"]["2025"]["error_type"] == "RuntimeError"


def test_stk_limit_at_documented_limit_requires_project_scope_completeness() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    downloader = SmokeDownloader(config, FakeTushareClient())  # type: ignore[arg-type]
    filler = [f"{number:06d}.BJ" for number in range(5800)]
    raw = pd.DataFrame(
        {
            "ts_code": [*filler[:-2], "000001.SZ", "600000.SH"],
            "trade_date": "20211229",
        }
    )
    scoped = raw[raw["ts_code"].isin({"000001.SZ", "600000.SH"})]

    downloader._verify_scoped_limit_response(
        raw=raw,
        scoped=scoped,
        required_codes={"000001.SZ", "600000.SH"},
        trade_date="20211229",
    )

    assert downloader.response_limit_events[-1]["resolution"] == "target_scope_complete"
    with pytest.raises(DataQualityError, match="omitted required project-scope codes"):
        downloader._verify_scoped_limit_response(
            raw=raw,
            scoped=scoped,
            required_codes={"000001.SZ", "600000.SH", "000002.SZ"},
            trade_date="20211229",
        )
