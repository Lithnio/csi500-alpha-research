from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from csi500_alpha.config import AppConfig, DateSettings
from csi500_alpha.data.client import FetchResult
from csi500_alpha.data.financial import (
    DISCLOSURE_FIELDS,
    FINANCIAL_API_FIELDS,
    FinancialDownloader,
    FinancialDownloadSpec,
    build_financial_download_plan,
    normalize_disclosure_schedule,
    normalize_financial_statement,
    select_financial_versions_asof,
)
from csi500_alpha.errors import DataQualityError


class FakeFinancialClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    def fetch(
        self,
        api_name: str,
        *,
        params: dict[str, Any],
        fields: tuple[str, ...],
        force: bool = False,
        cache_tag: str | None = None,
    ) -> FetchResult:
        del cache_tag
        self.calls.append((api_name, params, force))
        if api_name == "disclosure_date":
            rows = [
                {
                    "ts_code": instrument,
                    "ann_date": "20241201",
                    "end_date": str(params["end_date"]),
                    "pre_date": "20250102",
                    "actual_date": "20250102",
                    "modify_date": "20241215",
                }
                for instrument in ("000001.SZ", "000002.SZ")
            ]
        else:
            row: dict[str, Any] = {field: 1.0 for field in fields}
            row.update(
                {
                    "ts_code": str(params["ts_code"]),
                    "ann_date": "20250102",
                    "end_date": "20240930",
                    "report_type": "1",
                    "comp_type": "1",
                    "end_type": "4",
                    "update_flag": "0",
                }
            )
            if "f_ann_date" in fields:
                row["f_ann_date"] = "20250102"
            rows = [row]
        return FetchResult(
            frame=pd.DataFrame(rows, columns=list(fields)),
            request_key=f"{api_name}-{len(self.calls)}",
            cache_hit=False,
        )


def _spec(tmp_path: Path, *, api_names: tuple[str, ...] | None = None) -> FinancialDownloadSpec:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    base = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="financial-test"),
        dates=DateSettings(
            raw_start="20250101",
            backtest_start="20250101",
            end="20250106",
        ),
    )
    return FinancialDownloadSpec(
        config_path=root / "configs" / "financial_smoke.yaml",
        base_config=base,
        output_subdirectory="financial",
        announcement_start="20250101",
        announcement_end="20250103",
        report_period_start="20240930",
        report_period_end="20240930",
        api_names=api_names or tuple(FINANCIAL_API_FIELDS),
        report_type="1",
        instruments=(),
        instrument_limit=None,
        response_row_limit=100,
        include_disclosure_schedule=True,
        disclosure_response_row_limit=6000,
        availability_lag_open_days=1,
        cache_tag="unit-test",
    )


def _write_market_inputs(spec: FinancialDownloadSpec) -> None:
    spec.base_config.paths.silver_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "snapshot_date": ["20250102", "20250102"],
            "instrument": ["000001.SZ", "000002.SZ"],
            "weight": [0.5, 0.5],
        }
    ).to_parquet(
        spec.base_config.paths.silver_root / "benchmark_weights.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "trade_date": ["20250101", "20250102", "20250103", "20250106"],
            "is_open": [0, 1, 1, 1],
        }
    ).to_parquet(spec.base_config.paths.silver_root / "calendar.parquet", index=False)


def test_financial_downloader_materializes_point_in_time_tables(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write_market_inputs(spec)
    client = FakeFinancialClient()

    plan = build_financial_download_plan(spec)
    summary = FinancialDownloader(spec, client).run(force=True)  # type: ignore[arg-type]

    assert plan.base_requests == 9
    assert summary.network_requests == 9
    assert all(force for _, _, force in client.calls)
    statement_calls = [call for call in client.calls if call[0] != "disclosure_date"]
    assert all(
        params["start_date"] == spec.announcement_start
        and params["end_date"] == spec.announcement_end
        for _, params, _ in statement_calls
    )
    assert set(summary.paths) == {
        "fina_indicator",
        "income",
        "balancesheet",
        "cashflow",
        "disclosure_schedule",
        "availability_index",
    }
    income = pd.read_parquet(summary.paths["income"])
    assert income["available_date"].eq("20250103").all()
    assert income["source_announcement_date"].eq("20250102").all()
    assert income["version_sequence"].eq(1).all()
    availability = pd.read_parquet(summary.paths["availability_index"])
    assert len(availability) == 8
    quality = json.loads(summary.quality_path.read_text(encoding="utf-8"))
    assert quality["status"] == "success"
    assert quality["validation"]["passed"]
    assert summary.manifest_path.exists()
    assert summary.progress_path.exists()


def test_revisions_become_visible_only_after_their_next_open_day() -> None:
    fields = FINANCIAL_API_FIELDS["income"]
    rows: list[dict[str, Any]] = []
    for announcement_date, update_flag, net_income in (
        ("20250102", "0", 10.0),
        ("20250103", "1", 12.0),
    ):
        row: dict[str, Any] = {field: 1.0 for field in fields}
        row.update(
            {
                "ts_code": "000001.SZ",
                "ann_date": announcement_date,
                "f_ann_date": announcement_date,
                "end_date": "20240930",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "4",
                "n_income": net_income,
                "update_flag": update_flag,
            }
        )
        rows.append(row)
    raw = pd.DataFrame(rows, columns=list(fields))
    schedule = normalize_disclosure_schedule(
        [
            pd.DataFrame(
                [
                    (
                        "000001.SZ",
                        "20241201",
                        "20240930",
                        "20250102",
                        "20250102",
                        "20241215",
                    )
                ],
                columns=list(DISCLOSURE_FIELDS),
            )
        ],
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
    )

    normalized = normalize_financial_statement(
        "income",
        [raw],
        disclosure_schedule=schedule,
        open_dates=("20250102", "20250103", "20250106"),
        availability_lag_open_days=1,
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
    )

    assert normalized["available_date"].tolist() == ["20250103", "20250106"]
    assert normalized["version_sequence"].tolist() == [1, 2]
    before_revision = select_financial_versions_asof(normalized, "20250103")
    after_revision = select_financial_versions_asof(normalized, "20250106")
    assert before_revision.iloc[0]["n_income"] == 10.0
    assert after_revision.iloc[0]["n_income"] == 12.0
    assert after_revision.iloc[0]["decision_date"] == "20250106"


def test_disclosure_schedule_preserves_modified_date_history() -> None:
    schedule = normalize_disclosure_schedule(
        [
            pd.DataFrame(
                [
                    (
                        "000001.SZ",
                        "20150401",
                        "20150331",
                        "20150508",
                        "20150630",
                        "20150508, 20150627,20150630",
                    )
                ],
                columns=list(DISCLOSURE_FIELDS),
            )
        ],
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
    )

    assert schedule.iloc[0]["schedule_modified_date_history"] == (
        "20150508,20150627,20150630"
    )
    assert schedule.iloc[0]["schedule_modified_date"] == "20150630"


def test_disclosure_schedule_rejects_invalid_modified_date_history() -> None:
    raw = pd.DataFrame(
        [
            (
                "000001.SZ",
                "20150401",
                "20150331",
                "20150508",
                "20150630",
                "20150508,not-a-date",
            )
        ],
        columns=list(DISCLOSURE_FIELDS),
    )

    with pytest.raises(DataQualityError, match="comma-delimited YYYYMMDD"):
        normalize_disclosure_schedule(
            [raw],
            instruments=("000001.SZ",),
            materialized_at="2026-08-27T00:00:00+00:00",
        )


def test_disclosure_cross_check_can_only_delay_availability() -> None:
    fields = FINANCIAL_API_FIELDS["fina_indicator"]
    row: dict[str, Any] = {field: 1.0 for field in fields}
    row.update(
        {
            "ts_code": "000001.SZ",
            "ann_date": "20250102",
            "end_date": "20240930",
            "update_flag": "0",
        }
    )
    schedule = normalize_disclosure_schedule(
        [
            pd.DataFrame(
                [
                    (
                        "000001.SZ",
                        "20241201",
                        "20240930",
                        "20250103",
                        "20250103",
                        "20241215",
                    )
                ],
                columns=list(DISCLOSURE_FIELDS),
            )
        ],
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
    )

    normalized = normalize_financial_statement(
        "fina_indicator",
        [pd.DataFrame([row], columns=list(fields))],
        disclosure_schedule=schedule,
        open_dates=("20250102", "20250103", "20250106"),
        availability_lag_open_days=1,
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
    )

    assert normalized.iloc[0]["source_announcement_date"] == "20250103"
    assert normalized.iloc[0]["available_date"] == "20250106"


def test_announcement_cutoff_removes_later_versions_without_misaligning_dates() -> None:
    fields = FINANCIAL_API_FIELDS["income"]
    rows: list[dict[str, Any]] = []
    for announcement_date in ("20241231", "20250102", "20260102"):
        row: dict[str, Any] = {field: 1.0 for field in fields}
        row.update(
            {
                "ts_code": "000001.SZ",
                "ann_date": announcement_date,
                "f_ann_date": announcement_date,
                "end_date": "20240930",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "4",
                "update_flag": "0",
            }
        )
        rows.append(row)

    normalized = normalize_financial_statement(
        "income",
        [pd.DataFrame(rows, columns=list(fields))],
        disclosure_schedule=pd.DataFrame(),
        open_dates=("20250102", "20250103", "20250106"),
        availability_lag_open_days=1,
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
        announcement_start="20250101",
        announcement_end="20251231",
    )

    assert normalized["announcement_date"].tolist() == ["20250102"]
    assert normalized["available_date"].tolist() == ["20250103"]
    assert normalized.attrs["normalization"] == {
        "source_rows": 3,
        "rows_before_announcement_window": 1,
        "rows_after_announcement_window": 1,
        "rows_before_report_period_window": 0,
        "rows_after_report_period_window": 0,
        "materialized_rows": 1,
        "announcement_start": "20250101",
        "announcement_end": "20251231",
        "report_period_start": None,
        "report_period_end": None,
    }


def test_report_period_window_is_independent_from_announcement_window() -> None:
    fields = FINANCIAL_API_FIELDS["income"]
    rows: list[dict[str, Any]] = []
    for report_period in ("20240630", "20240930", "20241231"):
        row: dict[str, Any] = {field: 1.0 for field in fields}
        row.update(
            {
                "ts_code": "000001.SZ",
                "ann_date": "20250102",
                "f_ann_date": "20250102",
                "end_date": report_period,
                "report_type": "1",
                "comp_type": "1",
                "end_type": "4",
                "update_flag": "0",
            }
        )
        rows.append(row)

    normalized = normalize_financial_statement(
        "income",
        [pd.DataFrame(rows, columns=list(fields))],
        disclosure_schedule=pd.DataFrame(),
        open_dates=("20250102", "20250103"),
        availability_lag_open_days=1,
        instruments=("000001.SZ",),
        materialized_at="2026-08-27T00:00:00+00:00",
        announcement_start="20250101",
        announcement_end="20250131",
        report_period_start="20240930",
        report_period_end="20240930",
    )

    assert normalized["report_period"].tolist() == ["20240930"]
    assert normalized.attrs["normalization"]["rows_before_report_period_window"] == 1
    assert normalized.attrs["normalization"]["rows_after_report_period_window"] == 1


class SaturatingFinancialClient(FakeFinancialClient):
    def fetch(
        self,
        api_name: str,
        *,
        params: dict[str, Any],
        fields: tuple[str, ...],
        force: bool = False,
        cache_tag: str | None = None,
    ) -> FetchResult:
        del cache_tag
        self.calls.append((api_name, params, force))
        start_date = str(params["start_date"])
        end_date = str(params["end_date"])
        saturated = start_date == "20241231" and end_date == "20250331"
        dates = ["20250102", "20250103"] if saturated else ["20250102"]
        rows = []
        for date in dates:
            row: dict[str, Any] = {field: 1.0 for field in fields}
            row.update(
                {
                    "ts_code": str(params["ts_code"]),
                    "ann_date": date,
                    "f_ann_date": date,
                    "end_date": "20241231",
                    "report_type": "1",
                    "comp_type": "1",
                    "end_type": "4",
                    "update_flag": "0",
                }
            )
            rows.append(row)
        return FetchResult(
            frame=pd.DataFrame(rows, columns=list(fields)),
            request_key=f"{api_name}-{len(self.calls)}",
            cache_hit=False,
        )


def test_saturated_statement_response_is_split_until_complete(tmp_path: Path) -> None:
    spec = replace(
        _spec(tmp_path, api_names=("income",)),
        announcement_start="20241231",
        announcement_end="20250331",
        report_period_start="20241231",
        report_period_end="20241231",
        instruments=("000001.SZ",),
        response_row_limit=2,
        include_disclosure_schedule=False,
    )
    _write_market_inputs(spec)
    client = SaturatingFinancialClient()

    summary = FinancialDownloader(spec, client).run(force=True)  # type: ignore[arg-type]

    income = pd.read_parquet(summary.paths["income"])
    assert income["announcement_date"].tolist() == ["20250102"]
    assert summary.network_requests == 3
