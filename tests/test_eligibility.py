from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from csi500_alpha.config import AppConfig, DateSettings
from csi500_alpha.data.client import FetchResult
from csi500_alpha.data.eligibility import (
    EligibilityDownloader,
    build_eligibility_download_plan,
    restrict_name_history_to_window,
    validate_eligibility_data,
)
from csi500_alpha.data.normalize import normalize_name_history


class FakeEligibilityClient:
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
        if api_name == "namechange":
            instrument = str(params["ts_code"])
            name = "*ST测试" if instrument == "000001.SZ" else "正常公司"
            rows = [(instrument, name, "20200101", None, "20191231", "测试")]
            if instrument == "000001.SZ":
                rows.append(
                    (instrument, "未来名称", "20260101", None, "20251231", "改名")
                )
        elif api_name == "suspend_d":
            rows = [("000001.SZ", "20250103", None, "R")]
        else:
            rows = []
        return FetchResult(
            frame=pd.DataFrame(rows, columns=list(fields)),
            request_key=f"{api_name}-{len(self.calls)}",
            cache_hit=False,
        )


def test_eligibility_supplement_is_resumable_and_quality_checked(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="eligibility"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
    )
    config.paths.silver_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "snapshot_date": ["20250102", "20250102"],
            "instrument": ["000001.SZ", "000002.SZ"],
            "weight": [0.5, 0.5],
        }
    ).to_parquet(config.paths.silver_root / "benchmark_weights.parquet", index=False)
    client = FakeEligibilityClient()

    plan = build_eligibility_download_plan(config)
    summary = EligibilityDownloader(config, client).run(force=True)  # type: ignore[arg-type]

    assert plan.estimated_requests == 3
    assert summary.network_requests == 3
    assert all(force for _, _, force in client.calls)
    names = pd.read_parquet(summary.paths["name_history"])
    resumptions = pd.read_parquet(summary.paths["resumptions"])
    assert names.set_index("instrument").loc["000001.SZ", "is_st"]
    assert not names.set_index("instrument").loc["000002.SZ", "is_st"]
    assert names["start_date"].eq("20250102").all()
    assert names["end_date"].eq("20250103").all()
    assert not names["name"].eq("未来名称").any()
    assert resumptions.loc[0, "suspend_type"] == "R"
    assert summary.quality_path.exists()
    assert summary.progress_path.exists()


def test_name_refresh_is_limited_to_window_constituents(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="refresh-names"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
    )
    config.paths.silver_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "snapshot_date": ["20241231", "20250102", "20250103"],
            "instrument": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "weight": [1.0, 1.0, 1.0],
        }
    ).to_parquet(config.paths.silver_root / "benchmark_weights.parquet", index=False)
    client = FakeEligibilityClient()

    summary = EligibilityDownloader(config, client).run(  # type: ignore[arg-type]
        refresh_names_from="20250103"
    )

    name_forces = {
        str(params["ts_code"]): force
        for api_name, params, force in client.calls
        if api_name == "namechange"
    }
    assert name_forces == {
        "000001.SZ": False,
        "000002.SZ": True,
        "000003.SZ": True,
    }
    name_tags = {
        str(params["ts_code"]): cache_tag
        for (api_name, params, _), cache_tag in zip(
            client.calls,
            client.cache_tags,
            strict=True,
        )
        if api_name == "namechange"
    }
    assert name_tags == {
        "000001.SZ": None,
        "000002.SZ": "refresh-names-20250103.eligibility",
        "000003.SZ": "refresh-names-20250103.eligibility",
    }
    assert all(
        not force
        for api_name, _, force in client.calls
        if api_name == "suspend_d"
    )
    quality = json.loads(summary.quality_path.read_text(encoding="utf-8"))
    assert quality["request_policy"]["refresh_name_instruments"] == 2


def test_configured_name_snapshot_is_versioned_and_resumable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data", dataset="final-names"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250102",
            end="20250103",
        ),
        download=replace(
            base.download,
            reference_cache_tag="final-20250103",
            eligibility_refresh_start="20250103",
        ),
    )
    config.paths.silver_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "snapshot_date": ["20241231", "20250102", "20250103"],
            "instrument": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "weight": [1.0, 1.0, 1.0],
        }
    ).to_parquet(config.paths.silver_root / "benchmark_weights.parquet", index=False)
    client = FakeEligibilityClient()

    summary = EligibilityDownloader(config, client).run()  # type: ignore[arg-type]

    assert all(not force for _, _, force in client.calls)
    name_tags = {
        str(params["ts_code"]): cache_tag
        for (api_name, params, _), cache_tag in zip(
            client.calls,
            client.cache_tags,
            strict=True,
        )
        if api_name == "namechange"
    }
    assert name_tags == {
        "000001.SZ": None,
        "000002.SZ": "final-20250103.eligibility",
        "000003.SZ": "final-20250103.eligibility",
    }
    quality = json.loads(summary.quality_path.read_text(encoding="utf-8"))
    assert quality["request_policy"] == {
        "force_all": False,
        "configured_refresh_names_from": "20250103",
        "refresh_names_from": "20250103",
        "explicit_name_refresh": False,
        "refresh_name_instruments": 2,
        "name_cache_tag": "final-20250103.eligibility",
    }


def test_revised_open_name_event_keeps_latest_effective_date() -> None:
    fields = [
        "ts_code",
        "name",
        "start_date",
        "end_date",
        "ann_date",
        "change_reason",
    ]
    raw = pd.DataFrame(
        [
            ("600388.SH", "龙净环保", "20230717", "20260816", "20230714", "撤销ST"),
            ("600388.SH", "紫金龙净", "20260422", None, "20260422", "其他"),
            ("600388.SH", "紫金龙净", "20260817", None, "20260422", "其他"),
        ],
        columns=fields,
    )

    normalized = normalize_name_history([raw])
    assert normalized["start_date"].tolist() == ["20230717", "20260817"]
    assert normalized.attrs["normalization"] == {
        "raw_rows": 3,
        "exact_duplicate_rows_removed": 0,
        "superseded_open_name_events_removed": 1,
    }

    window = restrict_name_history_to_window(
        normalized,
        start_date="20260101",
        end_date="20260630",
    )
    assert window[["name", "start_date", "end_date"]].to_dict("records") == [
        {
            "name": "龙净环保",
            "start_date": "20260101",
            "end_date": "20260630",
        }
    ]
    validation = validate_eligibility_data(
        window,
        pd.DataFrame(
            columns=["instrument", "trade_date", "suspend_timing", "suspend_type"]
        ),
        ("600388.SH",),
        start_date="20260101",
        end_date="20260630",
    )
    assert validation["passed"]
    assert validation["name_history"]["overlapping_intervals"] == 0
