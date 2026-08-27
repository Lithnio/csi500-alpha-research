from dataclasses import replace
from pathlib import Path

import pandas as pd

from csi500_alpha.config import AppConfig, DateSettings
from csi500_alpha.data.quality import validate_smoke


def _quality_inputs() -> dict[str, pd.DataFrame]:
    instruments = [f"{number:06d}.SZ" for number in range(1, 501)]
    snapshot_date = "20250102"
    trade_date = "20250103"
    weights = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "instrument": instruments,
            "weight_pct": 0.2,
            "weight": 0.002,
        }
    )
    bars = pd.DataFrame(
        {
            "trade_date": trade_date,
            "instrument": instruments,
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.1,
        }
    )
    keys = bars[["trade_date", "instrument"]]
    return {
        "calendar": pd.DataFrame(
            {
                "trade_date": [snapshot_date, trade_date],
                "is_open": [1, 1],
            }
        ),
        "benchmark_weights": weights,
        "stock_bars": bars,
        "adjustments": keys.assign(adj_factor=1.0),
        "price_limits": keys.assign(up_limit=11.0, down_limit=9.0),
        "index_bars": pd.DataFrame(),
    }


def test_quality_gate_passes_complete_synthetic_snapshot(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250103",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=False,
            include_instrument_master=False,
            include_industry=False,
        ),
    )

    report = validate_smoke(config, _quality_inputs())

    assert not report.critical_failures
    assert (config.paths.quality_root / "data-quality.json").exists()


def test_quality_gate_rejects_duplicate_and_invalid_bar(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250103",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=False,
            include_instrument_master=False,
            include_industry=False,
        ),
    )
    tables = _quality_inputs()
    invalid = tables["stock_bars"].iloc[[0]].copy()
    invalid["high"] = 8.0
    tables["stock_bars"] = pd.concat(
        [tables["stock_bars"], invalid],
        ignore_index=True,
    )

    report = validate_smoke(config, tables)
    failures = {check.name for check in report.critical_failures}

    assert "stock_bar_primary_key" in failures
    assert "stock_bar_ohlc" in failures


def test_dynamic_universe_missing_must_be_explained(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250103",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=True,
            include_instrument_master=True,
            include_industry=False,
        ),
    )
    tables = _quality_inputs()
    missing_codes = tables["stock_bars"]["instrument"].iloc[:2].tolist()
    tables["stock_bars"] = tables["stock_bars"].iloc[2:].reset_index(drop=True)
    keys = tables["stock_bars"][["trade_date", "instrument"]]
    tables["adjustments"] = keys.assign(adj_factor=1.0)
    tables["price_limits"] = keys.assign(up_limit=11.0, down_limit=9.0)
    tables["suspensions"] = pd.DataFrame(
        {
            "trade_date": ["20250103"],
            "instrument": [missing_codes[0]],
            "suspend_timing": [None],
            "suspend_type": ["S"],
        }
    )
    instruments = tables["benchmark_weights"]["instrument"].tolist()
    tables["instrument_master"] = pd.DataFrame(
        {
            "instrument": instruments,
            "list_date": "20200101",
            "delist_date": [
                "20250103" if instrument == missing_codes[1] else None
                for instrument in instruments
            ],
        }
    )

    report = validate_smoke(config, tables)
    explanation = next(
        check for check in report.checks if check.name == "dynamic_universe_missing_explanations"
    )
    assert explanation.passed
    assert explanation.details["suspension_explained"] == 1
    assert explanation.details["listing_interval_explained"] == 1
    assert explanation.details["unexplained"] == 0

    tables["suspensions"] = tables["suspensions"].iloc[0:0]
    tables["instrument_master"]["delist_date"] = None
    failed = validate_smoke(config, tables)
    failures = {check.name for check in failed.critical_failures}
    assert "dynamic_universe_missing_explanations" in failures


def test_dynamic_cross_table_coverage_requires_every_traded_member(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250103",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=False,
            include_instrument_master=False,
            include_industry=False,
        ),
    )
    tables = _quality_inputs()
    tables["adjustments"] = tables["adjustments"].iloc[1:].reset_index(drop=True)

    report = validate_smoke(config, tables)
    failures = {check.name for check in report.critical_failures}

    assert "adjustment_coverage" not in failures
    assert "dynamic_universe_cross_table_coverage" in failures


def test_optional_eligibility_tables_enter_main_quality_gate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        paths=replace(base.paths, data_root=tmp_path / "data"),
        dates=DateSettings(
            raw_start="20250102",
            backtest_start="20250103",
            end="20250103",
        ),
        download=replace(
            base.download,
            include_daily_basic=False,
            include_suspensions=False,
            include_instrument_master=False,
            include_industry=False,
        ),
    )
    tables = _quality_inputs()
    instruments = tables["benchmark_weights"]["instrument"].tolist()
    tables["name_history"] = pd.DataFrame(
        {
            "instrument": instruments,
            "name": "示例股份",
            "start_date": "20250102",
            "end_date": "20250103",
            "announcement_date": "20200101",
            "change_reason": "other",
            "is_st": False,
        }
    )
    tables["resumptions"] = pd.DataFrame(
        columns=["trade_date", "instrument", "suspend_timing", "suspend_type"]
    )

    report = validate_smoke(config, tables)
    checks = {check.name: check for check in report.checks}
    assert checks["eligibility_supplement_pair"].passed
    assert checks["eligibility_supplement_quality"].passed
    assert checks["name_history_dynamic_member_coverage"].passed
    assert checks["resumption_cross_table_integrity"].passed

    tables["name_history"].loc[0, "end_date"] = "20250102"
    missing_name = validate_smoke(config, tables)
    failures = {check.name for check in missing_name.critical_failures}
    assert "name_history_dynamic_member_coverage" in failures
    tables["name_history"].loc[0, "end_date"] = "20250103"

    tables["resumptions"] = pd.DataFrame(
        {
            "trade_date": ["20250102"],
            "instrument": [instruments[0]],
            "suspend_timing": [None],
            "suspend_type": ["R"],
        }
    )
    delayed_bar = validate_smoke(config, tables)
    delayed_check = next(
        check
        for check in delayed_bar.checks
        if check.name == "resumption_cross_table_integrity"
    )
    assert delayed_check.passed
    assert delayed_check.details["missing_same_day_bars"] == 1
    assert delayed_check.details["delayed_confirmations"] == 1
    assert delayed_check.details["unconfirmed_resumptions"] == 0

    tables["resumptions"] = pd.DataFrame(
        {
            "trade_date": ["20250103"],
            "instrument": ["999999.SZ"],
            "suspend_timing": [None],
            "suspend_type": ["R"],
        }
    )
    unconfirmed = validate_smoke(config, tables)
    failures = {check.name for check in unconfirmed.critical_failures}
    assert "resumption_cross_table_integrity" in failures

    tables["resumptions"] = pd.DataFrame(
        {
            "trade_date": ["20250103"],
            "instrument": [instruments[0]],
            "suspend_timing": [None],
            "suspend_type": ["S"],
        }
    )
    failed = validate_smoke(config, tables)
    failures = {check.name for check in failed.critical_failures}
    assert "eligibility_supplement_quality" in failures
