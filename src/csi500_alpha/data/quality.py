from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.benchmark import (
    EVENT_COLUMNS,
    INTERVAL_COLUMNS,
    active_membership_asof,
)
from csi500_alpha.data.eligibility import validate_eligibility_data
from csi500_alpha.data.storage import write_json_atomic
from csi500_alpha.errors import DataQualityError
from csi500_alpha.research.industry import (
    industry_coverage_by_date,
    overlapping_memberships,
)
from csi500_alpha.utils import utc_now

RESUMPTION_CONFIRMATION_MAX_OPEN_DAYS = 5


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    severity: str
    details: dict[str, Any]


@dataclass(frozen=True)
class QualityReport:
    created_at: str
    checks: tuple[QualityCheck, ...]

    @property
    def critical_failures(self) -> tuple[QualityCheck, ...]:
        return tuple(
            check for check in self.checks if not check.passed and check.severity == "error"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "passed": not self.critical_failures,
            "checks": [asdict(check) for check in self.checks],
        }

    def raise_for_failures(self) -> None:
        if self.critical_failures:
            names = ", ".join(check.name for check in self.critical_failures)
            raise DataQualityError(f"Critical data-quality checks failed: {names}")


def load_silver(config: AppConfig) -> dict[str, pd.DataFrame]:
    root = config.paths.silver_root
    required = [
        "calendar",
        "benchmark_weights",
        "benchmark_membership_events",
        "benchmark_membership_intervals",
        "index_bars",
        "stock_bars",
        "adjustments",
        "price_limits",
    ]
    if config.download.include_daily_basic:
        required.append("daily_characteristics")
    if config.download.include_suspensions:
        required.append("suspensions")
    if config.download.include_instrument_master:
        required.append("instrument_master")
    if config.download.include_industry:
        required.extend(("industry_classification", "industry_membership"))
    missing = [name for name in required if not (root / f"{name}.parquet").exists()]
    if missing:
        raise DataQualityError(f"Missing silver tables: {missing}")
    optional = [
        name
        for name in ("name_history", "resumptions")
        if (root / f"{name}.parquet").exists()
    ]
    tables = {
        name: pd.read_parquet(root / f"{name}.parquet")
        for name in [*required, *optional]
    }
    financial_root = root / "financial"
    if financial_root.exists():
        for path in sorted(financial_root.glob("*.parquet")):
            tables[f"financial_{path.stem}"] = pd.read_parquet(path)
    return tables


def validate_smoke(config: AppConfig, tables: dict[str, pd.DataFrame]) -> QualityReport:
    checks: list[QualityCheck] = []
    calendar = tables["calendar"]
    weights = tables["benchmark_weights"]
    membership_events = tables["benchmark_membership_events"]
    membership_intervals = tables["benchmark_membership_intervals"]
    bars = tables["stock_bars"]
    adjustments = tables["adjustments"]
    limits = tables["price_limits"]
    index_bars = tables["index_bars"]

    open_dates = calendar.loc[calendar["is_open"] == 1, "trade_date"].astype(str)
    decision_dates = [date for date in open_dates if date >= config.dates.backtest_start]
    checks.append(
        QualityCheck(
            "open_calendar_nonempty",
            not open_dates.empty,
            "error",
            {"open_dates": int(len(open_dates))},
        )
    )

    index_required = {
        "trade_date",
        "index_code",
        "total_return_index_code",
        "open",
        "close",
        "total_return_close",
        "total_return_pre_close",
        "total_return_factor",
        "benchmark_open",
        "benchmark_close",
        "benchmark_pre_close",
        "benchmark_method",
    }
    missing_index_columns = sorted(index_required.difference(index_bars.columns))
    checks.append(
        QualityCheck(
            "benchmark_total_return_schema",
            not missing_index_columns and not index_bars.empty,
            "error",
            {
                "rows": int(len(index_bars)),
                "missing_columns": missing_index_columns,
            },
        )
    )
    if not missing_index_columns and not index_bars.empty:
        index_dates = index_bars["trade_date"].astype(str)
        expected_dates = set(open_dates)
        actual_dates = set(index_dates)
        missing_dates = sorted(expected_dates.difference(actual_dates))
        extra_dates = sorted(actual_dates.difference(expected_dates))
        duplicate_dates = int(index_dates.duplicated().sum())
        price_codes = sorted(index_bars["index_code"].dropna().astype(str).unique())
        total_return_codes = sorted(
            index_bars["total_return_index_code"].dropna().astype(str).unique()
        )
        checks.extend(
            [
                QualityCheck(
                    "benchmark_total_return_calendar",
                    duplicate_dates == 0 and not missing_dates and not extra_dates,
                    "error",
                    {
                        "duplicates": duplicate_dates,
                        "missing_dates": missing_dates[:20],
                        "extra_dates": extra_dates[:20],
                        "expected_rows": len(expected_dates),
                        "actual_rows": len(index_bars),
                    },
                ),
                QualityCheck(
                    "benchmark_total_return_codes",
                    price_codes == [config.source.index_code]
                    and total_return_codes
                    == [config.source.total_return_index_code],
                    "error",
                    {
                        "price_codes": price_codes,
                        "expected_price_code": config.source.index_code,
                        "total_return_codes": total_return_codes,
                        "expected_total_return_code": (
                            config.source.total_return_index_code
                        ),
                    },
                ),
            ]
        )

        numeric_columns = [
            "open",
            "close",
            "total_return_close",
            "total_return_pre_close",
            "total_return_factor",
            "benchmark_open",
            "benchmark_close",
            "benchmark_pre_close",
        ]
        numeric = index_bars[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        invalid_values = int((~np.isfinite(numeric) | (numeric <= 0)).sum().sum())
        expected_factor = numeric["total_return_close"] / numeric["close"]
        factor_errors = int(
            (~np.isclose(
                numeric["total_return_factor"],
                expected_factor,
                rtol=1e-12,
                atol=1e-12,
            )).sum()
        )
        open_errors = int(
            (~np.isclose(
                numeric["benchmark_open"],
                numeric["open"] * numeric["total_return_factor"],
                rtol=1e-12,
                atol=1e-12,
            )).sum()
        )
        close_errors = int(
            (~np.isclose(
                numeric["benchmark_close"],
                numeric["total_return_close"],
                rtol=1e-12,
                atol=1e-12,
            )).sum()
        )
        pre_close_errors = int(
            (~np.isclose(
                numeric["benchmark_pre_close"],
                numeric["total_return_pre_close"],
                rtol=1e-12,
                atol=1e-12,
            )).sum()
        )
        method_errors = int(
            (
                index_bars["benchmark_method"].astype(str)
                != "total_return_close_with_price_open_adjustment"
            ).sum()
        )
        checks.append(
            QualityCheck(
                "benchmark_total_return_identity",
                invalid_values == 0
                and factor_errors == 0
                and open_errors == 0
                and close_errors == 0
                and pre_close_errors == 0
                and method_errors == 0,
                "error",
                {
                    "invalid_values": invalid_values,
                    "factor_errors": factor_errors,
                    "open_errors": open_errors,
                    "close_errors": close_errors,
                    "pre_close_errors": pre_close_errors,
                    "method_errors": method_errors,
                    "method": str(index_bars["benchmark_method"].iloc[0]),
                },
            )
        )

    if config.download.include_daily_basic:
        characteristics = tables["daily_characteristics"]
        duplicate_characteristics = int(
            characteristics.duplicated(["trade_date", "instrument"]).sum()
        )
        characteristic_match = bars[["trade_date", "instrument"]].merge(
            characteristics,
            on=["trade_date", "instrument"],
            how="left",
            indicator=True,
        )
        characteristic_coverage = float((characteristic_match["_merge"] == "both").mean())
        invalid_characteristics = int(
            (
                (characteristics["turnover_rate"].notna()) & (characteristics["turnover_rate"] < 0)
            ).sum()
            + (characteristics["circ_mv_cny"].notna() & (characteristics["circ_mv_cny"] <= 0)).sum()
        )
        checks.extend(
            [
                QualityCheck(
                    "daily_characteristics_primary_key",
                    duplicate_characteristics == 0 and not characteristics.empty,
                    "error",
                    {
                        "rows": int(len(characteristics)),
                        "duplicates": duplicate_characteristics,
                    },
                ),
                QualityCheck(
                    "daily_characteristics_coverage",
                    characteristic_coverage >= 0.98,
                    "error",
                    {"coverage": characteristic_coverage, "threshold": 0.98},
                ),
                QualityCheck(
                    "daily_characteristics_values",
                    invalid_characteristics == 0,
                    "error",
                    {"invalid_values": invalid_characteristics},
                ),
            ]
        )

    if config.download.include_instrument_master:
        master = tables["instrument_master"]
        duplicate_master = int(master.duplicated("instrument").sum())
        universe_master_coverage = float(
            weights["instrument"].isin(set(master["instrument"])).mean()
        )
        checks.append(
            QualityCheck(
                "instrument_master_coverage",
                duplicate_master == 0 and universe_master_coverage >= 0.99,
                "error",
                {
                    "rows": int(len(master)),
                    "duplicates": duplicate_master,
                    "benchmark_row_coverage": universe_master_coverage,
                },
            )
        )

    if config.download.include_suspensions:
        suspensions = tables["suspensions"]
        duplicate_suspensions = int(
            suspensions.duplicated(["trade_date", "instrument", "suspend_type"]).sum()
        )
        checks.append(
            QualityCheck(
                "suspension_primary_key",
                duplicate_suspensions == 0,
                "error",
                {
                    "rows": int(len(suspensions)),
                    "duplicates": duplicate_suspensions,
                },
            )
        )

    if config.download.include_industry:
        membership = tables["industry_membership"]
        invalid_intervals = int(
            (
                membership["out_date"].notna()
                & (membership["out_date"].astype(str) <= membership["in_date"].astype(str))
            ).sum()
        )
        overlaps = overlapping_memberships(membership)
        industry_coverage = industry_coverage_by_date(
            membership,
            weights,
            decision_dates,
            transition_date=config.features.industry_transition_date,
        )
        minimum_industry_coverage = (
            float(industry_coverage["coverage"].min()) if not industry_coverage.empty else 0.0
        )
        minimum_row = (
            industry_coverage.loc[industry_coverage["coverage"].idxmin()]
            if not industry_coverage.empty
            else None
        )
        yearly_minimum = (
            industry_coverage.assign(year=industry_coverage["decision_date"].astype(str).str[:4])
            .groupby("year")["coverage"]
            .min()
            .to_dict()
            if not industry_coverage.empty
            else {}
        )
        checks.extend(
            [
                QualityCheck(
                    "industry_membership_intervals",
                    invalid_intervals == 0 and overlaps.empty,
                    "error",
                    {
                        "rows": int(len(membership)),
                        "invalid_intervals": invalid_intervals,
                        "overlaps": int(len(overlaps)),
                    },
                ),
                QualityCheck(
                    "industry_decision_date_coverage",
                    minimum_industry_coverage >= config.features.industry_coverage_threshold,
                    "warning",
                    {
                        "minimum_coverage": minimum_industry_coverage,
                        "threshold": config.features.industry_coverage_threshold,
                        "days_checked": len(industry_coverage),
                        "minimum_date": (
                            str(minimum_row["decision_date"]) if minimum_row is not None else None
                        ),
                        "minimum_taxonomy": (
                            str(minimum_row["taxonomy"]) if minimum_row is not None else None
                        ),
                        "minimum_missing_members": (
                            int(minimum_row["missing_members"]) if minimum_row is not None else None
                        ),
                        "minimum_missing_sample": (
                            list(minimum_row["missing_instruments"][:10])
                            if minimum_row is not None
                            else []
                        ),
                        "yearly_minimum": {
                            str(year): float(coverage) for year, coverage in yearly_minimum.items()
                        },
                    },
                ),
            ]
        )

    weight_stats = weights.groupby("snapshot_date").agg(
        members=("instrument", "nunique"),
        raw_weight_sum=("weight_pct", "sum"),
        weight_sum=("weight", "sum"),
    )
    bad_members = weight_stats.index[weight_stats["members"] != 500].astype(str).tolist()
    bad_raw_sums = (
        weight_stats.index[~weight_stats["raw_weight_sum"].between(99.5, 100.5)]
        .astype(str)
        .tolist()
    )
    bad_normalized = (
        weight_stats.index[~np.isclose(weight_stats["weight_sum"], 1.0, atol=1e-10)]
        .astype(str)
        .tolist()
    )
    checks.extend(
        [
            QualityCheck(
                "benchmark_snapshot_member_count",
                not bad_members and not weight_stats.empty,
                "error",
                {"snapshots": int(len(weight_stats)), "bad_snapshots": bad_members},
            ),
            QualityCheck(
                "benchmark_raw_weight_sum",
                not bad_raw_sums and not weight_stats.empty,
                "error",
                {"bad_snapshots": bad_raw_sums},
            ),
            QualityCheck(
                "benchmark_normalized_weight_sum",
                not bad_normalized and not weight_stats.empty,
                "error",
                {"bad_snapshots": bad_normalized},
            ),
        ]
    )
    checks.extend(
        _benchmark_membership_checks(
            weights,
            membership_events,
            membership_intervals,
            tuple(open_dates.astype(str)),
        )
    )

    eligibility_tables = {"name_history", "resumptions"}.intersection(tables)
    if eligibility_tables:
        missing_eligibility_tables = sorted(
            {"name_history", "resumptions"}.difference(tables)
        )
        checks.append(
            QualityCheck(
                "eligibility_supplement_pair",
                not missing_eligibility_tables,
                "error",
                {"missing_tables": missing_eligibility_tables},
            )
        )
        if not missing_eligibility_tables:
            benchmark_instruments = tuple(
                sorted(weights["instrument"].dropna().astype(str).unique())
            )
            eligibility_quality = validate_eligibility_data(
                tables["name_history"],
                tables["resumptions"],
                benchmark_instruments,
                start_date=config.dates.raw_start,
                end_date=config.dates.end,
            )
            checks.append(
                QualityCheck(
                    "eligibility_supplement_quality",
                    bool(eligibility_quality["passed"]),
                    "error",
                    eligibility_quality,
                )
            )

    duplicate_bars = int(bars.duplicated(["trade_date", "instrument"]).sum())
    invalid_ohlc = int(
        (
            (bars["high"] < bars[["open", "close", "low"]].max(axis=1))
            | (bars["low"] > bars[["open", "close", "high"]].min(axis=1))
            | (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()
    )
    checks.extend(
        [
            QualityCheck(
                "stock_bar_primary_key",
                duplicate_bars == 0 and not bars.empty,
                "error",
                {"rows": int(len(bars)), "duplicates": duplicate_bars},
            ),
            QualityCheck(
                "stock_bar_ohlc",
                invalid_ohlc == 0,
                "error",
                {"invalid_rows": invalid_ohlc},
            ),
        ]
    )

    invalid_adjustment = int(
        ((~np.isfinite(adjustments["adj_factor"])) | (adjustments["adj_factor"] <= 0)).sum()
    )
    checks.append(
        QualityCheck(
            "adjustment_factor_positive",
            invalid_adjustment == 0 and not adjustments.empty,
            "error",
            {"rows": int(len(adjustments)), "invalid_rows": invalid_adjustment},
        )
    )

    merged = (
        bars[["trade_date", "instrument"]]
        .merge(
            adjustments[["trade_date", "instrument"]],
            on=["trade_date", "instrument"],
            how="left",
            indicator="adjustment_match",
        )
        .merge(
            limits[["trade_date", "instrument"]],
            on=["trade_date", "instrument"],
            how="left",
            indicator="limit_match",
        )
    )
    adjustment_coverage = float((merged["adjustment_match"] == "both").mean())
    limit_coverage = float((merged["limit_match"] == "both").mean())
    limit_domain = bars[
        ["trade_date", "instrument", "open"]
    ].drop_duplicates(["trade_date", "instrument"], keep="last").merge(
        limits[
            ["trade_date", "instrument", "up_limit", "down_limit"]
        ].drop_duplicates(["trade_date", "instrument"], keep="last"),
        on=["trade_date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    open_outside_limit = limit_domain[
        (limit_domain["open"] > limit_domain["up_limit"] * (1.0 + 1e-8))
        | (limit_domain["open"] < limit_domain["down_limit"] * (1.0 - 1e-8))
    ]
    checks.extend(
        [
            QualityCheck(
                "adjustment_coverage",
                adjustment_coverage >= 0.98,
                "error",
                {"coverage": adjustment_coverage, "threshold": 0.98},
            ),
            QualityCheck(
                "price_limit_coverage",
                limit_coverage >= 0.98,
                "error",
                {
                    "coverage": limit_coverage,
                    "threshold": 0.98,
                    "special_open_outside_limit_rows": int(len(open_outside_limit)),
                    "special_open_outside_limit_sample": open_outside_limit[
                        ["trade_date", "instrument", "open", "up_limit", "down_limit"]
                    ].head(10).to_dict("records"),
                    "execution_policy": "nominal_limit_not_applied_when_open_is_outside_range",
                },
            ),
        ]
    )

    member_keys = _dynamic_member_keys(
        weights,
        membership_intervals,
        decision_dates,
    )
    if {"name_history", "resumptions"}.issubset(tables):
        name_coverage = _name_history_member_coverage(
            member_keys,
            tables["name_history"],
        )
        checks.append(
            QualityCheck(
                "name_history_dynamic_member_coverage",
                name_coverage["missing_member_days"] == 0
                and name_coverage["multiple_member_days"] == 0,
                "error",
                name_coverage,
            )
        )

        resumptions = tables["resumptions"]
        resumption_keys = resumptions[
            ["trade_date", "instrument"]
        ].drop_duplicates()
        bar_keys = bars[["trade_date", "instrument"]].drop_duplicates()
        resumption_bar_match = resumption_keys.merge(
            bar_keys,
            on=["trade_date", "instrument"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        missing_resumption_bars = int(
            resumption_bar_match["_merge"].ne("both").sum()
        )
        delayed_resumption_confirmation = _delayed_resumption_bar_confirmation(
            resumption_bar_match.loc[
                resumption_bar_match["_merge"].ne("both"),
                ["trade_date", "instrument"],
            ],
            bar_keys,
            tuple(open_dates.astype(str)),
        )
        unconfirmed_resumptions = delayed_resumption_confirmation.loc[
            delayed_resumption_confirmation["open_day_lag"].isna()
            | delayed_resumption_confirmation["open_day_lag"].gt(
                RESUMPTION_CONFIRMATION_MAX_OPEN_DAYS
            )
        ]
        delayed_confirmations = delayed_resumption_confirmation.loc[
            delayed_resumption_confirmation["open_day_lag"].notna()
            & delayed_resumption_confirmation["open_day_lag"].le(
                RESUMPTION_CONFIRMATION_MAX_OPEN_DAYS
            )
        ]
        open_date_set = set(open_dates.astype(str))
        non_open_resumptions = int(
            (~resumptions["trade_date"].astype(str).isin(open_date_set)).sum()
        )
        outside_resumption_window = int(
            (
                ~resumptions["trade_date"].astype(str).between(
                    config.dates.raw_start,
                    config.dates.end,
                )
            ).sum()
        )
        orphan_resumptions = 0
        master_check_enabled = "instrument_master" in tables
        if master_check_enabled:
            master_names = set(tables["instrument_master"]["instrument"].astype(str))
            orphan_resumptions = int(
                (~resumptions["instrument"].astype(str).isin(master_names)).sum()
            )
        checks.append(
            QualityCheck(
                "resumption_cross_table_integrity",
                non_open_resumptions == 0
                and outside_resumption_window == 0
                and orphan_resumptions == 0
                and unconfirmed_resumptions.empty,
                "error",
                {
                    "rows": int(len(resumptions)),
                    "non_open_dates": non_open_resumptions,
                    "outside_configured_window": outside_resumption_window,
                    "master_check_enabled": master_check_enabled,
                    "orphan_instruments": orphan_resumptions,
                    "missing_same_day_bars": missing_resumption_bars,
                    "same_day_bar_coverage": (
                        1.0 - missing_resumption_bars / len(resumption_keys)
                        if len(resumption_keys)
                        else 1.0
                    ),
                    "confirmation_max_open_days": (
                        RESUMPTION_CONFIRMATION_MAX_OPEN_DAYS
                    ),
                    "delayed_confirmations": int(len(delayed_confirmations)),
                    "delayed_confirmation_sample": delayed_confirmations.head(20).to_dict(
                        "records"
                    ),
                    "unconfirmed_resumptions": int(len(unconfirmed_resumptions)),
                    "unconfirmed_sample": unconfirmed_resumptions.head(20).to_dict(
                        "records"
                    ),
                },
            )
        )

    availability = member_keys.merge(
        bars[["trade_date", "instrument"]].drop_duplicates().assign(has_bar=True),
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    availability["has_bar"] = availability["has_bar"].fillna(False).astype(bool)
    coverage_by_date = availability.groupby("trade_date", sort=True)["has_bar"].mean()
    min_universe_coverage = float(coverage_by_date.min()) if not coverage_by_date.empty else 0.0
    minimum_coverage_date = (
        str(coverage_by_date.idxmin()) if not coverage_by_date.empty else None
    )
    explanations_available = (
        config.download.include_suspensions
        and config.download.include_instrument_master
        and "suspensions" in tables
        and "instrument_master" in tables
    )
    checks.append(
        QualityCheck(
            "dynamic_universe_daily_bar_coverage",
            min_universe_coverage >= 0.90,
            "warning" if explanations_available else "error",
            {
                "minimum_coverage": min_universe_coverage,
                "minimum_date": minimum_coverage_date,
                "days_checked": int(len(coverage_by_date)),
                "member_days": int(len(availability)),
                "missing_member_days": int((~availability["has_bar"]).sum()),
                "raw_coverage_threshold": 0.90,
                "interpretation": (
                    "diagnostic_only_when_missing_explanations_are_available"
                    if explanations_available
                    else "critical_fallback_without_missing_explanations"
                ),
            },
        )
    )

    if explanations_available:
        availability = _classify_dynamic_missing(availability, tables)
        missing = ~availability["has_bar"]
        suspension_explained = int((missing & availability["suspension_explained"]).sum())
        listing_explained = int((missing & availability["listing_explained"]).sum())
        unexplained = availability[
            missing
            & ~availability["suspension_explained"]
            & ~availability["listing_explained"]
        ]
        checks.append(
            QualityCheck(
                "dynamic_universe_missing_explanations",
                unexplained.empty,
                "error",
                {
                    "missing_member_days": int(missing.sum()),
                    "suspension_explained": suspension_explained,
                    "listing_interval_explained": listing_explained,
                    "unexplained": int(len(unexplained)),
                    "unexplained_sample": unexplained[
                        ["trade_date", "instrument", "snapshot_date"]
                    ].head(10).to_dict("records"),
                },
            )
        )

    active_bar_keys = availability.loc[
        availability["has_bar"], ["trade_date", "instrument"]
    ]
    dynamic_table_names = ["adjustments", "price_limits"]
    if config.download.include_daily_basic:
        dynamic_table_names.append("daily_characteristics")
    table_coverage: dict[str, dict[str, float | int]] = {}
    total_active = int(len(active_bar_keys))
    for name in dynamic_table_names:
        matches = active_bar_keys.merge(
            tables[name][["trade_date", "instrument"]].drop_duplicates(),
            on=["trade_date", "instrument"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )["_merge"].eq("both")
        matched = int(matches.sum())
        table_coverage[name] = {
            "matched": matched,
            "missing": total_active - matched,
            "coverage": matched / total_active if total_active else 0.0,
        }
    checks.append(
        QualityCheck(
            "dynamic_universe_cross_table_coverage",
            total_active > 0
            and all(details["missing"] == 0 for details in table_coverage.values()),
            "error",
            {
                "active_bar_member_days": total_active,
                "tables": table_coverage,
            },
        )
    )

    report = QualityReport(created_at=utc_now(), checks=tuple(checks))
    write_json_atomic(report.to_dict(), config.paths.quality_root / "data-quality.json")
    return report


def _dynamic_member_keys(
    weights: pd.DataFrame,
    membership_intervals: pd.DataFrame,
    decision_dates: list[str] | pd.Series,
) -> pd.DataFrame:
    columns = ["trade_date", "snapshot_date", "instrument"]
    if weights.empty or membership_intervals.empty:
        return pd.DataFrame(columns=columns)
    snapshots = sorted(weights["snapshot_date"].astype(str).unique())
    frames: list[pd.DataFrame] = []
    for trade_date in sorted({str(value) for value in decision_dates}):
        snapshot_position = bisect_left(snapshots, trade_date) - 1
        if snapshot_position < 0:
            continue
        snapshot_date = snapshots[snapshot_position]
        active = active_membership_asof(membership_intervals, trade_date)
        if active.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": trade_date,
                    "snapshot_date": snapshot_date,
                    "instrument": active["instrument"].astype(str).to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _benchmark_membership_checks(
    weights: pd.DataFrame,
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    open_dates: tuple[str, ...],
) -> list[QualityCheck]:
    missing_event_columns = sorted(set(EVENT_COLUMNS).difference(events.columns))
    missing_interval_columns = sorted(
        set(INTERVAL_COLUMNS).difference(intervals.columns)
    )
    schema_ok = (
        not missing_event_columns
        and not missing_interval_columns
        and not events.empty
        and not intervals.empty
    )
    checks = [
        QualityCheck(
            "benchmark_membership_schema",
            schema_ok,
            "error",
            {
                "event_rows": int(len(events)),
                "interval_rows": int(len(intervals)),
                "missing_event_columns": missing_event_columns,
                "missing_interval_columns": missing_interval_columns,
            },
        )
    ]
    if not schema_ok:
        return checks

    normalized_events = events.copy()
    normalized_intervals = intervals.copy()
    event_string_columns = [
        "event_id",
        "published_date",
        "effective_from",
        "event_type",
        "action",
        "instrument",
        "confirmation_snapshot_date",
        "source",
    ]
    for column in event_string_columns:
        normalized_events[column] = normalized_events[column].astype(str)
    for column in (
        "instrument",
        "effective_from",
        "entry_event_id",
        "entry_published_date",
        "entry_source",
    ):
        normalized_intervals[column] = normalized_intervals[column].astype(str)

    open_date_set = set(open_dates)
    snapshot_date_set = set(weights["snapshot_date"].astype(str))
    duplicate_actions = int(
        normalized_events.duplicated(["event_id", "action", "instrument"]).sum()
    )
    invalid_actions = int(
        (~normalized_events["action"].isin({"add", "remove"})).sum()
    )
    late_publications = int(
        (
            normalized_events["published_date"]
            > normalized_events["effective_from"]
        ).sum()
    )
    non_open_events = int(
        (~normalized_events["effective_from"].isin(open_date_set)).sum()
    )
    unknown_confirmations = int(
        (
            ~normalized_events["confirmation_snapshot_date"].isin(
                snapshot_date_set
            )
        ).sum()
    )
    malformed_events: list[dict[str, Any]] = []
    for event_id, frame in normalized_events.groupby("event_id", sort=True):
        event_type = str(frame["event_type"].iloc[0])
        additions = set(frame.loc[frame["action"] == "add", "instrument"])
        removals = set(frame.loc[frame["action"] == "remove", "instrument"])
        metadata_unique = all(
            frame[column].nunique(dropna=False) == 1
            for column in (
                "published_date",
                "effective_from",
                "event_type",
                "confirmation_snapshot_date",
                "source",
            )
        )
        valid_change = (
            not removals and bool(additions)
            if event_type == "baseline_snapshot"
            else bool(additions)
            and len(additions) == len(removals)
            and not additions.intersection(removals)
            and event_type in {"regular", "temporary"}
        )
        if not metadata_unique or not valid_change:
            malformed_events.append(
                {
                    "event_id": str(event_id),
                    "event_type": event_type,
                    "additions": len(additions),
                    "removals": len(removals),
                    "metadata_unique": metadata_unique,
                }
            )
    checks.append(
        QualityCheck(
            "benchmark_membership_event_integrity",
            duplicate_actions == 0
            and invalid_actions == 0
            and late_publications == 0
            and non_open_events == 0
            and unknown_confirmations == 0
            and not malformed_events,
            "error",
            {
                "events": int(normalized_events["event_id"].nunique()),
                "duplicate_actions": duplicate_actions,
                "invalid_actions": invalid_actions,
                "published_after_effective": late_publications,
                "non_open_effective_dates": non_open_events,
                "unknown_confirmation_snapshots": unknown_confirmations,
                "malformed_events": malformed_events[:20],
            },
        )
    )

    normalized_intervals["effective_to"] = normalized_intervals[
        "effective_to"
    ].fillna("").astype(str)
    duplicate_intervals = int(
        normalized_intervals.duplicated(["instrument", "effective_from"]).sum()
    )
    invalid_ranges = int(
        (
            normalized_intervals["effective_to"].ne("")
            & (
                normalized_intervals["effective_to"]
                <= normalized_intervals["effective_from"]
            )
        ).sum()
    )
    future_entry_sources = int(
        (
            normalized_intervals["entry_published_date"]
            > normalized_intervals["effective_from"]
        ).sum()
    )
    event_ids = set(normalized_events["event_id"])
    unknown_entry_events = int(
        (~normalized_intervals["entry_event_id"].isin(event_ids)).sum()
    )
    exit_event_ids = normalized_intervals["exit_event_id"].dropna().astype(str)
    unknown_exit_events = int((~exit_event_ids.isin(event_ids)).sum())
    missing_sources = int(
        normalized_intervals["entry_source"].str.strip().eq("").sum()
    )
    overlap_rows: list[dict[str, str]] = []
    for instrument, frame in normalized_intervals.groupby("instrument", sort=True):
        ordered = frame.sort_values("effective_from")
        previous_to = ""
        for position, row in enumerate(ordered.itertuples(index=False)):
            current_from = str(row.effective_from)
            if position > 0 and (not previous_to or previous_to > current_from):
                overlap_rows.append(
                    {"instrument": str(instrument), "effective_from": current_from}
                )
            previous_to = str(row.effective_to or "")
    checks.append(
        QualityCheck(
            "benchmark_membership_interval_integrity",
            duplicate_intervals == 0
            and invalid_ranges == 0
            and future_entry_sources == 0
            and unknown_entry_events == 0
            and unknown_exit_events == 0
            and missing_sources == 0
            and not overlap_rows,
            "error",
            {
                "duplicate_intervals": duplicate_intervals,
                "invalid_ranges": invalid_ranges,
                "entry_source_after_effective": future_entry_sources,
                "unknown_entry_events": unknown_entry_events,
                "unknown_exit_events": unknown_exit_events,
                "missing_entry_sources": missing_sources,
                "overlap_sample": overlap_rows[:20],
            },
        )
    )

    expected_members = int(
        weights.groupby("snapshot_date")["instrument"].nunique().mode().iloc[0]
    )
    event_transition_errors: list[dict[str, Any]] = []
    for effective_from, frame in normalized_events.groupby(
        "effective_from", sort=True
    ):
        date_position = bisect_left(open_dates, str(effective_from))
        previous_date = open_dates[date_position - 1] if date_position > 0 else None
        before = (
            set(
                active_membership_asof(normalized_intervals, previous_date)[
                    "instrument"
                ].astype(str)
            )
            if previous_date is not None
            else set()
        )
        additions = set(frame.loc[frame["action"] == "add", "instrument"])
        removals = set(frame.loc[frame["action"] == "remove", "instrument"])
        expected = before.difference(removals).union(additions)
        actual = set(
            active_membership_asof(normalized_intervals, str(effective_from))[
                "instrument"
            ].astype(str)
        )
        if expected != actual or len(actual) != expected_members:
            event_transition_errors.append(
                {
                    "effective_from": str(effective_from),
                    "expected_members": len(expected),
                    "actual_members": len(actual),
                    "missing": sorted(expected.difference(actual))[:10],
                    "extra": sorted(actual.difference(expected))[:10],
                }
            )
    checks.append(
        QualityCheck(
            "benchmark_membership_event_transitions",
            not event_transition_errors,
            "error",
            {
                "effective_dates": int(
                    normalized_events["effective_from"].nunique()
                ),
                "expected_members_per_date": expected_members,
                "errors": event_transition_errors[:20],
            },
        )
    )

    snapshot_errors: list[dict[str, Any]] = []
    for snapshot_date, frame in weights.groupby("snapshot_date", sort=True):
        position = bisect_right(open_dates, str(snapshot_date))
        if position >= len(open_dates):
            continue
        check_date = open_dates[position]
        expected = set(frame["instrument"].astype(str))
        actual = set(
            active_membership_asof(normalized_intervals, check_date)[
                "instrument"
            ].astype(str)
        )
        if expected != actual:
            snapshot_errors.append(
                {
                    "snapshot_date": str(snapshot_date),
                    "check_date": check_date,
                    "missing": sorted(expected.difference(actual))[:10],
                    "extra": sorted(actual.difference(expected))[:10],
                }
            )
    checks.append(
        QualityCheck(
            "benchmark_membership_snapshot_reconciliation",
            not snapshot_errors,
            "error",
            {
                "snapshots_checked": int(
                    sum(
                        bisect_right(open_dates, str(snapshot)) < len(open_dates)
                        for snapshot in weights["snapshot_date"].astype(str).unique()
                    )
                ),
                "errors": snapshot_errors[:20],
            },
        )
    )
    return checks


def _classify_dynamic_missing(
    availability: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = availability.copy()
    suspensions = tables["suspensions"]
    if "suspend_type" in suspensions:
        suspensions = suspensions[suspensions["suspend_type"].astype(str).eq("S")]
    suspension_keys = suspensions[["trade_date", "instrument"]].drop_duplicates().assign(
        suspension_explained=True
    )
    result = result.merge(
        suspension_keys,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    result["suspension_explained"] = (
        result["suspension_explained"].fillna(False).astype(bool)
    )

    master = (
        tables["instrument_master"][["instrument", "list_date", "delist_date"]]
        .drop_duplicates("instrument", keep="last")
        .copy()
    )
    master["list_date"] = master["list_date"].fillna("").astype(str)
    master["delist_date"] = master["delist_date"].fillna("").astype(str)
    result = result.merge(master, on="instrument", how="left", validate="many_to_one")
    before_listing = result["list_date"].ne("") & (
        result["trade_date"] < result["list_date"]
    )
    after_delisting = result["delist_date"].ne("") & (
        result["trade_date"] >= result["delist_date"]
    )
    result["listing_explained"] = (
        ~result["has_bar"]
        & ~result["suspension_explained"]
        & (before_listing | after_delisting)
    )
    return result


def _name_history_member_coverage(
    member_keys: pd.DataFrame,
    name_history: pd.DataFrame,
) -> dict[str, Any]:
    keys = member_keys[["trade_date", "instrument"]].reset_index(drop=True).copy()
    counts = np.zeros(len(keys), dtype=np.int16)
    history_groups = {
        str(instrument): group
        for instrument, group in name_history.groupby("instrument", sort=False)
    }
    for instrument, positions in keys.groupby("instrument", sort=False).indices.items():
        history = history_groups.get(str(instrument))
        if history is None:
            continue
        position_array = np.asarray(positions, dtype=int)
        dates = keys.iloc[position_array]["trade_date"].astype(str).to_numpy()
        for row in history.itertuples(index=False):
            start = "" if pd.isna(row.start_date) else str(row.start_date)
            end = "99999999" if pd.isna(row.end_date) else str(row.end_date)
            counts[position_array] += ((dates >= start) & (dates <= end)).astype(
                np.int16
            )
    missing = keys.loc[counts == 0]
    multiple = keys.loc[counts > 1]
    return {
        "member_days": int(len(keys)),
        "exactly_one": int((counts == 1).sum()),
        "missing_member_days": int(len(missing)),
        "multiple_member_days": int(len(multiple)),
        "missing_sample": missing.head(20).to_dict("records"),
        "multiple_sample": multiple.head(20).to_dict("records"),
    }


def _delayed_resumption_bar_confirmation(
    missing_same_day: pd.DataFrame,
    bar_keys: pd.DataFrame,
    open_dates: tuple[str, ...],
) -> pd.DataFrame:
    """Find the first actual bar after a vendor resumption-status date."""

    columns = ["instrument", "event_date", "next_bar_date", "open_day_lag"]
    if missing_same_day.empty:
        return pd.DataFrame(columns=columns)
    open_position = {date: position for position, date in enumerate(open_dates)}
    instruments = set(missing_same_day["instrument"].astype(str))
    relevant_bars = bar_keys.loc[
        bar_keys["instrument"].astype(str).isin(instruments),
        ["trade_date", "instrument"],
    ]
    bar_dates = {
        str(instrument): sorted(group["trade_date"].astype(str).unique())
        for instrument, group in relevant_bars.groupby("instrument", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for row in missing_same_day.itertuples(index=False):
        instrument = str(row.instrument)
        event_date = str(row.trade_date)
        dates = bar_dates.get(instrument, [])
        index = bisect_left(dates, event_date)
        next_bar_date = dates[index] if index < len(dates) else None
        event_position = open_position.get(event_date)
        next_position = (
            open_position.get(next_bar_date) if next_bar_date is not None else None
        )
        open_day_lag = (
            next_position - event_position
            if event_position is not None and next_position is not None
            else None
        )
        rows.append(
            {
                "instrument": instrument,
                "event_date": event_date,
                "next_bar_date": next_bar_date,
                "open_day_lag": open_day_lag,
            }
        )
    return pd.DataFrame(rows, columns=columns)
