from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import AppConfig
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
    return {
        name: pd.read_parquet(root / f"{name}.parquet")
        for name in [*required, *optional]
    }


def validate_smoke(config: AppConfig, tables: dict[str, pd.DataFrame]) -> QualityReport:
    checks: list[QualityCheck] = []
    calendar = tables["calendar"]
    weights = tables["benchmark_weights"]
    bars = tables["stock_bars"]
    adjustments = tables["adjustments"]
    limits = tables["price_limits"]

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

    member_keys = _dynamic_member_keys(weights, decision_dates)
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
    decision_dates: list[str] | pd.Series,
) -> pd.DataFrame:
    columns = ["trade_date", "snapshot_date", "instrument"]
    if weights.empty:
        return pd.DataFrame(columns=columns)
    grouped = {
        str(snapshot): frame["instrument"].astype(str).drop_duplicates().sort_values()
        for snapshot, frame in weights.groupby("snapshot_date", sort=True)
    }
    snapshots = sorted(grouped)
    snapshot_position = -1
    frames: list[pd.DataFrame] = []
    for trade_date in sorted({str(value) for value in decision_dates}):
        while (
            snapshot_position + 1 < len(snapshots)
            and snapshots[snapshot_position + 1] < trade_date
        ):
            snapshot_position += 1
        if snapshot_position < 0:
            continue
        snapshot_date = snapshots[snapshot_position]
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": trade_date,
                    "snapshot_date": snapshot_date,
                    "instrument": grouped[snapshot_date].to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


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
