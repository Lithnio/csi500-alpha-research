from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.data.financial import select_financial_versions_asof
from csi500_alpha.errors import DataQualityError
from csi500_alpha.features.catalog import FactorDefinition
from csi500_alpha.logging_utils import ProgressCallback, ProgressLogger

LOGGER = logging.getLogger(__name__)

_FINANCIAL_AVAILABILITY = (
    "latest visible statement revision; usable from the first open day after disclosure"
)
_NO_FALLBACK = "missing or stale accounting inputs remain missing; no report-period fallback"

FUNDAMENTAL_FACTOR_CATALOG: tuple[FactorDefinition, ...] = (
    FactorDefinition(
        "gross_profitability_ttm",
        1,
        4,
        "v1",
        ("revenue", "oper_cost", "total_assets"),
        _FINANCIAL_AVAILABILITY,
        "quality",
        _NO_FALLBACK,
        "(revenue_ttm - oper_cost_ttm) / average_total_assets",
        "Profitable production relative to assets may identify higher-quality firms.",
    ),
    FactorDefinition(
        "roe_ttm",
        1,
        4,
        "v1",
        ("n_income_attr_p", "total_hldr_eqy_exc_min_int"),
        _FINANCIAL_AVAILABILITY,
        "quality",
        _NO_FALLBACK,
        "attributable_net_income_ttm / average_parent_equity",
        "Sustainable profitability on shareholder capital may be rewarded.",
    ),
    FactorDefinition(
        "roa_ttm",
        1,
        4,
        "v1",
        ("n_income_attr_p", "total_assets"),
        _FINANCIAL_AVAILABILITY,
        "quality",
        _NO_FALLBACK,
        "attributable_net_income_ttm / average_total_assets",
        "Efficient use of assets may indicate business quality.",
    ),
    FactorDefinition(
        "operating_margin_ttm",
        1,
        4,
        "v1",
        ("operate_profit", "total_revenue"),
        _FINANCIAL_AVAILABILITY,
        "quality",
        _NO_FALLBACK,
        "operating_profit_ttm / total_revenue_ttm",
        "Higher operating margins may reflect pricing power and cost control.",
    ),
    FactorDefinition(
        "cash_return_on_assets_ttm",
        1,
        4,
        "v1",
        ("n_cashflow_act", "total_assets"),
        _FINANCIAL_AVAILABILITY,
        "quality",
        _NO_FALLBACK,
        "operating_cash_flow_ttm / average_total_assets",
        "Cash profitability is harder to sustain through accrual choices.",
    ),
    FactorDefinition(
        "book_to_market",
        1,
        0,
        "v1",
        ("total_hldr_eqy_exc_min_int", "total_mv_cny"),
        _FINANCIAL_AVAILABILITY,
        "value",
        _NO_FALLBACK,
        "parent_equity / total_market_cap",
        "A lower market price per unit of book equity may earn a value premium.",
    ),
    FactorDefinition(
        "earnings_yield_ttm",
        1,
        4,
        "v1",
        ("n_income_attr_p", "total_mv_cny"),
        _FINANCIAL_AVAILABILITY,
        "value",
        _NO_FALLBACK,
        "attributable_net_income_ttm / total_market_cap",
        "A lower price per unit of current earnings may earn a value premium.",
    ),
    FactorDefinition(
        "sales_to_price_ttm",
        1,
        4,
        "v1",
        ("total_revenue", "total_mv_cny"),
        _FINANCIAL_AVAILABILITY,
        "value",
        _NO_FALLBACK,
        "total_revenue_ttm / total_market_cap",
        "Sales provide a valuation anchor less affected by current margins.",
    ),
    FactorDefinition(
        "cfo_yield_ttm",
        1,
        4,
        "v1",
        ("n_cashflow_act", "total_mv_cny"),
        _FINANCIAL_AVAILABILITY,
        "value",
        _NO_FALLBACK,
        "operating_cash_flow_ttm / total_market_cap",
        "A lower price per unit of operating cash flow may be rewarded.",
    ),
    FactorDefinition(
        "revenue_growth_ttm_yoy",
        1,
        8,
        "v1",
        ("total_revenue",),
        _FINANCIAL_AVAILABILITY,
        "growth",
        _NO_FALLBACK,
        "total_revenue_ttm / total_revenue_ttm_lag4q - 1",
        "Persistent sales growth may contain information beyond current valuation.",
    ),
    FactorDefinition(
        "earnings_growth_ttm_yoy",
        1,
        8,
        "v1",
        ("n_income_attr_p",),
        _FINANCIAL_AVAILABILITY,
        "growth",
        _NO_FALLBACK,
        "(net_income_ttm - net_income_ttm_lag4q) / abs(net_income_ttm_lag4q)",
        "Improving attributable earnings may persist over intermediate horizons.",
    ),
    FactorDefinition(
        "cfo_growth_ttm_yoy",
        1,
        8,
        "v1",
        ("n_cashflow_act",),
        _FINANCIAL_AVAILABILITY,
        "growth",
        _NO_FALLBACK,
        "(operating_cash_flow_ttm - lag4q) / abs(lag4q)",
        "Cash-flow growth can corroborate reported earnings growth.",
    ),
    FactorDefinition(
        "roe_change_yoy",
        1,
        8,
        "v1",
        ("n_income_attr_p", "total_hldr_eqy_exc_min_int"),
        _FINANCIAL_AVAILABILITY,
        "growth",
        _NO_FALLBACK,
        "roe_ttm - roe_ttm_lag4q",
        "Improving capital efficiency may signal positive fundamental change.",
    ),
    FactorDefinition(
        "asset_growth_yoy",
        -1,
        4,
        "v1",
        ("total_assets",),
        _FINANCIAL_AVAILABILITY,
        "investment",
        _NO_FALLBACK,
        "total_assets / total_assets_lag4q - 1",
        "Rapid balance-sheet expansion may predict lower subsequent returns.",
    ),
    FactorDefinition(
        "capex_intensity_ttm",
        -1,
        4,
        "v1",
        ("c_pay_acq_const_fiolta", "total_assets"),
        _FINANCIAL_AVAILABILITY,
        "investment",
        _NO_FALLBACK,
        "cash_paid_for_long_lived_assets_ttm / average_total_assets",
        "Aggressive investment may be followed by weaker marginal returns.",
    ),
    FactorDefinition(
        "total_accruals_ttm",
        -1,
        4,
        "v1",
        ("n_income_attr_p", "n_cashflow_act", "total_assets"),
        _FINANCIAL_AVAILABILITY,
        "accrual",
        _NO_FALLBACK,
        "(net_income_ttm - operating_cash_flow_ttm) / average_total_assets",
        "A larger non-cash component of earnings may be less persistent.",
    ),
    FactorDefinition(
        "cash_conversion_ttm",
        1,
        4,
        "v1",
        ("n_cashflow_act", "n_income_attr_p"),
        _FINANCIAL_AVAILABILITY,
        "accrual",
        _NO_FALLBACK,
        "operating_cash_flow_ttm / abs(attributable_net_income_ttm)",
        "Cash-backed earnings may be more persistent than accrual-heavy earnings.",
    ),
)

FUNDAMENTAL_FACTOR_NAMES = tuple(
    definition.name for definition in FUNDAMENTAL_FACTOR_CATALOG
)
FUNDAMENTAL_DIRECTIONS = {
    definition.name: definition.direction for definition in FUNDAMENTAL_FACTOR_CATALOG
}
FUNDAMENTAL_FAMILIES = {
    definition.name: definition.family for definition in FUNDAMENTAL_FACTOR_CATALOG
}


@dataclass(frozen=True)
class FundamentalFactorResult:
    panel: pd.DataFrame
    quality: pd.DataFrame


def build_fundamental_factor_panel(
    base_features: pd.DataFrame,
    financial_tables: dict[str, pd.DataFrame],
    *,
    max_age_days: int = 180,
    progress_callback: ProgressCallback | None = None,
) -> FundamentalFactorResult:
    """Build point-in-time fundamental factors on an existing decision-date panel."""

    if max_age_days < 1:
        raise DataQualityError("max_age_days must be positive")
    base_required = {"decision_date", "instrument", "total_mv_cny"}
    missing_base = sorted(base_required.difference(base_features.columns))
    if missing_base:
        raise DataQualityError(f"Base feature panel is missing columns: {missing_base}")
    required_tables = {
        "income": {
            "instrument",
            "report_period",
            "report_type",
            "available_date",
            "version_sequence",
            "company_type",
            "total_revenue",
            "revenue",
            "oper_cost",
            "operate_profit",
            "n_income_attr_p",
        },
        "balancesheet": {
            "instrument",
            "report_period",
            "report_type",
            "available_date",
            "version_sequence",
            "total_assets",
            "total_hldr_eqy_exc_min_int",
        },
        "cashflow": {
            "instrument",
            "report_period",
            "report_type",
            "available_date",
            "version_sequence",
            "n_cashflow_act",
            "c_pay_acq_const_fiolta",
        },
    }
    for name, columns in required_tables.items():
        if name not in financial_tables:
            raise DataQualityError(f"Missing financial table: {name}")
        missing = sorted(columns.difference(financial_tables[name].columns))
        if missing:
            raise DataQualityError(f"Financial table {name} is missing columns: {missing}")

    base = base_features.copy()
    base["decision_date"] = base["decision_date"].astype(str)
    base["instrument"] = base["instrument"].astype(str)
    if base.duplicated(["decision_date", "instrument"]).any():
        raise DataQualityError("Base feature panel has duplicate decision-date instruments")
    rows: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    required_names = tuple(sorted(required_tables))
    decision_count = int(base["decision_date"].nunique())
    progress = (
        ProgressLogger(
            LOGGER,
            stage="fundamental_factor_panel",
            total=decision_count,
            callback=progress_callback,
        )
        if decision_count
        else None
    )
    for position, (decision_date, day_base, latest) in enumerate(
        _iter_latest_fundamentals(
            base,
            financial_tables,
            required_names=required_names,
        ),
        start=1,
    ):
        day = day_base.merge(
            latest,
            on="instrument",
            how="left",
            validate="one_to_one",
        )
        day = _derive_fundamental_factors(day)
        decision_timestamp = pd.Timestamp(str(decision_date))
        report_timestamp = pd.to_datetime(
            day["financial_report_period"],
            format="%Y%m%d",
            errors="coerce",
        )
        day["financial_age_days"] = (decision_timestamp - report_timestamp).dt.days
        day["financial_stale"] = (
            day["financial_age_days"].isna()
            | day["financial_age_days"].gt(max_age_days)
            | day["financial_age_days"].lt(0)
        )
        day.loc[day["financial_stale"], list(FUNDAMENTAL_FACTOR_NAMES)] = np.nan
        for factor in FUNDAMENTAL_FACTOR_NAMES:
            day[factor] = pd.to_numeric(day[factor], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            quality_rows.append(
                {
                    "decision_date": str(decision_date),
                    "factor": factor,
                    "coverage": float(day[factor].notna().mean()),
                    "observations": int(day[factor].notna().sum()),
                    "instruments": len(day),
                    "stale_instruments": int(day["financial_stale"].sum()),
                    "latest_report_period": (
                        str(day["financial_report_period"].dropna().max())
                        if day["financial_report_period"].notna().any()
                        else None
                    ),
                    "lookahead_violations": int(
                        (
                            day["financial_available_date"].notna()
                            & day["financial_available_date"].astype(str).gt(
                                str(decision_date)
                            )
                        ).sum()
                    ),
                }
            )
        rows.append(day)
        if progress is not None:
            progress.update(
                position,
                context={"decision_date": str(decision_date)},
            )
    audit_columns = [
        "financial_report_period",
        "financial_available_date",
        "financial_company_type",
        "financial_age_days",
        "financial_stale",
    ]
    if not rows:
        result = base.copy()
        for column in [*audit_columns, *FUNDAMENTAL_FACTOR_NAMES]:
            result[column] = pd.Series(dtype="object")
        return FundamentalFactorResult(
            panel=result,
            quality=pd.DataFrame(
                columns=[
                    "decision_date",
                    "factor",
                    "coverage",
                    "observations",
                    "instruments",
                    "stale_instruments",
                    "latest_report_period",
                    "lookahead_violations",
                ]
            ),
        )
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["decision_date", "instrument"]
    )
    output_columns = [
        *base.columns,
        *[column for column in audit_columns if column not in base.columns],
        *[column for column in FUNDAMENTAL_FACTOR_NAMES if column not in base.columns],
    ]
    return FundamentalFactorResult(
        panel=result.loc[:, output_columns].reset_index(drop=True),
        quality=pd.DataFrame(quality_rows),
    )


def _iter_latest_fundamentals(
    base: pd.DataFrame,
    financial_tables: dict[str, pd.DataFrame],
    *,
    required_names: tuple[str, ...],
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Update only instruments with newly visible statements at each decision date."""

    by_instrument: dict[str, dict[str, pd.DataFrame]] = {}
    event_frames: list[pd.DataFrame] = []
    for name in required_names:
        table = financial_tables[name].copy()
        table["instrument"] = table["instrument"].astype(str)
        table["available_date"] = table["available_date"].astype("string")
        by_instrument[name] = {
            str(instrument): group.reset_index(drop=True)
            for instrument, group in table.groupby("instrument", sort=False)
        }
        events = table.loc[
            table["available_date"].notna(),
            ["available_date", "instrument"],
        ].copy()
        if not events.empty:
            event_frames.append(events)

    event_schedule: list[tuple[str, set[str]]] = []
    if event_frames:
        events = pd.concat(event_frames, ignore_index=True).drop_duplicates()
        event_schedule = [
            (str(date), set(frame["instrument"].astype(str)))
            for date, frame in events.groupby("available_date", sort=True)
        ]

    empty_tables = {
        name: financial_tables[name].iloc[0:0].copy() for name in required_names
    }
    state: dict[str, dict[str, Any]] = {}
    initialized: set[str] = set()
    pending: set[str] = set()
    event_position = 0
    latest_columns = list(_latest_period_per_instrument(_empty_period_frame()).columns)

    for decision_date, day_base in base.groupby("decision_date", sort=True):
        date = str(decision_date)
        while (
            event_position < len(event_schedule)
            and event_schedule[event_position][0] <= date
        ):
            pending.update(event_schedule[event_position][1])
            event_position += 1

        current_instruments = set(day_base["instrument"].astype(str))
        refresh = (pending & current_instruments) | (
            current_instruments - initialized
        )
        for instrument in sorted(refresh):
            visible = {
                name: select_financial_versions_asof(
                    by_instrument[name].get(instrument, empty_tables[name]),
                    date,
                )
                for name in required_names
            }
            latest = _latest_period_per_instrument(
                _build_period_fundamentals(visible)
            )
            if latest.empty:
                state.pop(instrument, None)
            else:
                state[instrument] = latest.iloc[0].to_dict()
            initialized.add(instrument)
            pending.discard(instrument)

        latest_rows = [
            state[instrument]
            for instrument in sorted(current_instruments)
            if instrument in state
        ]
        latest_frame = pd.DataFrame(latest_rows, columns=latest_columns)
        yield date, day_base, latest_frame


def _build_period_fundamentals(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    income = _cumulative_flows_to_ttm(
        tables["income"],
        metrics=(
            "total_revenue",
            "revenue",
            "oper_cost",
            "operate_profit",
            "n_income_attr_p",
        ),
        prefix="income",
        include_company_type=True,
    )
    cashflow = _cumulative_flows_to_ttm(
        tables["cashflow"],
        metrics=("n_cashflow_act", "c_pay_acq_const_fiolta"),
        prefix="cashflow",
        include_company_type=False,
    )
    balance = _balance_pairs(tables["balancesheet"])
    if income.empty or cashflow.empty or balance.empty:
        return _empty_period_frame()
    period = income.merge(
        cashflow,
        on=["instrument", "report_period", "_period_ordinal"],
        how="inner",
        validate="one_to_one",
    ).merge(
        balance,
        on=["instrument", "report_period", "_period_ordinal"],
        how="inner",
        validate="one_to_one",
    )
    if period.empty:
        return _empty_period_frame()
    period["sales_ttm"] = period["total_revenue_ttm"].combine_first(
        period["revenue_ttm"]
    )
    period["average_total_assets"] = (
        period["total_assets"] + period["total_assets_lag4q"]
    ) / 2.0
    period["average_parent_equity"] = (
        period["parent_equity"] + period["parent_equity_lag4q"]
    ) / 2.0
    period["roe_ttm_value"] = _safe_ratio(
        period["n_income_attr_p_ttm"], period["average_parent_equity"], positive=True
    )
    period["financial_available_date"] = _row_max_date(
        period,
        (
            "income_ttm_available_date",
            "cashflow_ttm_available_date",
            "balance_pair_available_date",
        ),
    )
    prior_columns = [
        "sales_ttm",
        "n_income_attr_p_ttm",
        "n_cashflow_act_ttm",
        "roe_ttm_value",
        "financial_available_date",
    ]
    prior = period[["instrument", "_period_ordinal", *prior_columns]].copy()
    prior["_period_ordinal"] += 4
    prior = prior.rename(
        columns={column: f"{column}_lag4q" for column in prior_columns}
    )
    period = period.merge(
        prior,
        on=["instrument", "_period_ordinal"],
        how="left",
        validate="one_to_one",
    )
    period["financial_available_date"] = _row_max_date(
        period,
        ("financial_available_date", "financial_available_date_lag4q"),
    )
    return period


def _cumulative_flows_to_ttm(
    table: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    prefix: str,
    include_company_type: bool,
) -> pd.DataFrame:
    output_columns = [
        "instrument",
        "report_period",
        "_period_ordinal",
        *(f"{metric}_ttm" for metric in metrics),
        f"{prefix}_ttm_available_date",
    ]
    if include_company_type:
        output_columns.append("financial_company_type")
    if table.empty:
        return pd.DataFrame(columns=output_columns)
    frame = table.loc[table["report_type"].astype(str).eq("1")].copy()
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    frame["_period_ordinal"] = _period_ordinals(frame["report_period"])
    frame["_quarter"] = pd.to_datetime(
        frame["report_period"], format="%Y%m%d", errors="coerce"
    ).dt.quarter
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["_period_ordinal", "_quarter"]).sort_values(
        ["instrument", "_period_ordinal", "available_date", "version_sequence"]
    )
    frame = frame.drop_duplicates(["instrument", "_period_ordinal"], keep="last")
    rows: list[dict[str, Any]] = []
    for instrument, group in frame.groupby("instrument", sort=False):
        group = group.sort_values("_period_ordinal").reset_index(drop=True)
        quarter_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for position, row in group.iterrows():
            current_ordinal = int(row["_period_ordinal"])
            current_quarter = int(row["_quarter"])
            previous: pd.Series | None = (
                group.iloc[position - 1] if position > 0 else None
            )
            consecutive = (
                previous is not None
                and int(previous["_period_ordinal"]) == current_ordinal - 1
            )
            for metric in metrics:
                value = row[metric]
                if current_quarter == 1:
                    quarter_value = value
                elif (
                    previous is not None
                    and consecutive
                    and int(previous["_quarter"]) == current_quarter - 1
                ):
                    previous_value = previous[metric]
                    quarter_value = (
                        value - previous_value
                        if pd.notna(value) and pd.notna(previous_value)
                        else np.nan
                    )
                else:
                    quarter_value = np.nan
                quarter_values[metric].append(quarter_value)
        for position in range(len(group)):
            if position < 3:
                continue
            window = group.iloc[position - 3 : position + 1]
            if int(window.iloc[-1]["_period_ordinal"]) - int(
                window.iloc[0]["_period_ordinal"]
            ) != 3:
                continue
            payload: dict[str, Any] = {
                "instrument": str(instrument),
                "report_period": str(group.iloc[position]["report_period"]),
                "_period_ordinal": int(group.iloc[position]["_period_ordinal"]),
                f"{prefix}_ttm_available_date": _max_date_values(
                    window["available_date"]
                ),
            }
            if include_company_type:
                payload["financial_company_type"] = str(
                    group.iloc[position]["company_type"]
                )
            for metric in metrics:
                values = pd.Series(
                    quarter_values[metric][position - 3 : position + 1],
                    dtype=float,
                )
                payload[f"{metric}_ttm"] = (
                    float(values.sum()) if values.notna().all() else np.nan
                )
            rows.append(payload)
    return pd.DataFrame(rows, columns=output_columns)


def _balance_pairs(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "instrument",
        "report_period",
        "_period_ordinal",
        "total_assets",
        "parent_equity",
        "total_assets_lag4q",
        "parent_equity_lag4q",
        "balance_pair_available_date",
    ]
    if table.empty:
        return pd.DataFrame(columns=columns)
    frame = table.loc[table["report_type"].astype(str).eq("1")].copy()
    frame["_period_ordinal"] = _period_ordinals(frame["report_period"])
    frame["total_assets"] = pd.to_numeric(frame["total_assets"], errors="coerce")
    frame["parent_equity"] = pd.to_numeric(
        frame["total_hldr_eqy_exc_min_int"], errors="coerce"
    )
    frame = frame.dropna(subset=["_period_ordinal"]).sort_values(
        ["instrument", "_period_ordinal", "available_date", "version_sequence"]
    )
    frame = frame.drop_duplicates(["instrument", "_period_ordinal"], keep="last")
    current = frame[
        [
            "instrument",
            "report_period",
            "_period_ordinal",
            "total_assets",
            "parent_equity",
            "available_date",
        ]
    ].copy()
    prior = current[
        ["instrument", "_period_ordinal", "total_assets", "parent_equity", "available_date"]
    ].copy()
    prior["_period_ordinal"] += 4
    prior = prior.rename(
        columns={
            "total_assets": "total_assets_lag4q",
            "parent_equity": "parent_equity_lag4q",
            "available_date": "prior_balance_available_date",
        }
    )
    result = current.merge(
        prior,
        on=["instrument", "_period_ordinal"],
        how="left",
        validate="one_to_one",
    )
    result["balance_pair_available_date"] = _row_max_date(
        result,
        ("available_date", "prior_balance_available_date"),
    )
    return result.loc[:, columns]


def _latest_period_per_instrument(period: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "instrument",
        "financial_report_period",
        "financial_available_date",
        "financial_company_type",
        "sales_ttm",
        "oper_cost_ttm",
        "operate_profit_ttm",
        "n_income_attr_p_ttm",
        "n_cashflow_act_ttm",
        "c_pay_acq_const_fiolta_ttm",
        "average_total_assets",
        "average_parent_equity",
        "total_assets",
        "total_assets_lag4q",
        "parent_equity",
        "sales_ttm_lag4q",
        "n_income_attr_p_ttm_lag4q",
        "n_cashflow_act_ttm_lag4q",
        "roe_ttm_value",
        "roe_ttm_value_lag4q",
    ]
    if period.empty:
        return pd.DataFrame(columns=columns)
    latest = (
        period.sort_values(["instrument", "_period_ordinal"])
        .groupby("instrument", as_index=False, sort=False)
        .tail(1)
        .rename(columns={"report_period": "financial_report_period"})
    )
    return latest.loc[:, columns].reset_index(drop=True)


def _derive_fundamental_factors(day: pd.DataFrame) -> pd.DataFrame:
    result = day.copy()
    market_cap = pd.to_numeric(result["total_mv_cny"], errors="coerce")
    general_industry = result["financial_company_type"].astype(str).eq("1")
    gross_profit = result["sales_ttm"] - result["oper_cost_ttm"]
    result["gross_profitability_ttm"] = _safe_ratio(
        gross_profit, result["average_total_assets"], positive=True
    ).where(general_industry)
    result["roe_ttm"] = result["roe_ttm_value"]
    result["roa_ttm"] = _safe_ratio(
        result["n_income_attr_p_ttm"], result["average_total_assets"], positive=True
    )
    result["operating_margin_ttm"] = _safe_ratio(
        result["operate_profit_ttm"], result["sales_ttm"], positive=True
    ).where(general_industry)
    result["cash_return_on_assets_ttm"] = _safe_ratio(
        result["n_cashflow_act_ttm"], result["average_total_assets"], positive=True
    ).where(general_industry)
    result["book_to_market"] = _safe_ratio(
        result["parent_equity"], market_cap, positive=True
    )
    result["earnings_yield_ttm"] = _safe_ratio(
        result["n_income_attr_p_ttm"], market_cap, positive=True
    )
    result["sales_to_price_ttm"] = _safe_ratio(
        result["sales_ttm"], market_cap, positive=True
    ).where(general_industry)
    result["cfo_yield_ttm"] = _safe_ratio(
        result["n_cashflow_act_ttm"], market_cap, positive=True
    ).where(general_industry)
    result["revenue_growth_ttm_yoy"] = (
        _safe_ratio(result["sales_ttm"], result["sales_ttm_lag4q"], positive=True) - 1.0
    ).where(general_industry)
    result["earnings_growth_ttm_yoy"] = _signed_growth(
        result["n_income_attr_p_ttm"], result["n_income_attr_p_ttm_lag4q"]
    )
    result["cfo_growth_ttm_yoy"] = _signed_growth(
        result["n_cashflow_act_ttm"], result["n_cashflow_act_ttm_lag4q"]
    ).where(general_industry)
    result["roe_change_yoy"] = (
        result["roe_ttm_value"] - result["roe_ttm_value_lag4q"]
    )
    result["asset_growth_yoy"] = (
        _safe_ratio(result["total_assets"], result["total_assets_lag4q"], positive=True)
        - 1.0
    ).where(general_industry)
    result["capex_intensity_ttm"] = _safe_ratio(
        result["c_pay_acq_const_fiolta_ttm"],
        result["average_total_assets"],
        positive=True,
    ).where(general_industry)
    result["total_accruals_ttm"] = _safe_ratio(
        result["n_income_attr_p_ttm"] - result["n_cashflow_act_ttm"],
        result["average_total_assets"],
        positive=True,
    ).where(general_industry)
    material_earnings = result["n_income_attr_p_ttm"].gt(
        result["average_total_assets"].abs() * 0.005
    )
    result["cash_conversion_ttm"] = _safe_ratio(
        result["n_cashflow_act_ttm"], result["n_income_attr_p_ttm"].abs()
    ).where(general_industry & material_earnings)
    return result


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    positive: bool = False,
) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce")
    right = pd.to_numeric(denominator, errors="coerce")
    valid = right.gt(1e-12) if positive else right.abs().gt(1e-12)
    return left / right.where(valid)


def _signed_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    current_values = pd.to_numeric(current, errors="coerce")
    previous_values = pd.to_numeric(previous, errors="coerce")
    denominator = previous_values.abs().where(previous_values.abs().gt(1e-12))
    return (current_values - previous_values) / denominator


def _period_ordinals(report_periods: pd.Series) -> pd.Series:
    dates = pd.to_datetime(report_periods, format="%Y%m%d", errors="coerce")
    return (dates.dt.year * 4 + dates.dt.quarter - 1).astype("Int64")


def _row_max_date(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return frame.loc[:, list(columns)].apply(_max_date_values, axis=1).astype("string")


def _max_date_values(values: pd.Series) -> Any:
    dates = [str(value) for value in values if pd.notna(value) and str(value)]
    return max(dates) if dates else pd.NA


def _empty_period_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "instrument",
            "report_period",
            "_period_ordinal",
            "financial_available_date",
            "financial_company_type",
        ]
    )
