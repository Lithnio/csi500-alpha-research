from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _concat(frames: Iterable[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
    materialized = [frame for frame in frames if not frame.empty]
    if not materialized:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(materialized, ignore_index=True)


def _dates(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def normalize_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={"cal_date": "trade_date"}).copy()
    result["trade_date"] = _dates(result["trade_date"])
    result["prev_trade_date"] = _dates(result["pretrade_date"])
    result["is_open"] = pd.to_numeric(result["is_open"], errors="raise").astype("int8")
    return (
        result.loc[:, ["trade_date", "is_open", "prev_trade_date"]]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def normalize_weights(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raw = _concat(
        frames,
        ("index_code", "con_code", "trade_date", "weight"),
    ).rename(
        columns={
            "con_code": "instrument",
            "trade_date": "snapshot_date",
            "weight": "weight_pct",
        }
    )
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "index_code",
                "snapshot_date",
                "available_date_rule",
                "instrument",
                "weight_pct",
                "weight",
            ]
        )
    raw["snapshot_date"] = _dates(raw["snapshot_date"])
    raw["weight_pct"] = pd.to_numeric(raw["weight_pct"], errors="raise")
    raw = raw.drop_duplicates(["snapshot_date", "instrument"], keep="last")
    totals = raw.groupby("snapshot_date")["weight_pct"].transform("sum")
    raw["weight"] = raw["weight_pct"] / totals
    raw["available_date_rule"] = "next_trade_date"
    return raw.sort_values(["snapshot_date", "instrument"]).reset_index(drop=True)


def normalize_index_bars(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = _dates(result["trade_date"])
    numeric = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.rename(columns={"ts_code": "index_code"})
    return (
        result.drop_duplicates(["trade_date", "index_code"], keep="last")
        .sort_values(["trade_date", "index_code"])
        .reset_index(drop=True)
    )


def normalize_stock_bars(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raw = _concat(
        frames,
        ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"),
    ).rename(columns={"ts_code": "instrument"})
    if raw.empty:
        return raw
    raw["trade_date"] = _dates(raw["trade_date"])
    for column in ("open", "high", "low", "close", "pre_close", "vol", "amount"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["volume_shares"] = raw["vol"] * 100.0
    raw["amount_cny"] = raw["amount"] * 1000.0
    return (
        raw.drop(columns=["vol", "amount"])
        .drop_duplicates(["trade_date", "instrument"], keep="last")
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def normalize_adjustments(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raw = _concat(frames, ("ts_code", "trade_date", "adj_factor")).rename(
        columns={"ts_code": "instrument"}
    )
    if raw.empty:
        return raw
    raw["trade_date"] = _dates(raw["trade_date"])
    raw["adj_factor"] = pd.to_numeric(raw["adj_factor"], errors="coerce")
    return (
        raw.drop_duplicates(["trade_date", "instrument"], keep="last")
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def normalize_limits(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raw = _concat(frames, ("ts_code", "trade_date", "up_limit", "down_limit")).rename(
        columns={"ts_code": "instrument"}
    )
    if raw.empty:
        return raw
    raw["trade_date"] = _dates(raw["trade_date"])
    for column in ("up_limit", "down_limit"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return (
        raw.drop_duplicates(["trade_date", "instrument"], keep="last")
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def normalize_daily_basic(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = (
        "ts_code",
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "pb",
        "total_mv",
        "circ_mv",
    )
    raw = _concat(frames, columns).rename(columns={"ts_code": "instrument"})
    if raw.empty:
        return raw
    raw["trade_date"] = _dates(raw["trade_date"])
    for column in columns[2:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["total_mv_cny"] = raw["total_mv"] * 10000.0
    raw["circ_mv_cny"] = raw["circ_mv"] * 10000.0
    return (
        raw.drop(columns=["total_mv", "circ_mv"])
        .drop_duplicates(["trade_date", "instrument"], keep="last")
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def normalize_suspensions(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raw = _concat(
        frames,
        ("ts_code", "trade_date", "suspend_timing", "suspend_type"),
    ).rename(
        columns={"ts_code": "instrument"}
    )
    if raw.empty:
        return raw
    raw["trade_date"] = _dates(raw["trade_date"])
    return (
        raw.drop_duplicates(["trade_date", "instrument", "suspend_type"], keep="last")
        .sort_values(["trade_date", "instrument"])
        .reset_index(drop=True)
    )


def normalize_name_history(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = (
        "ts_code",
        "name",
        "start_date",
        "end_date",
        "ann_date",
        "change_reason",
    )
    raw = _concat(frames, columns).rename(
        columns={"ts_code": "instrument", "ann_date": "announcement_date"}
    )
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "instrument",
                "name",
                "start_date",
                "end_date",
                "announcement_date",
                "change_reason",
                "is_st",
            ]
        )
    raw_rows = len(raw)
    for column in ("start_date", "end_date", "announcement_date"):
        raw[column] = _dates(raw[column]).replace(
            {"nan": None, "None": None, "NaT": None, "": None}
        )
    raw["name"] = raw["name"].fillna("").astype(str).str.strip()
    raw["is_st"] = raw["name"].str.upper().str.contains("ST", regex=False)
    keys = ["instrument", "start_date", "end_date", "name"]
    raw = raw.drop_duplicates(keys, keep="last").copy()
    exact_duplicates_removed = raw_rows - len(raw)

    # Tushare can retain both a proposed and a later implemented effective date
    # for the same announced name event, with both rows left open-ended. Treat
    # these as revisions of one event and keep the latest effective start. Any
    # other interval overlap remains a hard eligibility quality failure.
    event_keys = ["instrument", "name", "announcement_date", "change_reason"]
    open_event = (
        raw["end_date"].isna()
        & raw["start_date"].notna()
        & raw["announcement_date"].notna()
    )
    revised_event = open_event & raw.duplicated(event_keys, keep=False)
    superseded = pd.Series(False, index=raw.index)
    if revised_event.any():
        latest_start = raw.loc[revised_event].groupby(
            event_keys,
            dropna=False,
        )["start_date"].transform("max")
        superseded.loc[revised_event] = raw.loc[revised_event, "start_date"].ne(
            latest_start
        )
    superseded_open_events_removed = int(superseded.sum())
    result = raw.loc[~superseded].sort_values(keys).reset_index(drop=True)
    result.attrs["normalization"] = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "superseded_open_name_events_removed": superseded_open_events_removed,
    }
    return result


def normalize_instrument_master(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = (
        "ts_code",
        "symbol",
        "name",
        "market",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
    )
    raw = _concat(frames, columns).rename(columns={"ts_code": "instrument"})
    if raw.empty:
        return raw
    raw["list_date"] = _dates(raw["list_date"])
    raw["delist_date"] = _dates(raw["delist_date"]).replace({"nan": None, "None": None})
    return (
        raw.drop_duplicates("instrument", keep="last")
        .sort_values("instrument")
        .reset_index(drop=True)
    )


def normalize_industry_classification(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = (
        "taxonomy",
        "index_code",
        "industry_name",
        "parent_code",
        "level",
        "industry_code",
        "is_pub",
        "src",
    )
    raw = _concat(frames, columns)
    if raw.empty:
        return raw
    return (
        raw.drop_duplicates(["taxonomy", "index_code"], keep="last")
        .sort_values(["taxonomy", "level", "index_code"])
        .reset_index(drop=True)
    )


def normalize_industry_membership(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    columns = (
        "taxonomy",
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "name",
        "in_date",
        "out_date",
        "is_new",
    )
    raw = _concat(frames, columns).rename(columns={"ts_code": "instrument"})
    if raw.empty:
        return raw
    raw["in_date"] = _dates(raw["in_date"])
    raw["out_date"] = _dates(raw["out_date"]).replace({"nan": None, "None": None})
    keys = ["taxonomy", "l1_code", "instrument", "in_date", "out_date"]
    raw = raw.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    return _reconcile_industry_intervals(raw)


def _reconcile_industry_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, group in raw.groupby(["taxonomy", "instrument"], sort=False):
        group = group.copy()
        group["raw_in_date"] = group["in_date"]
        group["raw_out_date"] = group["out_date"]
        historical = group[group["out_date"].notna()].sort_values(
            ["out_date", "in_date"]
        )
        historical = historical.drop_duplicates("out_date", keep="last")
        cursor: str | None = None
        first_start = str(group["in_date"].min())
        for _, row in historical.iterrows():
            end = str(row["out_date"])
            start = first_start if cursor is None else cursor
            if end <= start:
                continue
            record = row.to_dict()
            record["in_date"] = start
            record["out_date"] = end
            record["interval_source"] = "ordered_exit_events"
            records.append(record)
            cursor = end

        current = group[group["out_date"].isna()].sort_values("in_date")
        current = current.drop_duplicates("l1_code", keep="first")
        candidates: list[pd.Series] = []
        if cursor is None:
            candidates = [row for _, row in current.iterrows()]
        else:
            baseline = current[current["in_date"].astype(str) <= cursor]
            if not baseline.empty:
                candidates.append(baseline.iloc[-1])
            candidates.extend(
                row
                for _, row in current[current["in_date"].astype(str) > cursor].iterrows()
            )

        compact: list[pd.Series] = []
        for row in candidates:
            if compact and str(compact[-1]["l1_code"]) == str(row["l1_code"]):
                continue
            compact.append(row)
        for position, row in enumerate(compact):
            raw_start = str(row["in_date"])
            if cursor is not None and position == 0 and raw_start <= cursor:
                start = cursor
            else:
                start = raw_start
            current_end: str | None = (
                str(compact[position + 1]["in_date"])
                if position + 1 < len(compact)
                else None
            )
            if current_end is not None and current_end <= start:
                continue
            record = row.to_dict()
            record["in_date"] = start
            record["out_date"] = current_end
            record["interval_source"] = "current_transition_sequence"
            records.append(record)

    if not records:
        columns = [*raw.columns, "raw_in_date", "raw_out_date", "interval_source"]
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(records)
    keys = ["taxonomy", "instrument", "in_date", "out_date", "l1_code"]
    return result.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def build_market_panel(
    stock_bars: pd.DataFrame,
    adjustments: pd.DataFrame,
    limits: pd.DataFrame,
) -> pd.DataFrame:
    panel = stock_bars.merge(
        adjustments,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        limits,
        on=["trade_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    panel["adjusted_open"] = panel["open"] * panel["adj_factor"]
    panel["adjusted_close"] = panel["close"] * panel["adj_factor"]
    finite_limits = np.isfinite(panel["up_limit"]) & np.isfinite(panel["down_limit"])
    panel["price_limit_applicable"] = (
        finite_limits
        & (panel["open"] <= panel["up_limit"] * (1.0 + 1e-8))
        & (panel["open"] >= panel["down_limit"] * (1.0 - 1e-8))
    )
    panel["is_valid_bar"] = np.isfinite(panel["adjusted_close"]) & (panel["adjusted_close"] > 0)
    return panel.sort_values(["trade_date", "instrument"]).reset_index(drop=True)
