from __future__ import annotations

import re
from typing import Any

import pandas as pd

_OPEN_MINUTE = 9 * 60 + 30
_TIME_INTERVAL = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def suspension_blocks_open(suspend_timing: Any) -> bool:
    """Return whether a suspension record prevents a 09:30 execution.

    Tushare leaves ``suspend_timing`` empty for a full-day suspension.  For an
    intraday record, only an interval covering the market open blocks an order
    whose execution contract is the next session's open.  Unparseable non-empty
    values are treated conservatively as blocking.
    """

    if suspend_timing is None or bool(pd.isna(suspend_timing)):
        return True
    text = str(suspend_timing).strip()
    if not text:
        return True
    intervals = _TIME_INTERVAL.findall(text)
    if not intervals:
        return True
    for start_hour, start_minute, end_hour, end_minute in intervals:
        start = int(start_hour) * 60 + int(start_minute)
        end = int(end_hour) * 60 + int(end_minute)
        if start <= _OPEN_MINUTE < end:
            return True
    return False


def opening_suspensions_by_date(
    suspensions: pd.DataFrame | None,
) -> dict[str, set[str]]:
    """Index full-day and opening suspensions by date.

    Resumption rows can safely coexist in the source table; only ``S`` records
    participate in the restriction.
    """

    if suspensions is None or suspensions.empty:
        return {}
    required = {"trade_date", "instrument"}
    missing = sorted(required.difference(suspensions.columns))
    if missing:
        raise ValueError(f"Suspension table is missing columns: {missing}")
    frame = suspensions.copy()
    if "suspend_type" in frame:
        frame = frame[frame["suspend_type"].astype(str).eq("S")]
    if frame.empty:
        return {}
    if "suspend_timing" not in frame:
        frame["suspend_timing"] = None
    frame = frame[frame["suspend_timing"].map(suspension_blocks_open)]
    return {
        str(date): set(group["instrument"].astype(str))
        for date, group in frame.groupby("trade_date", sort=True)
    }
