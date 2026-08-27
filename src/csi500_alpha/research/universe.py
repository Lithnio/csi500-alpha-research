from __future__ import annotations

import pandas as pd


def benchmark_weights_asof(weights: pd.DataFrame, decision_date: str) -> pd.Series:
    """Return the latest benchmark weights known before a close-time decision."""
    eligible = weights[weights["snapshot_date"].astype(str) < str(decision_date)]
    if eligible.empty:
        return pd.Series(dtype=float, name="weight")
    snapshot = eligible["snapshot_date"].max()
    result = eligible.loc[eligible["snapshot_date"] == snapshot].set_index("instrument")["weight"]
    result.name = "weight"
    return result.sort_index()


def select_rebalance_dates(
    open_dates: list[str], *, start_date: str, end_date: str, every: int
) -> list[str]:
    eligible = [date for date in open_dates if start_date <= date <= end_date]
    return eligible[::every]

