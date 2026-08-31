from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from csi500_alpha.data.benchmark import active_membership_asof


@dataclass(frozen=True)
class BenchmarkWeightState:
    weights: pd.Series
    snapshot_date: str | None
    weight_source: str
    proxy_instruments: tuple[str, ...]
    membership_event_ids: tuple[str, ...]
    membership_sources: tuple[str, ...]


def benchmark_weight_state_asof(
    weights: pd.DataFrame,
    decision_date: str,
    membership_intervals: pd.DataFrame | None = None,
) -> BenchmarkWeightState:
    """Return effective-date membership with point-in-time benchmark weights.

    A monthly Tushare snapshot is available only after its snapshot date.  When
    an announced constituent change is already effective but the confirming
    month-end snapshot is not yet available, the entering members receive the
    frozen proxy weight recorded in their membership interval.
    """

    eligible = weights[weights["snapshot_date"].astype(str) < str(decision_date)]
    if eligible.empty:
        return BenchmarkWeightState(
            weights=pd.Series(dtype=float, name="weight"),
            snapshot_date=None,
            weight_source="unavailable",
            proxy_instruments=(),
            membership_event_ids=(),
            membership_sources=(),
        )
    snapshot = eligible["snapshot_date"].max()
    snapshot_weights = (
        eligible.loc[eligible["snapshot_date"] == snapshot]
        .set_index("instrument")["weight"]
        .pipe(pd.to_numeric, errors="coerce")
    )
    if membership_intervals is None or membership_intervals.empty:
        result = snapshot_weights.dropna().astype(float)
        result = result / result.sum()
        result.name = "weight"
        return BenchmarkWeightState(
            weights=result.sort_index(),
            snapshot_date=str(snapshot),
            weight_source="tushare_snapshot_legacy_membership",
            proxy_instruments=(),
            membership_event_ids=(),
            membership_sources=(),
        )

    active = active_membership_asof(membership_intervals, str(decision_date))
    if active.empty:
        return BenchmarkWeightState(
            weights=pd.Series(dtype=float, name="weight"),
            snapshot_date=str(snapshot),
            weight_source="membership_unavailable",
            proxy_instruments=(),
            membership_event_ids=(),
            membership_sources=(),
        )
    if active["instrument"].duplicated().any():
        raise ValueError("Effective benchmark membership is not unique")

    active = active.set_index("instrument")
    result = snapshot_weights.reindex(active.index)
    missing = result.isna()
    proxy = pd.to_numeric(active["entry_weight_proxy"], errors="coerce")
    result.loc[missing] = proxy.loc[missing]
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        missing_names = sorted(result.index[result.isna()].astype(str))
        raise ValueError(
            f"Effective benchmark members lack point-in-time weights: {missing_names[:10]}"
        )
    if (result < 0).any() or float(result.sum()) <= 0:
        raise ValueError("Effective benchmark weights must be nonnegative with positive total")
    result = result.astype(float) / float(result.sum())
    result.name = "weight"
    proxy_instruments = tuple(sorted(result.index[missing].astype(str)))
    return BenchmarkWeightState(
        weights=result.sort_index(),
        snapshot_date=str(snapshot),
        weight_source=(
            "tushare_snapshot_with_event_proxy"
            if proxy_instruments
            else "tushare_snapshot"
        ),
        proxy_instruments=proxy_instruments,
        membership_event_ids=tuple(sorted(active["entry_event_id"].astype(str).unique())),
        membership_sources=tuple(sorted(active["entry_source"].astype(str).unique())),
    )


def benchmark_weights_asof(
    weights: pd.DataFrame,
    decision_date: str,
    membership_intervals: pd.DataFrame | None = None,
) -> pd.Series:
    """Return benchmark weights effective on a close-time decision date."""

    return benchmark_weight_state_asof(
        weights,
        decision_date,
        membership_intervals,
    ).weights


def select_rebalance_dates(
    open_dates: list[str], *, start_date: str, end_date: str, every: int
) -> list[str]:
    eligible = [date for date in open_dates if start_date <= date <= end_date]
    return eligible[::every]
