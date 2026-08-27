from __future__ import annotations

import numpy as np
import pandas as pd

from csi500_alpha.execution.tradeability import opening_suspensions_by_date


def build_forward_labels(
    *,
    features: pd.DataFrame,
    market_panel: pd.DataFrame,
    index_bars: pd.DataFrame,
    open_dates: list[str],
    horizon: int,
    suspensions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build strict scheduled-open active-return labels.

    A label is valid only when both the scheduled entry and exit opens have a
    price and are executable under the same opening-suspension rule used by the
    event backtester.  Invalid observations remain in the output with an
    explicit status instead of being silently discarded.
    """
    dates = sorted(str(date) for date in open_dates)
    stock_open = market_panel.pivot(
        index="trade_date",
        columns="instrument",
        values="adjusted_open",
    ).reindex(dates)
    index_open = index_bars.set_index("trade_date")["open"].reindex(dates)
    entry_stock = stock_open.shift(-1)
    exit_stock = stock_open.shift(-(horizon + 1))
    stock_return = exit_stock / entry_stock - 1.0
    index_return = index_open.shift(-(horizon + 1)) / index_open.shift(-1) - 1.0
    opening_suspensions = opening_suspensions_by_date(suspensions)

    date_to_position = {date: position for position, date in enumerate(dates)}
    rows: list[pd.DataFrame] = []
    for decision_date, day_features in features.groupby("decision_date", sort=True):
        decision_date = str(decision_date)
        position = date_to_position.get(decision_date)
        instruments = day_features["instrument"].astype(str)
        if position is None or position + horizon + 1 >= len(dates):
            entry_date = None
            exit_date = None
            stock_values = pd.Series(np.nan, index=instruments)
            benchmark_value = np.nan
            entry_tradeable = pd.Series(False, index=instruments)
            exit_tradeable = pd.Series(False, index=instruments)
            label_status = pd.Series("outside_horizon", index=instruments, dtype="object")
        else:
            entry_date = dates[position + 1]
            exit_date = dates[position + horizon + 1]
            stock_values = stock_return.loc[decision_date].reindex(instruments)
            benchmark_value = float(index_return.loc[decision_date])
            entry_prices = entry_stock.loc[decision_date].reindex(instruments)
            exit_prices = exit_stock.loc[decision_date].reindex(instruments)
            entry_blocked = pd.Series(
                instruments.isin(opening_suspensions.get(entry_date, set())).to_numpy(),
                index=instruments,
            )
            exit_blocked = pd.Series(
                instruments.isin(opening_suspensions.get(exit_date, set())).to_numpy(),
                index=instruments,
            )
            entry_tradeable = entry_prices.notna() & ~entry_blocked
            exit_tradeable = exit_prices.notna() & ~exit_blocked
            label_status = _label_status(
                entry_prices=entry_prices,
                exit_prices=exit_prices,
                entry_blocked=entry_blocked,
                exit_blocked=exit_blocked,
            )
            stock_values = stock_values.where(entry_tradeable & exit_tradeable)
        day = pd.DataFrame(
            {
                "decision_date": decision_date,
                "instrument": instruments.to_numpy(),
                "label_entry_date": entry_date,
                "label_end_date": exit_date,
                "label_available_date": exit_date,
                "entry_tradeable": entry_tradeable.to_numpy(dtype=bool),
                "exit_tradeable": exit_tradeable.to_numpy(dtype=bool),
                "label_status": label_status.to_numpy(dtype=object),
                "forward_stock_return": stock_values.to_numpy(),
                "forward_benchmark_return": benchmark_value,
            }
        )
        day["forward_active_return"] = (
            day["forward_stock_return"] - day["forward_benchmark_return"]
        )
        day["label_valid"] = day["label_status"].eq("valid")
        rows.append(day)
    if not rows:
        return pd.DataFrame(
            columns=[
                "decision_date",
                "instrument",
                "label_entry_date",
                "label_end_date",
                "label_available_date",
                "entry_tradeable",
                "exit_tradeable",
                "label_status",
                "label_valid",
                "forward_stock_return",
                "forward_benchmark_return",
                "forward_active_return",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["decision_date", "instrument"]
    ).reset_index(drop=True)


def _label_status(
    *,
    entry_prices: pd.Series,
    exit_prices: pd.Series,
    entry_blocked: pd.Series,
    exit_blocked: pd.Series,
) -> pd.Series:
    status = pd.Series("valid", index=entry_prices.index, dtype="object")
    missing_entry = entry_prices.isna()
    missing_exit = exit_prices.isna()
    status.loc[missing_entry & missing_exit] = "missing_entry_and_exit_price"
    status.loc[missing_entry & ~missing_exit] = "missing_entry_price"
    status.loc[~missing_entry & missing_exit] = "missing_exit_price"
    status.loc[~missing_entry & ~missing_exit & entry_blocked] = "entry_suspended_at_open"
    status.loc[
        ~missing_entry & ~missing_exit & ~entry_blocked & exit_blocked
    ] = "exit_suspended_at_open"
    return status
