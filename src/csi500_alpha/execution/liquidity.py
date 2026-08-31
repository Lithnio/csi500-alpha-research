from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIQUIDITY_CONTRACT_VERSION = "close-known-trailing-adv-v1"


@dataclass(frozen=True)
class LiquiditySnapshot:
    """Trailing liquidity inputs frozen after one decision-day close."""

    reference_date: str
    adv_cny: pd.Series
    observation_count: pd.Series

    def max_trade_weights(
        self,
        *,
        portfolio_aum_cny: float,
        max_adv_participation: float,
    ) -> pd.Series:
        if portfolio_aum_cny <= 0:
            raise ValueError("Portfolio AUM must be positive for liquidity caps")
        if not 0 < max_adv_participation <= 1:
            raise ValueError("ADV participation must be in (0, 1]")
        result = (
            pd.to_numeric(self.adv_cny, errors="coerce")
            * float(max_adv_participation)
            / float(portfolio_aum_cny)
        )
        result.name = "max_trade_weight"
        return result


def build_trailing_adv_snapshots(
    market_panel: pd.DataFrame,
    *,
    lookback: int,
    min_observations: int,
) -> dict[str, LiquiditySnapshot]:
    """Build point-in-time ADV snapshots using data through each close.

    A snapshot dated ``t`` may be used for an order submitted at the next open.
    Later rows cannot affect an earlier snapshot.
    """

    required = {"trade_date", "instrument", "amount_cny"}
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError(f"Liquidity snapshots require columns: {missing}")
    if lookback < 1 or not 1 <= min_observations <= lookback:
        raise ValueError("Liquidity lookback requires 1 <= min_observations <= lookback")
    if market_panel.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("Liquidity trade_date/instrument key is not unique")

    ordered = market_panel[["trade_date", "instrument", "amount_cny"]].copy()
    ordered["trade_date"] = ordered["trade_date"].astype(str)
    ordered["instrument"] = ordered["instrument"].astype(str)
    amount = pd.to_numeric(ordered["amount_cny"], errors="coerce")
    ordered["valid_amount_cny"] = amount.where(np.isfinite(amount) & (amount > 0))
    ordered = ordered.sort_values(["instrument", "trade_date"])
    grouped = ordered.groupby("instrument", sort=False)["valid_amount_cny"]
    ordered["adv_cny"] = grouped.transform(
        lambda values: values.rolling(
            lookback,
            min_periods=min_observations,
        ).mean()
    )
    ordered["adv_observations"] = grouped.transform(
        lambda values: values.rolling(lookback, min_periods=1).count()
    ).astype(int)

    snapshots: dict[str, LiquiditySnapshot] = {}
    for date, frame in ordered.groupby("trade_date", sort=True):
        indexed = frame.set_index("instrument")
        snapshots[str(date)] = LiquiditySnapshot(
            reference_date=str(date),
            adv_cny=indexed["adv_cny"].astype(float).copy(),
            observation_count=indexed["adv_observations"].astype(int).copy(),
        )
    return snapshots
