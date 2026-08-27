from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from csi500_alpha.config import RiskSettings


@dataclass(frozen=True)
class RiskEstimate:
    as_of_date: str
    covariance: pd.DataFrame
    eligible: pd.Series
    method: str
    observations: int


class LedoitWolfRiskModel:
    """Estimate a daily covariance matrix using information available by ``as_of``."""

    def __init__(
        self,
        settings: RiskSettings,
        market_panel: pd.DataFrame,
        open_dates: list[str],
    ) -> None:
        self.settings = settings
        self.open_dates = sorted(str(date) for date in open_dates)
        required = {"trade_date", "instrument", "adjusted_close"}
        missing = required.difference(market_panel.columns)
        if missing:
            raise ValueError(f"Risk model market panel is missing columns: {sorted(missing)}")
        panel = market_panel.loc[:, sorted(required)].copy()
        panel["trade_date"] = panel["trade_date"].astype(str)
        self.close = panel.pivot(
            index="trade_date",
            columns="instrument",
            values="adjusted_close",
        ).sort_index()

    def estimate(self, as_of_date: str, instruments: list[str]) -> RiskEstimate:
        names = pd.Index(sorted(set(instruments)), name="instrument")
        if names.empty:
            raise ValueError("Risk estimation requires at least one instrument")
        available_dates = [date for date in self.open_dates if date <= str(as_of_date)]
        window_dates = available_dates[-(self.settings.lookback + 1) :]
        raw_prices = self.close.reindex(index=window_dates, columns=names)
        observed = raw_prices.notna().sum()
        prices = raw_prices.ffill()
        returns = prices.pct_change(fill_method=None).iloc[1:]
        returns = returns.clip(-self.settings.return_clip, self.settings.return_clip)
        eligible = observed.ge(self.settings.min_history + 1)
        eligible &= returns.notna().sum().ge(self.settings.min_history)

        missing_daily_variance = (
            self.settings.missing_annual_volatility**2 / self.settings.annualization
        )
        covariance = np.eye(len(names), dtype=float) * missing_daily_variance
        eligible_names = names[eligible.reindex(names, fill_value=False).to_numpy()]
        method = "high_variance_diagonal"

        if len(returns) >= 2 and len(eligible_names) >= 2:
            matrix = returns.loc[:, eligible_names].fillna(0.0).to_numpy(dtype=float)
            estimate = LedoitWolf(assume_centered=False).fit(matrix).covariance_
            positions = names.get_indexer(eligible_names)
            covariance[np.ix_(positions, positions)] = estimate
            method = "ledoit_wolf"
        elif len(eligible_names) >= 1:
            variances = (
                returns.loc[:, eligible_names]
                .fillna(0.0)
                .var(axis=0, ddof=1)
                .clip(lower=self.settings.variance_floor)
            )
            positions = names.get_indexer(eligible_names)
            covariance[positions, positions] = variances.to_numpy(dtype=float)
            method = "sample_diagonal"

        covariance = self._repair_covariance(covariance)
        covariance_frame = pd.DataFrame(covariance, index=names, columns=names)
        eligibility = eligible.reindex(names, fill_value=False).astype(bool)
        eligibility.name = "risk_eligible"
        return RiskEstimate(
            as_of_date=str(as_of_date),
            covariance=covariance_frame,
            eligible=eligibility,
            method=method,
            observations=len(returns),
        )

    def _repair_covariance(self, covariance: np.ndarray) -> np.ndarray:
        result = np.asarray(covariance, dtype=float)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = (result + result.T) / 2.0
        minimum = float(np.linalg.eigvalsh(result).min())
        if minimum < self.settings.variance_floor:
            result += np.eye(len(result)) * (self.settings.variance_floor - minimum)
        return result
