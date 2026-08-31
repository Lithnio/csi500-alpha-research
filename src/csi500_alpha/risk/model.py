from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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
    market_beta: pd.Series | None = None
    beta_method: str = "feature_beta_60"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RiskModel(Protocol):
    """Point-in-time covariance and beta estimator used by the optimizer."""

    def estimate(self, as_of_date: str, instruments: list[str]) -> RiskEstimate: ...


class _MarketHistory:
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

    def _return_window(
        self,
        as_of_date: str,
        names: pd.Index,
        *,
        lookback: int | None = None,
        min_history: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        window = self.settings.lookback if lookback is None else int(lookback)
        required = self.settings.min_history if min_history is None else int(min_history)
        available_dates = [date for date in self.open_dates if date <= str(as_of_date)]
        window_dates = available_dates[-(window + 1) :]
        raw_prices = self.close.reindex(index=window_dates, columns=names)
        observed = raw_prices.notna().sum()
        prices = raw_prices.ffill()
        returns = prices.pct_change(fill_method=None).iloc[1:]
        returns = returns.clip(-self.settings.return_clip, self.settings.return_clip)
        eligible = observed.ge(required + 1)
        eligible &= returns.notna().sum().ge(required)
        return returns, eligible.reindex(names, fill_value=False).astype(bool)

    def _repair_covariance(self, covariance: np.ndarray) -> np.ndarray:
        result = np.asarray(covariance, dtype=float)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = (result + result.T) / 2.0
        minimum = float(np.linalg.eigvalsh(result).min())
        if minimum < self.settings.variance_floor:
            result += np.eye(len(result)) * (self.settings.variance_floor - minimum)
        return result


class LedoitWolfRiskModel(_MarketHistory):
    """Estimate a daily covariance matrix using information available by ``as_of``."""

    def __init__(
        self,
        settings: RiskSettings,
        market_panel: pd.DataFrame,
        open_dates: list[str],
        *,
        index_bars: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(settings, market_panel, open_dates)
        self.benchmark_close = (
            _benchmark_close(index_bars)
            if settings.beta_model == "ewma_shrunk"
            else None
        )
        if settings.beta_model == "ewma_shrunk" and self.benchmark_close is None:
            raise ValueError("ewma_shrunk beta model requires index_bars")

    def estimate(self, as_of_date: str, instruments: list[str]) -> RiskEstimate:
        names = pd.Index(sorted(set(instruments)), name="instrument")
        if names.empty:
            raise ValueError("Risk estimation requires at least one instrument")
        returns, eligible = self._return_window(as_of_date, names)

        missing_daily_variance = (
            self.settings.missing_annual_volatility**2 / self.settings.annualization
        )
        covariance = np.eye(len(names), dtype=float) * missing_daily_variance
        eligible_names = names[eligible.to_numpy()]
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
        eligibility = eligible.rename("risk_eligible")
        market_beta: pd.Series | None = None
        beta_method = "feature_beta_60"
        beta_observed_fraction: float | None = None
        if self.settings.beta_model == "ewma_shrunk":
            if self.benchmark_close is None:
                raise AssertionError("EWMA beta requires initialized benchmark prices")
            market_beta, beta_observed = _estimate_ewma_beta(
                settings=self.settings,
                close=self.close,
                benchmark_close=self.benchmark_close,
                open_dates=self.open_dates,
                as_of_date=str(as_of_date),
                names=names,
            )
            beta_method = "ewma_shrunk_to_one"
            beta_observed_fraction = float(beta_observed.mean())
        return RiskEstimate(
            as_of_date=str(as_of_date),
            covariance=covariance_frame,
            eligible=eligibility,
            method=method,
            observations=len(returns),
            market_beta=market_beta,
            beta_method=beta_method,
            diagnostics={
                "factor_count": 0,
                "factor_return_observations": 0,
                "factor_model_fallback": False,
                **(
                    _beta_diagnostics(
                        market_beta,
                        beta_observed,
                        self.settings,
                    )
                    if market_beta is not None
                    else {"beta_observed_fraction": beta_observed_fraction}
                ),
            },
        )


class FactorEWMARiskModel(_MarketHistory):
    """Fundamental factor covariance with EWMA factor and specific risk.

    The model uses only information available on or before ``as_of_date``. Current
    point-in-time market, industry and style exposures project trailing daily stock
    returns into factor returns. Factor covariance and residual variances are then
    estimated with separate exponential half-lives and explicit shrinkage.
    """

    def __init__(
        self,
        settings: RiskSettings,
        market_panel: pd.DataFrame,
        open_dates: list[str],
        *,
        index_bars: pd.DataFrame,
        industry_exposures: pd.DataFrame,
        style_exposures: pd.DataFrame,
    ) -> None:
        super().__init__(settings, market_panel, open_dates)
        self.benchmark_close = _benchmark_close(index_bars)
        if self.benchmark_close is None:
            raise ValueError("Factor risk model requires index_bars")
        self.industry_by_date = self._exposure_frames(
            industry_exposures,
            label="industry",
        )
        self.style_by_date = self._exposure_frames(
            style_exposures,
            label="style",
        )
        self.fallback = LedoitWolfRiskModel(
            settings,
            market_panel,
            open_dates,
            index_bars=index_bars,
        )

    @staticmethod
    def _exposure_frames(
        frame: pd.DataFrame,
        *,
        label: str,
    ) -> dict[str, pd.DataFrame]:
        required = {"trade_date", "instrument"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Factor risk {label} exposures are missing columns: {sorted(missing)}"
            )
        if frame.empty:
            raise ValueError(f"Factor risk {label} exposures must be nonempty")
        if frame.duplicated(["trade_date", "instrument"]).any():
            raise ValueError(
                f"Factor risk {label} exposure date/instrument key is not unique"
            )
        source = frame.copy()
        source["trade_date"] = source["trade_date"].astype(str)
        return {
            str(date): group.drop(columns=["trade_date"]).set_index("instrument")
            for date, group in source.groupby("trade_date", sort=True)
        }

    @staticmethod
    def _asof_frame(
        frames: dict[str, pd.DataFrame],
        as_of_date: str,
    ) -> pd.DataFrame | None:
        available = [date for date in frames if date <= str(as_of_date)]
        if not available:
            return None
        return frames[max(available)]

    def estimate(self, as_of_date: str, instruments: list[str]) -> RiskEstimate:
        names = pd.Index(sorted(set(instruments)), name="instrument")
        if names.empty:
            raise ValueError("Risk estimation requires at least one instrument")
        returns, history_eligible = self._return_window(as_of_date, names)
        beta, beta_observed = self._estimate_market_beta(as_of_date, names)
        exposure_result = self._current_exposures(as_of_date, names, beta)
        if exposure_result is None:
            return self._fallback_estimate(
                as_of_date,
                names,
                beta,
                beta_observed,
                reason="missing_point_in_time_exposures",
            )
        exposures, exposure_diagnostics = exposure_result
        eligible_names = names[history_eligible.to_numpy()]
        minimum_cross_section = max(
            self.settings.min_factor_cross_section,
            exposures.shape[1] + 2,
        )
        if len(eligible_names) < minimum_cross_section or len(returns) < 2:
            return self._fallback_estimate(
                as_of_date,
                names,
                beta,
                beta_observed,
                reason="insufficient_factor_cross_section",
                diagnostics=exposure_diagnostics,
            )

        x = exposures.loc[eligible_names].to_numpy(dtype=float)
        stock_returns = returns.loc[:, eligible_names].fillna(0.0).to_numpy(dtype=float)
        gram = x.T @ x
        ridge = np.eye(gram.shape[0], dtype=float) * self.settings.factor_ridge
        projection = x @ np.linalg.pinv(gram + ridge)
        factor_returns = stock_returns @ projection
        fitted = factor_returns @ x.T
        residuals = stock_returns - fitted

        factor_covariance = _ewma_covariance(
            factor_returns,
            half_life=self.settings.factor_half_life,
        )
        diagonal = np.diag(np.diag(factor_covariance))
        shrinkage = self.settings.factor_covariance_shrinkage
        factor_covariance = (1.0 - shrinkage) * factor_covariance + shrinkage * diagonal
        factor_covariance = self._repair_covariance(factor_covariance)

        residual_covariance = _ewma_covariance(
            residuals,
            half_life=self.settings.specific_half_life,
        )
        specific_variance = np.diag(residual_covariance).clip(
            min=self.settings.variance_floor
        )
        finite_specific = specific_variance[np.isfinite(specific_variance)]
        specific_target = (
            float(np.median(finite_specific))
            if len(finite_specific)
            else self.settings.variance_floor
        )
        specific_shrinkage = self.settings.specific_variance_shrinkage
        specific_variance = (
            (1.0 - specific_shrinkage) * specific_variance
            + specific_shrinkage * specific_target
        ).clip(min=self.settings.variance_floor)

        modeled = x @ factor_covariance @ x.T + np.diag(specific_variance)
        modeled = self._repair_covariance(modeled)
        missing_daily_variance = (
            self.settings.missing_annual_volatility**2 / self.settings.annualization
        )
        covariance = np.eye(len(names), dtype=float) * missing_daily_variance
        positions = names.get_indexer(eligible_names)
        covariance[np.ix_(positions, positions)] = modeled
        covariance = self._repair_covariance(covariance)
        covariance_frame = pd.DataFrame(covariance, index=names, columns=names)

        condition_number = float(np.linalg.cond(factor_covariance))
        diagnostics: dict[str, Any] = {
            **exposure_diagnostics,
            "factor_count": int(exposures.shape[1]),
            "factor_return_observations": int(len(factor_returns)),
            "factor_cross_section": int(len(eligible_names)),
            "factor_model_fallback": False,
            "factor_covariance_condition_number": condition_number,
            "median_specific_annual_volatility": float(
                np.sqrt(np.median(specific_variance) * self.settings.annualization)
            ),
            **_beta_diagnostics(beta, beta_observed, self.settings),
        }
        return RiskEstimate(
            as_of_date=str(as_of_date),
            covariance=covariance_frame,
            eligible=history_eligible.rename("risk_eligible"),
            method="factor_ewma",
            observations=len(returns),
            market_beta=(beta if self.settings.beta_model == "ewma_shrunk" else None),
            beta_method=(
                "ewma_shrunk_to_one"
                if self.settings.beta_model == "ewma_shrunk"
                else "feature_beta_60"
            ),
            diagnostics=diagnostics,
        )

    def _current_exposures(
        self,
        as_of_date: str,
        names: pd.Index,
        beta: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        industry_frame = self._asof_frame(self.industry_by_date, as_of_date)
        style_frame = self._asof_frame(self.style_by_date, as_of_date)
        if industry_frame is None or style_frame is None:
            return None

        industry = industry_frame.reindex(names).apply(pd.to_numeric, errors="coerce")
        industry_observed = industry.notna().any(axis=1)
        industry_observed &= industry.fillna(0.0).abs().sum(axis=1).gt(0)
        industry = industry.fillna(0.0)
        missing_column = "industry___MISSING__"
        if missing_column not in industry:
            industry[missing_column] = 0.0
        industry.loc[~industry_observed, missing_column] = 1.0
        industry = industry.loc[:, industry.abs().sum(axis=0).gt(0)]

        styles = style_frame.reindex(names).apply(pd.to_numeric, errors="coerce")
        raw_beta = styles.get("market_beta_60")
        styles = styles.drop(columns=["market_beta_60"], errors="ignore")
        styles = styles.replace([np.inf, -np.inf], np.nan)
        style_observed_fraction = (
            float(styles.notna().mean().mean()) if not styles.empty else 0.0
        )
        styles = styles.fillna(0.0).clip(
            -self.settings.style_exposure_clip,
            self.settings.style_exposure_clip,
        )
        styles = styles.loc[:, styles.abs().sum(axis=0).gt(1e-12)]

        exposures = pd.concat(
            [
                beta.rename("market_beta"),
                industry,
                styles,
            ],
            axis=1,
        ).astype(float)
        diagnostics = {
            "industry_factor_count": int(industry.shape[1]),
            "style_factor_count": int(styles.shape[1]),
            "industry_observed_fraction": float(industry_observed.mean()),
            "style_observed_fraction": style_observed_fraction,
            "feature_beta_observed_fraction": (
                float(pd.to_numeric(raw_beta, errors="coerce").notna().mean())
                if raw_beta is not None
                else 0.0
            ),
        }
        return exposures, diagnostics

    def _estimate_market_beta(
        self,
        as_of_date: str,
        names: pd.Index,
    ) -> tuple[pd.Series, pd.Series]:
        return _estimate_ewma_beta(
            settings=self.settings,
            close=self.close,
            benchmark_close=self.benchmark_close,
            open_dates=self.open_dates,
            as_of_date=str(as_of_date),
            names=names,
        )

    def _fallback_estimate(
        self,
        as_of_date: str,
        names: pd.Index,
        beta: pd.Series,
        beta_observed: pd.Series,
        *,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> RiskEstimate:
        fallback = self.fallback.estimate(str(as_of_date), names.astype(str).tolist())
        return RiskEstimate(
            as_of_date=fallback.as_of_date,
            covariance=fallback.covariance,
            eligible=fallback.eligible,
            method=f"factor_ewma_fallback:{fallback.method}",
            observations=fallback.observations,
            market_beta=(beta if self.settings.beta_model == "ewma_shrunk" else None),
            beta_method=(
                "ewma_shrunk_to_one"
                if self.settings.beta_model == "ewma_shrunk"
                else "feature_beta_60"
            ),
            diagnostics={
                **(diagnostics or {}),
                "factor_count": 0,
                "factor_return_observations": 0,
                "factor_model_fallback": True,
                "factor_model_fallback_reason": reason,
                **_beta_diagnostics(beta, beta_observed, self.settings),
            },
        )


def _ewma_weights(observations: int, half_life: float) -> np.ndarray:
    if observations < 1:
        return np.empty(0, dtype=float)
    age = (observations - 1) - np.arange(observations, dtype=float)
    weights = np.power(0.5, age / float(half_life))
    return weights / weights.sum()


def _benchmark_close(index_bars: pd.DataFrame | None) -> pd.Series | None:
    if index_bars is None:
        return None
    required = {"trade_date", "benchmark_close"}
    missing = required.difference(index_bars.columns)
    if missing:
        raise ValueError(f"Risk model index bars are missing columns: {sorted(missing)}")
    benchmark = index_bars.loc[:, ["trade_date", "benchmark_close"]].copy()
    benchmark["trade_date"] = benchmark["trade_date"].astype(str)
    if benchmark["trade_date"].duplicated().any():
        raise ValueError("Risk model index trade_date key is not unique")
    return benchmark.set_index("trade_date")["benchmark_close"].sort_index()


def _estimate_ewma_beta(
    *,
    settings: RiskSettings,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_dates: list[str],
    as_of_date: str,
    names: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    available_dates = [date for date in open_dates if date <= str(as_of_date)]
    available_dates = available_dates[-(settings.beta_lookback + 1) :]
    raw_stock_close = close.reindex(index=available_dates, columns=names)
    stock_observations = raw_stock_close.notna().sum()
    stock_returns = (
        raw_stock_close.ffill()
        .pct_change(fill_method=None)
        .iloc[1:]
        .clip(-settings.return_clip, settings.return_clip)
    )
    benchmark_returns = (
        benchmark_close.reindex(available_dates)
        .ffill()
        .pct_change(fill_method=None)
        .iloc[1:]
        .clip(-settings.return_clip, settings.return_clip)
        .reindex(stock_returns.index)
    )

    beta = pd.Series(np.nan, index=names, dtype=float)
    observed = pd.Series(False, index=names, dtype=bool)
    full_weights = _ewma_weights(len(stock_returns), settings.beta_half_life)
    benchmark_array = benchmark_returns.to_numpy(dtype=float)
    for name in names:
        if int(stock_observations.loc[name]) < settings.beta_min_history + 1:
            continue
        stock_array = stock_returns[name].to_numpy(dtype=float)
        valid = np.isfinite(stock_array) & np.isfinite(benchmark_array)
        if int(valid.sum()) < settings.beta_min_history:
            continue
        weights = full_weights[valid]
        weights /= weights.sum()
        market = benchmark_array[valid]
        stock = stock_array[valid]
        market_centered = market - float(weights @ market)
        stock_centered = stock - float(weights @ stock)
        market_variance = float(weights @ np.square(market_centered))
        if market_variance <= settings.variance_floor:
            continue
        raw_beta = float(weights @ (market_centered * stock_centered)) / market_variance
        shrunk = (1.0 - settings.beta_shrinkage) * raw_beta + settings.beta_shrinkage
        beta.loc[name] = float(
            np.clip(
                shrunk,
                settings.beta_clip_min,
                settings.beta_clip_max,
            )
        )
        observed.loc[name] = True
    beta = beta.fillna(1.0).rename("market_beta_model")
    return beta, observed


def _beta_diagnostics(
    beta: pd.Series,
    observed: pd.Series,
    settings: RiskSettings,
) -> dict[str, float]:
    observed_beta = beta.loc[observed.reindex(beta.index, fill_value=False)]
    if observed_beta.empty:
        return {
            "beta_observed_fraction": 0.0,
            "beta_lower_clip_fraction": 0.0,
            "beta_upper_clip_fraction": 0.0,
            "beta_clip_fraction": 0.0,
            "beta_median": 1.0,
        }
    tolerance = 1e-12
    lower = observed_beta.le(settings.beta_clip_min + tolerance)
    upper = observed_beta.ge(settings.beta_clip_max - tolerance)
    return {
        "beta_observed_fraction": float(observed.mean()),
        "beta_lower_clip_fraction": float(lower.mean()),
        "beta_upper_clip_fraction": float(upper.mean()),
        "beta_clip_fraction": float((lower | upper).mean()),
        "beta_median": float(observed_beta.median()),
    }


def _ewma_covariance(matrix: np.ndarray, *, half_life: float) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("EWMA covariance requires a two-dimensional sample")
    weights = _ewma_weights(values.shape[0], half_life)
    mean = weights @ values
    centered = values - mean
    denominator = max(1.0 - float(weights @ weights), 1e-12)
    return (centered.T @ (centered * weights[:, None])) / denominator


def build_risk_model(
    settings: RiskSettings,
    market_panel: pd.DataFrame,
    open_dates: list[str],
    *,
    index_bars: pd.DataFrame | None = None,
    industry_exposures: pd.DataFrame | None = None,
    style_exposures: pd.DataFrame | None = None,
) -> RiskModel:
    """Construct the configured risk model without changing optimizer semantics."""

    if settings.model == "ledoit_wolf":
        return LedoitWolfRiskModel(
            settings,
            market_panel,
            open_dates,
            index_bars=index_bars,
        )
    if settings.model == "factor_ewma":
        missing = [
            name
            for name, value in (
                ("index_bars", index_bars),
                ("industry_exposures", industry_exposures),
                ("style_exposures", style_exposures),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "factor_ewma risk model requires point-in-time context: "
                f"{', '.join(missing)}"
            )
        return FactorEWMARiskModel(
            settings,
            market_panel,
            open_dates,
            index_bars=index_bars,
            industry_exposures=industry_exposures,
            style_exposures=style_exposures,
        )
    raise ValueError(f"Unsupported risk model: {settings.model}")
