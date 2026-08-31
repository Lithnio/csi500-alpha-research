from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from csi500_alpha.config import FeatureSettings
from csi500_alpha.features.catalog import (
    A2_DAILY_FACTOR_NAMES,
    A3_ALL_DAILY_FACTOR_NAMES,
    A3_DAILY_FACTOR_NAMES,
    FACTOR_NAMES,
)
from csi500_alpha.logging_utils import ProgressCallback, ProgressLogger
from csi500_alpha.research.industry import industry_asof
from csi500_alpha.research.universe import benchmark_weights_asof, select_rebalance_dates

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedFactors:
    features: pd.DataFrame
    quality: pd.DataFrame


def _pivot(frame: pd.DataFrame, column: str, open_dates: list[str]) -> pd.DataFrame:
    result = frame.pivot(index="trade_date", columns="instrument", values=column)
    return result.reindex(open_dates).sort_index()


def build_raw_factor_panel(
    *,
    market_panel: pd.DataFrame,
    daily_characteristics: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    open_dates: list[str],
    start_date: str,
    end_date: str,
    rebalance_every: int,
    index_bars: pd.DataFrame | None = None,
    industry_membership: pd.DataFrame | None = None,
    benchmark_membership_intervals: pd.DataFrame | None = None,
    industry_transition_date: str = "20211213",
    factor_names: Sequence[str] = FACTOR_NAMES,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Build raw factors using only rows dated at or before each decision date."""
    requested_factors = tuple(factor_names)
    unsupported = sorted(
        set(requested_factors).difference(A3_ALL_DAILY_FACTOR_NAMES)
    )
    if unsupported:
        raise ValueError(f"Unsupported built-in daily factors: {unsupported}")
    dates = sorted(str(date) for date in open_dates)
    close_raw = _pivot(market_panel, "adjusted_close", dates)
    close = close_raw.ffill()
    log_close = np.log(close.where(close > 0))
    returns = log_close.diff()
    if index_bars is None or index_bars.empty:
        index_returns = pd.Series(np.nan, index=dates, dtype=float)
    else:
        index_frame = index_bars.copy()
        index_frame["trade_date"] = index_frame["trade_date"].astype(str)
        index_close = (
            index_frame.drop_duplicates("trade_date", keep="last")
            .set_index("trade_date")["benchmark_close"]
            .pipe(pd.to_numeric, errors="coerce")
            .reindex(dates)
            .ffill()
        )
        index_returns = np.log(index_close.where(index_close > 0)).diff()

    high = _pivot(market_panel, "high", dates)
    low = _pivot(market_panel, "low", dates)
    raw_close = _pivot(market_panel, "close", dates)
    amount = _pivot(market_panel, "amount_cny", dates)
    valid_history = close.notna()

    log_range = np.log(high.where(high > 0) / low.where(low > 0)).pow(2)
    log_range = log_range.where(high.notna() & low.notna(), 0.0).where(valid_history)
    amihud_daily = returns.abs() / amount.where(amount > 0)
    spread = high - low
    observed_range = high.notna() & low.notna() & raw_close.notna()
    close_location = (2.0 * raw_close - high - low) / spread.where(spread.ne(0.0))
    # A zero-range session has no directional close location.  Treat it as the
    # neutral value instead of poisoning the entire rolling window with NaN.
    close_location = close_location.mask(observed_range & spread.eq(0.0), 0.0)

    characteristics = daily_characteristics.copy()
    characteristic_dates = characteristics["trade_date"].astype(str)
    characteristics["trade_date"] = characteristic_dates
    turnover_raw = _pivot(characteristics, "turnover_rate", dates)
    turnover = turnover_raw.fillna(0.0).where(valid_history)
    free_turnover_raw = _pivot(characteristics, "turnover_rate_f", dates)
    free_turnover = free_turnover_raw.fillna(0.0).where(valid_history)
    turnover_ratio = turnover.rolling(5, min_periods=5).mean() / turnover.rolling(
        60,
        min_periods=60,
    ).mean().where(lambda value: value > 0)
    circ_mv = _pivot(characteristics, "circ_mv_cny", dates).ffill()
    total_mv = _pivot(characteristics, "total_mv_cny", dates).ffill()
    pb = _pivot(characteristics, "pb", dates).ffill()

    rolling_market_mean = index_returns.rolling(60, min_periods=60).mean()
    rolling_market_variance = (
        index_returns.pow(2).rolling(60, min_periods=60).mean()
        - rolling_market_mean.pow(2)
    ).where(lambda value: value > 1e-16)
    rolling_stock_mean = returns.rolling(60, min_periods=60).mean()
    rolling_stock_variance = (
        returns.pow(2).rolling(60, min_periods=60).mean()
        - rolling_stock_mean.pow(2)
    ).clip(lower=0.0)
    rolling_covariance = (
        returns.mul(index_returns, axis="index").rolling(60, min_periods=60).mean()
        - rolling_stock_mean.mul(rolling_market_mean, axis="index")
    )
    beta = rolling_covariance.div(rolling_market_variance, axis="index")
    idiosyncratic_variance = (
        rolling_stock_variance
        - beta.pow(2).mul(rolling_market_variance, axis="index")
    ).clip(lower=0.0)
    reversal_5d = -(log_close - log_close.shift(5))
    turnover_shock = np.log(turnover_ratio.where(turnover_ratio > 0))

    raw_factors: dict[str, pd.DataFrame] = {
        "reversal_1d": -(log_close - log_close.shift(1)),
        "reversal_5d": reversal_5d,
        "reversal_10d": -(log_close - log_close.shift(10)),
        "reversal_20d": -(log_close - log_close.shift(20)),
        "momentum_20_5": log_close.shift(5) - log_close.shift(20),
        "momentum_60_20": log_close.shift(20) - log_close.shift(60),
        "momentum_120_20": log_close.shift(20) - log_close.shift(120),
        "momentum_250_20": log_close.shift(20) - log_close.shift(250),
        "low_vol_20": -returns.rolling(20, min_periods=20).std(ddof=1),
        "low_downside_vol_60": -np.sqrt(
            returns.clip(upper=0.0).pow(2).rolling(60, min_periods=60).mean()
        ),
        "low_range_vol_20": -np.sqrt(
            log_range.rolling(20, min_periods=20).mean() / (4.0 * np.log(2.0))
        ),
        "beta_60": beta,
        "low_idio_vol_60": -np.sqrt(idiosyncratic_variance),
        "skewness_60": returns.rolling(60, min_periods=60).skew(),
        "max_return_20": returns.rolling(20, min_periods=20).max(),
        "amihud_20": np.log(
            amihud_daily.rolling(20, min_periods=20).mean().where(lambda value: value > 0)
        ),
        "free_turnover_20": np.log(
            free_turnover.rolling(20, min_periods=20).mean().where(lambda value: value > 0)
        ),
        "turnover_shock_5_60": turnover_shock,
        "zero_return_20": (
            returns.abs().le(1e-12).where(returns.notna()).rolling(20, min_periods=20).mean()
        ),
        "close_location_20": close_location.rolling(20, min_periods=20).mean(),
        "size": -np.log(circ_mv.where(circ_mv > 0)),
        "total_size": -np.log(total_mv.where(total_mv > 0)),
        "free_float_ratio": np.log(
            (circ_mv / total_mv).where(
                (circ_mv > 0) & (total_mv > 0) & (circ_mv <= total_mv * (1.0 + 1e-8))
            )
        ),
        "book_to_price": -np.log(pb.where(pb > 0)),
        "abnormal_turnover_reversal": reversal_5d * turnover_shock.clip(lower=0.0),
    }
    expanded_names = set(A2_DAILY_FACTOR_NAMES) | set(A3_DAILY_FACTOR_NAMES)
    if set(requested_factors).intersection(expanded_names):
        adjusted_open = _pivot(market_panel, "adjusted_open", dates)
        up_limit = _pivot(market_panel, "up_limit", dates)
        lagged_beta = beta.shift(1)
        market_residual_return = returns - lagged_beta.mul(
            index_returns,
            axis="index",
        )
        mean_turnover_60 = turnover.rolling(60, min_periods=60).mean()
        turnover_ratio_60 = turnover.div(mean_turnover_60.where(mean_turnover_60 > 0))
        high_turnover = np.log(turnover_ratio_60.where(turnover_ratio_60 > 0)).clip(
            lower=0.0
        )
        high_turnover = high_turnover.mask(
            turnover.eq(0.0) & mean_turnover_60.notna(),
            0.0,
        ).where(mean_turnover_60.notna() & valid_history)
        intraday_return = np.log(
            close_raw.where(close_raw > 0) / adjusted_open.where(adjusted_open > 0)
        )
        upper_limit_observed = high.notna() & raw_close.notna() & up_limit.notna()
        hit_upper_limit = high.ge(up_limit * (1.0 - 1e-6))
        close_at_upper_limit = raw_close.ge(up_limit * (1.0 - 1e-6))
        limit_up_close = close_at_upper_limit.astype(float).where(upper_limit_observed)
        failed_limit_up = (hit_upper_limit & ~close_at_upper_limit).astype(float).where(
            upper_limit_observed
        )
        raw_factors.update(
            {
                "market_residual_reversal_20": -market_residual_return.rolling(
                    20,
                    min_periods=20,
                ).sum(),
                "market_residual_momentum_120_20": market_residual_return.shift(
                    20
                ).rolling(100, min_periods=100).sum(),
                "turnover_volatility_20": np.log1p(
                    turnover.clip(lower=0.0)
                ).rolling(20, min_periods=20).std(ddof=1),
                "high_turnover_return_20": (returns * high_turnover).rolling(
                    20,
                    min_periods=20,
                ).sum(),
                "intraday_strength_20": intraday_return.rolling(
                    20,
                    min_periods=20,
                ).mean(),
                "limit_up_close_rate_20": limit_up_close.rolling(
                    20,
                    min_periods=20,
                ).mean(),
                "failed_limit_up_rate_20": failed_limit_up.rolling(
                    20,
                    min_periods=20,
                ).mean(),
            }
        )
        if set(requested_factors).intersection(A3_DAILY_FACTOR_NAMES):
            raw_open = _pivot(market_panel, "open", dates)
            volume_shares = _pivot(market_panel, "volume_shares", dates)
            down_limit = _pivot(market_panel, "down_limit", dates)
            lower_limit_observed = (
                low.notna() & raw_close.notna() & down_limit.notna()
            )
            hit_lower_limit = low.le(down_limit * (1.0 + 1e-6))
            close_at_lower_limit = raw_close.le(
                down_limit * (1.0 + 1e-6)
            )
            limit_down_close = close_at_lower_limit.astype(float).where(
                lower_limit_observed
            )
            failed_limit_down = (
                hit_lower_limit & ~close_at_lower_limit
            ).astype(float).where(lower_limit_observed)
            upper_limit_exclusion = close_at_upper_limit | close_at_upper_limit.shift(
                1,
                fill_value=False,
            )
            limit_adjusted_returns = returns.mask(
                upper_limit_exclusion,
                0.0,
            )
            prior_close = close.shift(1)
            overnight_return = np.log(
                adjusted_open.where(adjusted_open > 0)
                / prior_close.where(prior_close > 0)
            )
            raw_factors.update(
                {
                    "limit_adjusted_momentum_120_20": (
                        limit_adjusted_returns.shift(20)
                        .rolling(100, min_periods=100)
                        .sum()
                    ),
                    "alpha006_open_volume_corr_10": -raw_open.rolling(
                        10,
                        min_periods=10,
                    ).corr(volume_shares),
                    # The point-in-time benchmark price rank is applied below.
                    "high_price_momentum_250_20": raw_factors[
                        "momentum_250_20"
                    ],
                    "overnight_intraday_divergence_20": (
                        overnight_return - intraday_return
                    ).rolling(20, min_periods=20).mean(),
                    "limit_down_close_rate_20": limit_down_close.rolling(
                        20,
                        min_periods=20,
                    ).mean(),
                    "failed_limit_down_rate_20": failed_limit_down.rolling(
                        20,
                        min_periods=20,
                    ).mean(),
                }
            )

    decision_dates = select_rebalance_dates(
        dates,
        start_date=start_date,
        end_date=end_date,
        every=rebalance_every,
    )
    rows: list[pd.DataFrame] = []
    membership = industry_membership if industry_membership is not None else pd.DataFrame()
    progress = (
        ProgressLogger(
            LOGGER,
            stage="raw_factor_panel",
            total=len(decision_dates),
            callback=progress_callback,
        )
        if decision_dates
        else None
    )
    for position, decision_date in enumerate(decision_dates, start=1):
        benchmark = benchmark_weights_asof(
            benchmark_weights,
            decision_date,
            benchmark_membership_intervals,
        )
        if benchmark.empty:
            if progress is not None:
                progress.update(position, context={"decision_date": decision_date})
            continue
        instruments = benchmark.index
        day = pd.DataFrame(
            {
                "decision_date": decision_date,
                "instrument": instruments,
                "benchmark_weight": benchmark.reindex(instruments).to_numpy(),
                "circ_mv_cny": circ_mv.loc[decision_date].reindex(instruments).to_numpy(),
                "total_mv_cny": total_mv.loc[decision_date].reindex(instruments).to_numpy(),
                "pb": pb.loc[decision_date].reindex(instruments).to_numpy(),
            }
        )
        industries = industry_asof(
            membership,
            decision_date,
            transition_date=industry_transition_date,
        )
        day["industry_code"] = day["instrument"].map(industries)
        for name in requested_factors:
            values = raw_factors[name].loc[decision_date].reindex(instruments)
            if name == "high_price_momentum_250_20":
                price_rank = (
                    raw_close.loc[decision_date]
                    .reindex(instruments)
                    .rank(method="average", pct=True)
                )
                values = values * price_rank
            day[name] = values.to_numpy()
        rows.append(day)
        if progress is not None:
            progress.update(position, context={"decision_date": decision_date})
    if not rows:
        columns = [
            "decision_date",
            "instrument",
            "benchmark_weight",
            "circ_mv_cny",
            "total_mv_cny",
            "pb",
            "industry_code",
            *requested_factors,
        ]
        return pd.DataFrame(columns=columns)
    result = pd.concat(rows, ignore_index=True)
    for factor in requested_factors:
        result[factor] = pd.to_numeric(result[factor], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return result.sort_values(["decision_date", "instrument"]).reset_index(drop=True)


def process_factor_panel(
    raw_features: pd.DataFrame,
    settings: FeatureSettings,
    factor_names: Sequence[str] = FACTOR_NAMES,
    progress_callback: ProgressCallback | None = None,
) -> ProcessedFactors:
    result = raw_features.copy()
    names = tuple(factor_names)
    for factor in names:
        if factor not in result.columns:
            raise ValueError(f"Raw feature panel is missing factor: {factor}")
        result[f"{factor}__z"] = np.nan
    quality_rows: list[dict[str, object]] = []
    decision_count = int(result["decision_date"].nunique())
    progress = (
        ProgressLogger(
            LOGGER,
            stage="factor_preprocessing",
            total=decision_count,
            callback=progress_callback,
        )
        if decision_count
        else None
    )

    for position, (decision_date, frame) in enumerate(
        result.groupby("decision_date", sort=True),
        start=1,
    ):
        industry_coverage = float(frame["industry_code"].notna().mean())
        industry_enabled = industry_coverage >= settings.industry_coverage_threshold
        for factor in names:
            values = pd.to_numeric(frame[factor], errors="coerce")
            valid = values.notna()
            coverage = float(valid.mean())
            active = coverage >= settings.min_factor_coverage and int(valid.sum()) >= 3
            clipped_fraction = 0.0
            residual_std = np.nan
            if active:
                sample = values[valid]
                median = float(sample.median())
                robust_scale = float((sample - median).abs().median() * 1.4826)
                if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
                    robust_scale = float(sample.std(ddof=1))
                if np.isfinite(robust_scale) and robust_scale > 1e-12:
                    lower = median - settings.mad_clip * robust_scale
                    upper = median + settings.mad_clip * robust_scale
                    clipped = sample.clip(lower, upper)
                    clipped_fraction = float((clipped != sample).mean())
                    residual = _neutralize(
                        clipped,
                        frame.loc[valid],
                        factor=factor,
                        use_industry=industry_enabled,
                    )
                    residual_std = float(residual.std(ddof=1))
                    if np.isfinite(residual_std) and residual_std > 1e-12:
                        standardized = (residual - residual.mean()) / residual_std
                        result.loc[frame.index[valid], f"{factor}__z"] = standardized
                    else:
                        active = False
                else:
                    active = False
            quality_rows.append(
                {
                    "decision_date": str(decision_date),
                    "factor": factor,
                    "coverage": coverage,
                    "active": active,
                    "clipped_fraction": clipped_fraction,
                    "residual_std": residual_std,
                    "industry_coverage": industry_coverage,
                    "industry_neutralized": industry_enabled,
                }
            )
        if progress is not None:
            progress.update(
                position,
                context={"decision_date": str(decision_date)},
            )
    quality = pd.DataFrame(quality_rows)
    return ProcessedFactors(features=result, quality=quality)


def _neutralize(
    values: pd.Series,
    frame: pd.DataFrame,
    *,
    factor: str,
    use_industry: bool,
) -> pd.Series:
    controls: list[np.ndarray] = [np.ones(len(values), dtype=float)]
    if factor != "size":
        log_size = np.log(frame["circ_mv_cny"].where(frame["circ_mv_cny"] > 0))
        log_size = log_size.fillna(log_size.median())
        controls.append(log_size.to_numpy(dtype=float))
    if use_industry:
        dummies = pd.get_dummies(
            frame["industry_code"].fillna("__MISSING__"),
            dtype=float,
        )
        if dummies.shape[1] > 1:
            controls.extend(
                dummies.iloc[:, 1:].to_numpy(dtype=float).T
            )
    design = np.column_stack(controls)
    target = values.to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return pd.Series(target - design @ coefficients, index=values.index)
