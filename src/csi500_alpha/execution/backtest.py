from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import ResearchSettings
from csi500_alpha.execution.liquidity import (
    LIQUIDITY_CONTRACT_VERSION,
    LiquiditySnapshot,
    build_trailing_adv_snapshots,
)
from csi500_alpha.execution.tradeability import opening_suspensions_by_date
from csi500_alpha.logging_utils import ProgressCallback, ProgressLogger
from csi500_alpha.portfolio.audit import (
    audit_executed_portfolio,
    summarize_constraint_audits,
)
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.universe import benchmark_weights_asof, select_rebalance_dates
from csi500_alpha.risk.model import RiskModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    targets: pd.DataFrame
    optimization: pd.DataFrame
    positions: pd.DataFrame
    constraint_audits: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PortfolioContext:
    exposures: pd.DataFrame | None
    cannot_buy: set[str]
    cannot_sell: set[str]
    max_trade_weights: pd.Series | None


@dataclass(frozen=True)
class PendingTarget:
    signal_date: str
    execution_date: str
    target: pd.Series
    benchmark: pd.Series
    covariance: pd.DataFrame | None
    industry_exposures: pd.DataFrame | None
    style_exposures: pd.DataFrame | None
    liquidity: LiquiditySnapshot | None
    risk_method: str | None = None
    beta_method: str | None = None


def enrich_active_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Recalculate portfolio, benchmark and relative-wealth return series."""

    required = {"nav", "benchmark_nav"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"Backtest daily data are missing columns: {missing}")
    if daily.empty:
        raise ValueError("Backtest daily data must be nonempty")

    result = daily.copy()
    nav = pd.to_numeric(result["nav"], errors="raise").astype(float)
    benchmark_nav = pd.to_numeric(
        result["benchmark_nav"],
        errors="raise",
    ).astype(float)
    if (
        not np.isfinite(nav).all()
        or not np.isfinite(benchmark_nav).all()
        or (nav <= 0).any()
        or (benchmark_nav <= 0).any()
    ):
        raise ValueError("Portfolio and benchmark NAV must be finite and positive")

    portfolio_nav = nav / float(nav.iloc[0])
    benchmark_nav = benchmark_nav / float(benchmark_nav.iloc[0])
    active_nav = portfolio_nav / benchmark_nav
    result["benchmark_nav"] = benchmark_nav
    result["portfolio_return"] = portfolio_nav.pct_change().fillna(0.0)
    result["benchmark_return"] = benchmark_nav.pct_change().fillna(0.0)
    result["active_nav"] = active_nav
    result["active_return"] = active_nav.pct_change().fillna(0.0)

    rolling_covariance = result["portfolio_return"].rolling(
        60,
        min_periods=40,
    ).cov(result["benchmark_return"])
    rolling_variance = result["benchmark_return"].rolling(
        60,
        min_periods=40,
    ).var(ddof=1)
    result["rolling_beta_60"] = rolling_covariance / rolling_variance.where(
        rolling_variance > 1e-16
    )
    result["rolling_beta_deviation_60"] = result["rolling_beta_60"] - 1.0
    return result


def calculate_backtest_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate unambiguous absolute and benchmark-relative performance."""

    required = {
        "trade_date",
        "nav",
        "benchmark_nav",
        "portfolio_return",
        "benchmark_return",
        "active_return",
        "active_nav",
        "turnover",
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"Backtest metric inputs are missing columns: {missing}")
    if daily.empty:
        raise ValueError("Backtest metric inputs must be nonempty")

    observations = max(len(daily) - 1, 1)
    years = observations / 252.0
    initial_nav = float(daily["nav"].iloc[0])
    final_nav = float(daily["nav"].iloc[-1])
    initial_benchmark = float(daily["benchmark_nav"].iloc[0])
    final_benchmark = float(daily["benchmark_nav"].iloc[-1])
    portfolio_growth = final_nav / initial_nav
    benchmark_growth = final_benchmark / initial_benchmark
    active_growth = portfolio_growth / benchmark_growth

    portfolio_returns = pd.to_numeric(
        daily["portfolio_return"], errors="raise"
    ).astype(float).iloc[1:]
    benchmark_returns = pd.to_numeric(
        daily["benchmark_return"], errors="raise"
    ).astype(float).iloc[1:]
    active_returns = pd.to_numeric(
        daily["active_return"], errors="raise"
    ).astype(float).iloc[1:]
    tracking_error = (
        float(active_returns.std(ddof=1) * np.sqrt(252.0))
        if len(active_returns) > 1
        else 0.0
    )
    annualized_active_mean = (
        float(active_returns.mean() * 252.0) if len(active_returns) else np.nan
    )
    information_ratio = (
        annualized_active_mean / tracking_error
        if tracking_error > 0
        else np.nan
    )

    portfolio_wealth = pd.to_numeric(daily["nav"], errors="raise").astype(float)
    active_wealth = pd.to_numeric(daily["active_nav"], errors="raise").astype(float)
    portfolio_drawdown = portfolio_wealth / portfolio_wealth.cummax() - 1.0
    active_drawdown = active_wealth / active_wealth.cummax() - 1.0
    alpha_daily, beta = fit_capm(portfolio_returns, benchmark_returns)
    alpha_annualized = alpha_daily * 252.0
    beta_drag = (
        (beta - 1.0) * float(benchmark_returns.mean()) * 252.0
        if np.isfinite(beta) and len(benchmark_returns)
        else np.nan
    )
    reconciliation_error = (
        annualized_active_mean - alpha_annualized - beta_drag
        if np.isfinite(annualized_active_mean)
        and np.isfinite(alpha_annualized)
        and np.isfinite(beta_drag)
        else np.nan
    )

    rolling_beta_deviation = pd.to_numeric(
        daily.get("rolling_beta_deviation_60", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna().abs()
    costs = (
        float(
            trades["linear_cost"].sum()
            + trades["stamp_duty"].sum()
            + trades["impact_cost"].sum()
        )
        if not trades.empty
        else 0.0
    )
    active_max_drawdown = float(active_drawdown.min())
    return {
        "start_date": str(daily["trade_date"].iloc[0]),
        "end_date": str(daily["trade_date"].iloc[-1]),
        "observations": len(daily),
        "final_nav": final_nav,
        "final_benchmark_nav": final_benchmark,
        "final_active_nav": float(active_wealth.iloc[-1]),
        "total_return": portfolio_growth - 1.0,
        "benchmark_total_return": benchmark_growth - 1.0,
        "relative_active_total_return": active_growth - 1.0,
        "annualized_return": (
            portfolio_growth ** (1.0 / years) - 1.0
            if years > 0 and portfolio_growth > 0
            else np.nan
        ),
        "annualized_active_return": (
            active_growth ** (1.0 / years) - 1.0
            if years > 0 and active_growth > 0
            else np.nan
        ),
        "annualized_active_mean": annualized_active_mean,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "portfolio_max_drawdown": float(portfolio_drawdown.min()),
        "active_max_drawdown": active_max_drawdown,
        "max_drawdown": active_max_drawdown,
        "capm_alpha_annualized": alpha_annualized,
        "capm_beta": beta,
        "capm_beta_drag_annualized": beta_drag,
        "capm_reconciliation_error": reconciliation_error,
        "rolling_beta_abs_deviation_p95": (
            float(rolling_beta_deviation.quantile(0.95))
            if not rolling_beta_deviation.empty
            else np.nan
        ),
        "average_turnover": float(daily["turnover"].fillna(0.0).mean()),
        "transaction_cost": costs,
        "filled_orders": int((trades["status"] == "filled").sum())
        if not trades.empty
        else 0,
        "partial_orders": int((trades["status"] == "partial").sum())
        if not trades.empty
        else 0,
        "executed_orders": (
            int(trades["status"].isin(["filled", "partial"]).sum())
            if not trades.empty
            else 0
        ),
        "blocked_orders": int((trades["status"] == "blocked").sum())
        if not trades.empty
        else 0,
    }


def fit_capm(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    sample = pd.DataFrame(
        {
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 2 or float(sample["benchmark"].var(ddof=1)) <= 1e-16:
        return np.nan, np.nan
    design = np.column_stack(
        [np.ones(len(sample)), sample["benchmark"].to_numpy(dtype=float)]
    )
    alpha, beta = np.linalg.lstsq(
        design,
        sample["portfolio"].to_numpy(dtype=float),
        rcond=None,
    )[0]
    return float(alpha), float(beta)


class SmokeEventBacktester:
    """Research event loop proving signal-time, execution and cost boundaries."""

    def __init__(
        self,
        settings: ResearchSettings,
        *,
        risk_model: RiskModel | None = None,
        optimizer: ActivePortfolioOptimizer | None = None,
    ) -> None:
        self.settings = settings
        self.linear_rate = settings.linear_cost_bps / 10000.0
        if (risk_model is None) != (optimizer is None):
            raise ValueError("Risk model and optimizer must be supplied together")
        self.risk_model = risk_model
        self.optimizer = optimizer

    def run(
        self,
        *,
        calendar: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        index_bars: pd.DataFrame,
        market_panel: pd.DataFrame,
        signals: pd.DataFrame,
        portfolio_exposures: pd.DataFrame | None = None,
        portfolio_styles: pd.DataFrame | None = None,
        portfolio_restrictions: pd.DataFrame | None = None,
        suspensions: pd.DataFrame | None = None,
        start_date: str,
        end_date: str,
        rebalance_dates: Sequence[str] | None = None,
        benchmark_membership_intervals: pd.DataFrame | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BacktestResult:
        required_signal_columns = {"trade_date", "instrument", "score"}
        missing_signal_columns = sorted(required_signal_columns.difference(signals.columns))
        if missing_signal_columns:
            raise ValueError(f"Portfolio signals are missing columns: {missing_signal_columns}")
        if self.optimizer is not None and "expected_return" not in signals:
            raise ValueError("Active optimization requires calibrated expected_return")
        if signals.duplicated(["trade_date", "instrument"]).any():
            raise ValueError("Portfolio signal trade_date/instrument key is not unique")
        open_dates = calendar.loc[calendar["is_open"] == 1, "trade_date"].astype(str).tolist()
        backtest_dates = [date for date in open_dates if start_date <= date <= end_date]
        if len(backtest_dates) < 2:
            raise ValueError("Smoke backtest requires at least two open dates")
        if rebalance_dates is None:
            rebalance_date_set = set(
                select_rebalance_dates(
                    open_dates,
                    start_date=start_date,
                    end_date=end_date,
                    every=self.settings.rebalance_every,
                )
            )
        else:
            rebalance_date_set = {
                str(date) for date in rebalance_dates if start_date <= str(date) <= end_date
            }
            invalid = rebalance_date_set.difference(backtest_dates)
            if invalid:
                raise ValueError(
                    f"Rebalance dates are not open backtest dates: {sorted(invalid)}"
                )

        bars_by_date = {
            str(date): frame.set_index("instrument", drop=False)
            for date, frame in market_panel.groupby("trade_date", sort=True)
        }
        signals_by_date = {
            str(date): frame.set_index("instrument")
            for date, frame in signals.groupby("trade_date", sort=True)
        }
        exposures_by_date = self._frames_by_date(
            portfolio_exposures,
            date_column="trade_date",
            drop_columns={"trade_date"},
        )
        styles_by_date = self._frames_by_date(
            portfolio_styles,
            date_column="trade_date",
            drop_columns={"trade_date"},
        )
        restrictions_by_date = self._frames_by_date(
            portfolio_restrictions,
            date_column="trade_date",
            drop_columns={"trade_date"},
        )
        suspended_by_date = self._suspensions_by_date(suspensions)
        liquidity_by_date = self._liquidity_by_date(market_panel)
        index_close = (
            index_bars.set_index("trade_date")["benchmark_close"]
            .sort_index()
            .reindex(backtest_dates)
            .ffill()
        )
        if index_close.isna().any():
            raise ValueError("Index close is missing inside the smoke backtest range")

        historical = market_panel[market_panel["trade_date"] < start_date]
        last_close = (
            historical.sort_values("trade_date")
            .groupby("instrument")["adjusted_close"]
            .last()
            .dropna()
            .to_dict()
        )
        holdings: dict[str, float] = {}
        cash = self.settings.initial_cash
        pending: PendingTarget | None = None
        daily_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        optimization_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        constraint_audit_rows: list[dict[str, Any]] = []
        progress = ProgressLogger(
            LOGGER,
            stage="event_backtest",
            total=len(backtest_dates),
            callback=progress_callback,
        )

        for position, trade_date in enumerate(backtest_dates):
            day = bars_by_date.get(trade_date, pd.DataFrame()).copy()
            pre_nav = self._portfolio_value(
                holdings,
                cash,
                day,
                last_close,
                price_column="adjusted_open",
            )
            daily_gross = 0.0
            daily_cost = 0.0

            if pending is not None:
                if trade_date != pending.execution_date:
                    raise AssertionError(
                        "Pending target reached an unexpected execution date: "
                        f"expected={pending.execution_date}, actual={trade_date}"
                    )
                execution_pre_weights, execution_pre_cash_weight = (
                    self._current_weights(
                        holdings,
                        cash,
                        day,
                        last_close,
                        price_column="adjusted_open",
                    )
                )
                cash, gross, cost, executed_rows = self._execute_target(
                    signal_date=pending.signal_date,
                    trade_date=trade_date,
                    target=pending.target,
                    holdings=holdings,
                    cash=cash,
                    pre_nav=pre_nav,
                    day=day,
                    last_close=last_close,
                    suspended_instruments=suspended_by_date.get(trade_date, set()),
                    liquidity=pending.liquidity,
                )
                daily_gross += gross
                daily_cost += cost
                trade_rows.extend(executed_rows)
                if self.optimizer is not None and pending.covariance is not None:
                    actual_weights, actual_cash_weight = self._current_weights(
                        holdings,
                        cash,
                        day,
                        last_close,
                        price_column="adjusted_open",
                    )
                    audit = audit_executed_portfolio(
                        signal_date=pending.signal_date,
                        execution_date=trade_date,
                        settings=self.optimizer.settings,
                        risk_annualization=self.optimizer.risk.annualization,
                        pre_weights=execution_pre_weights,
                        pre_cash_weight=execution_pre_cash_weight,
                        actual_weights=actual_weights,
                        actual_cash_weight=actual_cash_weight,
                        benchmark=pending.benchmark,
                        target=pending.target,
                        covariance=pending.covariance,
                        industry_exposures=pending.industry_exposures,
                        style_exposures=pending.style_exposures,
                        trade_records=executed_rows,
                        execution_day=day,
                        pre_nav=pre_nav,
                        risk_method=pending.risk_method,
                        beta_method=pending.beta_method,
                    )
                    position_rows.extend(audit.positions.to_dict(orient="records"))
                    constraint_audit_rows.append(audit.summary)
                pending = None

            if not day.empty:
                valid_close = day["adjusted_close"].replace([np.inf, -np.inf], np.nan).dropna()
                last_close.update(valid_close.to_dict())
            nav = self._portfolio_value(
                holdings,
                cash,
                day,
                last_close,
                price_column="adjusted_close",
            )
            daily_rows.append(
                {
                    "trade_date": trade_date,
                    "nav": nav,
                    "cash": cash,
                    "holdings": len([units for units in holdings.values() if units > 1e-14]),
                    "gross_trade_value": daily_gross,
                    "turnover": daily_gross / pre_nav if pre_nav > 0 else np.nan,
                    "transaction_cost": daily_cost,
                }
            )

            if trade_date in rebalance_date_set and position + 1 < len(backtest_dates):
                signal_frame = signals_by_date.get(trade_date, pd.DataFrame())
                scores = (
                    signal_frame["score"]
                    if not signal_frame.empty
                    else pd.Series(dtype=float)
                )
                expected_returns = (
                    signal_frame["expected_return"]
                    if "expected_return" in signal_frame
                    else pd.Series(dtype=float)
                )
                benchmark = benchmark_weights_asof(
                    benchmark_weights,
                    trade_date,
                    benchmark_membership_intervals,
                )
                execution_date = backtest_dates[position + 1]
                liquidity_snapshot = liquidity_by_date.get(trade_date)
                target: pd.Series | None = None
                audit_covariance: pd.DataFrame | None = None
                audit_industry_exposures: pd.DataFrame | None = None
                audit_style_exposures: pd.DataFrame | None = None
                construction_method = "top_n_equal_weight"
                if not benchmark.empty and self.optimizer is not None:
                    pre_weights, pre_cash_weight = self._current_weights(
                        holdings,
                        cash,
                        day,
                        last_close,
                        price_column="adjusted_close",
                    )
                    names = sorted(set(benchmark.index) | set(pre_weights.index))
                    if self.risk_model is None:
                        raise AssertionError("Optimizer requires an initialized risk model")
                    risk_estimate = self.risk_model.estimate(trade_date, names)
                    portfolio_context = self._portfolio_context(
                        decision_date=trade_date,
                        names=names,
                        pre_cash_weight=pre_cash_weight,
                        exposures=exposures_by_date.get(trade_date),
                        restrictions=restrictions_by_date.get(trade_date),
                        liquidity=liquidity_snapshot,
                    )
                    style_exposures = styles_by_date.get(trade_date)
                    aligned_styles = (
                        style_exposures.reindex(names).copy()
                        if style_exposures is not None
                        else None
                    )
                    if risk_estimate.market_beta is not None:
                        if aligned_styles is None:
                            aligned_styles = pd.DataFrame(index=pd.Index(names))
                        if "market_beta_60" in aligned_styles:
                            aligned_styles["market_beta_raw_60"] = aligned_styles[
                                "market_beta_60"
                            ]
                        aligned_styles["market_beta_60"] = (
                            risk_estimate.market_beta.reindex(names)
                        )
                    market_beta = (
                        aligned_styles["market_beta_60"]
                        if aligned_styles is not None
                        and "market_beta_60" in aligned_styles
                        else None
                    )
                    optimization_result = self.optimizer.solve(
                        decision_date=trade_date,
                        execution_date=execution_date,
                        expected_returns=expected_returns,
                        benchmark=benchmark,
                        pre_weights=pre_weights,
                        pre_cash_weight=pre_cash_weight,
                        risk_estimate=risk_estimate,
                        exposures=portfolio_context.exposures,
                        market_beta=market_beta,
                        cannot_buy=portfolio_context.cannot_buy,
                        cannot_sell=portfolio_context.cannot_sell,
                        max_trade_weights=portfolio_context.max_trade_weights,
                    )
                    target = optimization_result.target
                    audit_covariance = risk_estimate.covariance
                    audit_industry_exposures = portfolio_context.exposures
                    audit_style_exposures = aligned_styles
                    construction_method = "active_optimizer"
                    diagnostics = optimization_result.diagnostics.copy()
                    if liquidity_snapshot is not None:
                        available_liquidity = liquidity_snapshot.adv_cny.reindex(
                            names
                        ).notna()
                        diagnostics.update(
                            {
                                "liquidity_contract": LIQUIDITY_CONTRACT_VERSION,
                                "liquidity_reference_date": (
                                    liquidity_snapshot.reference_date
                                ),
                                "liquidity_available_instruments": int(
                                    available_liquidity.sum()
                                ),
                                "liquidity_universe_coverage": float(
                                    available_liquidity.mean()
                                ),
                            }
                        )
                    for key, value in tuple(diagnostics.items()):
                        if isinstance(value, (dict, list)):
                            diagnostics[key] = json.dumps(value, sort_keys=True)
                    optimization_rows.append(diagnostics)
                elif not benchmark.empty:
                    eligible = scores.reindex(benchmark.index).dropna().sort_values(
                        ascending=False
                    )
                    selected = eligible.head(self.settings.top_n)
                    if not selected.empty:
                        target = pd.Series(
                            1.0 / len(selected),
                            index=selected.index,
                            name="target_weight",
                        )

                if target is not None and not target.empty:
                    pending = PendingTarget(
                        signal_date=trade_date,
                        execution_date=execution_date,
                        target=target,
                        benchmark=benchmark,
                        covariance=audit_covariance,
                        industry_exposures=audit_industry_exposures,
                        style_exposures=audit_style_exposures,
                        liquidity=liquidity_snapshot,
                        risk_method=(
                            risk_estimate.method if self.optimizer is not None else None
                        ),
                        beta_method=(
                            risk_estimate.beta_method
                            if self.optimizer is not None
                            else None
                        ),
                    )
                    target_rows.extend(
                        {
                            "signal_date": trade_date,
                            "execution_date": execution_date,
                            "instrument": instrument,
                            "score": self._finite_or_none(scores.get(instrument, np.nan)),
                            "expected_return": self._finite_or_none(
                                expected_returns.get(instrument, np.nan)
                            ),
                            "benchmark_weight": float(benchmark.get(instrument, 0.0)),
                            "target_weight": float(weight),
                            "active_weight": float(
                                weight - benchmark.get(instrument, 0.0)
                            ),
                            "construction_method": construction_method,
                            "liquidity_contract": (
                                LIQUIDITY_CONTRACT_VERSION
                                if liquidity_snapshot is not None
                                else None
                            ),
                            "liquidity_reference_date": (
                                liquidity_snapshot.reference_date
                                if liquidity_snapshot is not None
                                else None
                            ),
                            "adv_cny": (
                                self._finite_or_none(
                                    liquidity_snapshot.adv_cny.get(
                                        instrument,
                                        np.nan,
                                    )
                                )
                                if liquidity_snapshot is not None
                                else None
                            ),
                            "adv_observations": (
                                int(
                                    liquidity_snapshot.observation_count.get(
                                        instrument,
                                        0,
                                    )
                                )
                                if liquidity_snapshot is not None
                                else 0
                            ),
                        }
                        for instrument, weight in target.items()
                        if weight > 1e-10
                    )

            progress.update(
                position + 1,
                context={
                    "optimization_attempts": len(optimization_rows),
                    "trade_date": trade_date,
                },
            )

        daily = pd.DataFrame(daily_rows)
        benchmark_nav = index_close / float(index_close.iloc[0])
        daily["benchmark_nav"] = daily["trade_date"].map(benchmark_nav)
        daily = enrich_active_performance(daily)

        trades = pd.DataFrame(trade_rows)
        targets = pd.DataFrame(target_rows)
        optimization = pd.DataFrame(optimization_rows)
        positions = pd.DataFrame(position_rows)
        constraint_audits = pd.DataFrame(constraint_audit_rows)
        metrics = calculate_backtest_metrics(daily, trades)
        metrics.update(summarize_constraint_audits(constraint_audits))
        return BacktestResult(
            daily=daily,
            trades=trades,
            targets=targets,
            optimization=optimization,
            positions=positions,
            constraint_audits=constraint_audits,
            metrics=metrics,
        )

    @staticmethod
    def _frames_by_date(
        frame: pd.DataFrame | None,
        *,
        date_column: str,
        drop_columns: set[str],
    ) -> dict[str, pd.DataFrame]:
        if frame is None or frame.empty:
            return {}
        required = {date_column, "instrument"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Context table is missing columns: {missing}")
        if frame.duplicated([date_column, "instrument"]).any():
            raise ValueError("Context date/instrument key is not unique")
        return {
            str(date): group.drop(columns=list(drop_columns)).set_index("instrument")
            for date, group in frame.groupby(date_column, sort=True)
        }

    @staticmethod
    def _suspensions_by_date(
        suspensions: pd.DataFrame | None,
    ) -> dict[str, set[str]]:
        return opening_suspensions_by_date(suspensions)

    def _liquidity_by_date(
        self,
        market_panel: pd.DataFrame,
    ) -> dict[str, LiquiditySnapshot]:
        if self.optimizer is None or not self.optimizer.settings.liquidity_enabled:
            return {}
        settings = self.optimizer.settings
        return build_trailing_adv_snapshots(
            market_panel,
            lookback=settings.adv_lookback,
            min_observations=settings.min_adv_observations,
        )

    def _portfolio_context(
        self,
        *,
        decision_date: str,
        names: list[str],
        pre_cash_weight: float,
        exposures: pd.DataFrame | None,
        restrictions: pd.DataFrame | None,
        liquidity: LiquiditySnapshot | None,
    ) -> PortfolioContext:
        if liquidity is not None and liquidity.reference_date != decision_date:
            raise ValueError(
                "Optimizer liquidity must be frozen on the decision date: "
                f"decision={decision_date}, reference={liquidity.reference_date}"
            )
        cannot_buy: set[str] = set()
        cannot_sell: set[str] = set()
        if restrictions is not None and not restrictions.empty:
            if "cannot_buy" in restrictions:
                cannot_buy = set(
                    restrictions.index[restrictions["cannot_buy"].astype(bool)].astype(str)
                )
            if "cannot_sell" in restrictions:
                cannot_sell = set(
                    restrictions.index[restrictions["cannot_sell"].astype(bool)].astype(str)
                )
        aligned_exposures = exposures.reindex(names) if exposures is not None else None
        if aligned_exposures is not None and not aligned_exposures.empty:
            # A carried holding can leave the current benchmark/feature universe.
            # Keep it visible to the exposure constraints as an explicit unknown
            # bucket instead of silently treating the row as zero exposure.
            missing_exposure = aligned_exposures.isna().any(axis=1)
            aligned_exposures = aligned_exposures.fillna(0.0)
            unknown_column = "__unknown_exposure__"
            if unknown_column not in aligned_exposures:
                aligned_exposures[unknown_column] = 0.0
            aligned_exposures.loc[missing_exposure, unknown_column] = 1.0
        # Initial construction remains unconstrained by ADV so the fully-invested
        # optimizer does not become infeasible solely because the account starts in cash.
        max_trade_weights = (
            liquidity.max_trade_weights(
                portfolio_aum_cny=self.optimizer.settings.portfolio_aum_cny,
                max_adv_participation=self.optimizer.settings.max_adv_participation,
            ).reindex(names)
            if liquidity is not None
            and self.optimizer is not None
            and pre_cash_weight <= 0.5
            else None
        )
        return PortfolioContext(
            exposures=aligned_exposures,
            cannot_buy=cannot_buy,
            cannot_sell=cannot_sell,
            max_trade_weights=max_trade_weights,
        )

    def _current_weights(
        self,
        holdings: dict[str, float],
        cash: float,
        day: pd.DataFrame,
        last_close: dict[str, float],
        *,
        price_column: str,
    ) -> tuple[pd.Series, float]:
        values = pd.Series(
            {
                instrument: units
                * self._price(instrument, day, last_close, price_column)
                for instrument, units in holdings.items()
                if units > 1e-14
            },
            dtype=float,
        )
        nav = float(values.sum() + cash)
        if nav <= 0:
            raise ArithmeticError("Portfolio NAV must be positive before optimization")
        return values / nav, cash / nav

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    def _execute_target(
        self,
        *,
        signal_date: str,
        trade_date: str,
        target: pd.Series,
        holdings: dict[str, float],
        cash: float,
        pre_nav: float,
        day: pd.DataFrame,
        last_close: dict[str, float],
        suspended_instruments: set[str],
        liquidity: LiquiditySnapshot | None,
    ) -> tuple[float, float, float, list[dict[str, Any]]]:
        if self.optimizer is not None and self.optimizer.settings.liquidity_enabled:
            if liquidity is not None and liquidity.reference_date >= trade_date:
                raise ValueError(
                    "Execution liquidity must predate the execution session: "
                    f"reference={liquidity.reference_date}, execution={trade_date}"
                )
            if liquidity is not None and liquidity.reference_date != signal_date:
                raise ValueError(
                    "Execution liquidity must be frozen on the signal date: "
                    f"signal={signal_date}, reference={liquidity.reference_date}"
                )
        records: list[dict[str, Any]] = []
        gross_total = 0.0
        cost_total = 0.0
        instruments = sorted(set(holdings) | set(target.index))

        current_values = {
            instrument: holdings.get(instrument, 0.0)
            * self._price(instrument, day, last_close, "adjusted_open")
            for instrument in instruments
        }

        for instrument in instruments:
            desired = float(target.get(instrument, 0.0)) * pre_nav
            requested = max(current_values[instrument] - desired, 0.0)
            if requested <= 1e-14:
                continue
            allowed, reason = self._can_trade(
                instrument,
                day,
                side="sell",
                suspended=instrument in suspended_instruments,
            )
            if not allowed:
                records.append(
                    self._trade_record(
                        signal_date,
                        trade_date,
                        instrument,
                        "sell",
                        "blocked",
                        requested,
                        0.0,
                        0.0,
                        0.0,
                        reason,
                    )
                )
                continue
            price = float(day.at[instrument, "adjusted_open"])
            capacity, capacity_reason = self._execution_capacity(
                instrument,
                liquidity,
                pre_nav,
            )
            if capacity <= 1e-14:
                records.append(
                    self._trade_record(
                        signal_date,
                        trade_date,
                        instrument,
                        "sell",
                        "blocked",
                        requested,
                        0.0,
                        0.0,
                        0.0,
                        capacity_reason,
                    )
                )
                continue
            gross = min(requested, holdings.get(instrument, 0.0) * price, capacity)
            linear_cost = gross * self.linear_rate
            stamp = gross * self._stamp_rate(trade_date)
            impact = self._impact_cost(gross, instrument, liquidity, pre_nav)
            holdings[instrument] = max(holdings.get(instrument, 0.0) - gross / price, 0.0)
            cash += gross - linear_cost - stamp - impact
            gross_total += gross
            cost_total += linear_cost + stamp + impact
            partial = gross < requested - 1e-12
            records.append(
                self._trade_record(
                    signal_date,
                    trade_date,
                    instrument,
                    "sell",
                    "partial" if partial else "filled",
                    requested,
                    gross,
                    linear_cost,
                    stamp,
                    capacity_reason if partial else "",
                    impact_cost=impact,
                )
            )

        buy_requests: dict[str, tuple[float, float, str]] = {}
        for instrument in target.index:
            price = self._price(instrument, day, last_close, "adjusted_open")
            current = holdings.get(instrument, 0.0) * price
            requested = max(float(target[instrument]) * pre_nav - current, 0.0)
            if requested <= 1e-14:
                continue
            allowed, reason = self._can_trade(
                instrument,
                day,
                side="buy",
                suspended=instrument in suspended_instruments,
            )
            if not allowed:
                records.append(
                    self._trade_record(
                        signal_date,
                        trade_date,
                        instrument,
                        "buy",
                        "blocked",
                        requested,
                        0.0,
                        0.0,
                        0.0,
                        reason,
                    )
                )
                continue
            capacity, capacity_reason = self._execution_capacity(
                instrument,
                liquidity,
                pre_nav,
            )
            if capacity <= 1e-14:
                records.append(
                    self._trade_record(
                        signal_date,
                        trade_date,
                        instrument,
                        "buy",
                        "blocked",
                        requested,
                        0.0,
                        0.0,
                        0.0,
                        capacity_reason,
                    )
                )
                continue
            buy_requests[instrument] = (
                requested,
                min(requested, capacity),
                capacity_reason,
            )

        cash_needed = sum(
            executable * (1.0 + self.linear_rate)
            + self._impact_cost(executable, instrument, liquidity, pre_nav)
            for instrument, (_, executable, _) in buy_requests.items()
        )
        scale = min(1.0, cash / cash_needed) if cash_needed > 0 else 0.0
        for instrument, (requested, executable, capacity_reason) in sorted(
            buy_requests.items()
        ):
            gross = executable * scale
            if gross <= 1e-14:
                records.append(
                    self._trade_record(
                        signal_date,
                        trade_date,
                        instrument,
                        "buy",
                        "blocked",
                        requested,
                        0.0,
                        0.0,
                        0.0,
                        "cash_constraint",
                    )
                )
                continue
            price = float(day.at[instrument, "adjusted_open"])
            linear_cost = gross * self.linear_rate
            impact = self._impact_cost(gross, instrument, liquidity, pre_nav)
            holdings[instrument] = holdings.get(instrument, 0.0) + gross / price
            cash -= gross + linear_cost + impact
            gross_total += gross
            cost_total += linear_cost + impact
            partial = gross < requested - 1e-12
            reason = ""
            if executable < requested - 1e-12:
                reason = capacity_reason
            elif scale < 1.0 - 1e-12:
                reason = "cash_constraint"
            records.append(
                self._trade_record(
                    signal_date,
                    trade_date,
                    instrument,
                    "buy",
                    "partial" if partial else "filled",
                    requested,
                    gross,
                    linear_cost,
                    0.0,
                    reason,
                    impact_cost=impact,
                )
            )
        if cash < -1e-10:
            raise ArithmeticError(f"Cash became negative after execution: {cash}")
        self._annotate_liquidity_audit(
            records,
            liquidity=liquidity,
            execution_day=day,
            pre_nav=pre_nav,
        )
        return max(cash, 0.0), gross_total, cost_total, records

    def _execution_capacity(
        self,
        instrument: str,
        liquidity: LiquiditySnapshot | None,
        pre_nav: float,
    ) -> tuple[float, str]:
        if self.optimizer is None or not self.optimizer.settings.liquidity_enabled:
            return float("inf"), ""
        if liquidity is None or instrument not in liquidity.adv_cny.index:
            return 0.0, "missing_ex_ante_adv"
        adv_cny = float(liquidity.adv_cny.at[instrument])
        if not np.isfinite(adv_cny) or adv_cny <= 0:
            return 0.0, "missing_ex_ante_adv"
        settings = self.optimizer.settings
        capacity_weight = (
            adv_cny
            * settings.max_adv_participation
            / settings.portfolio_aum_cny
        )
        return max(capacity_weight * pre_nav, 0.0), "ex_ante_adv_participation_cap"

    def _impact_cost(
        self,
        gross: float,
        instrument: str,
        liquidity: LiquiditySnapshot | None,
        pre_nav: float,
    ) -> float:
        if (
            gross <= 0
            or pre_nav <= 0
            or self.optimizer is None
            or not self.optimizer.settings.liquidity_enabled
        ):
            return 0.0
        if liquidity is None or instrument not in liquidity.adv_cny.index:
            return 0.0
        adv_cny = float(liquidity.adv_cny.at[instrument])
        if not np.isfinite(adv_cny) or adv_cny <= 0:
            return 0.0
        settings = self.optimizer.settings
        notional_cny = gross / pre_nav * settings.portfolio_aum_cny
        participation = notional_cny / adv_cny
        participation_ratio = max(
            participation / settings.max_adv_participation,
            0.0,
        )
        impact_rate = (
            settings.impact_bps_at_max_participation
            / 10000.0
            * np.sqrt(participation_ratio)
        )
        return float(gross * impact_rate)

    def _annotate_liquidity_audit(
        self,
        records: list[dict[str, Any]],
        *,
        liquidity: LiquiditySnapshot | None,
        execution_day: pd.DataFrame,
        pre_nav: float,
    ) -> None:
        if self.optimizer is None or not self.optimizer.settings.liquidity_enabled:
            return
        settings = self.optimizer.settings
        for record in records:
            instrument = str(record["instrument"])
            gross = float(record.get("gross_value", 0.0) or 0.0)
            requested = float(record.get("requested_value", 0.0) or 0.0)
            notional_scale = settings.portfolio_aum_cny / pre_nav
            adv_cny = (
                float(liquidity.adv_cny.at[instrument])
                if liquidity is not None and instrument in liquidity.adv_cny.index
                else np.nan
            )
            observations = (
                int(liquidity.observation_count.at[instrument])
                if liquidity is not None
                and instrument in liquidity.observation_count.index
                else 0
            )
            realized_amount = (
                float(execution_day.at[instrument, "amount_cny"])
                if instrument in execution_day.index
                and "amount_cny" in execution_day
                else np.nan
            )
            record.update(
                {
                    "liquidity_contract": LIQUIDITY_CONTRACT_VERSION,
                    "liquidity_reference_date": (
                        liquidity.reference_date if liquidity is not None else None
                    ),
                    "adv_cny": adv_cny,
                    "adv_observations": observations,
                    "requested_adv_participation": (
                        requested * notional_scale / adv_cny
                        if np.isfinite(adv_cny) and adv_cny > 0
                        else np.nan
                    ),
                    "executed_adv_participation": (
                        gross * notional_scale / adv_cny
                        if np.isfinite(adv_cny) and adv_cny > 0
                        else np.nan
                    ),
                    "execution_day_amount_cny": realized_amount,
                    "realized_day_participation": (
                        gross * notional_scale / realized_amount
                        if np.isfinite(realized_amount) and realized_amount > 0
                        else np.nan
                    ),
                }
            )

    def _can_trade(
        self,
        instrument: str,
        day: pd.DataFrame,
        *,
        side: str,
        suspended: bool = False,
    ) -> tuple[bool, str]:
        if suspended:
            return False, "suspended"
        if day.empty or instrument not in day.index:
            return False, "missing_execution_bar"
        row = day.loc[instrument]
        required = ("open", "adjusted_open", "up_limit", "down_limit")
        if any(not np.isfinite(float(row[column])) for column in required):
            return False, "missing_execution_or_limit_price"
        tolerance = self.settings.price_limit_tolerance
        limit_applicable = (
            bool(row["price_limit_applicable"])
            if "price_limit_applicable" in row.index
            else True
        )
        if (
            limit_applicable
            and side == "buy"
            and float(row["open"]) >= float(row["up_limit"]) * (1.0 - tolerance)
        ):
            return False, "open_at_up_limit"
        if (
            limit_applicable
            and side == "sell"
            and float(row["open"]) <= float(row["down_limit"]) * (1.0 + tolerance)
        ):
            return False, "open_at_down_limit"
        return True, ""

    @staticmethod
    def _price(
        instrument: str,
        day: pd.DataFrame,
        last_close: dict[str, float],
        price_column: str,
    ) -> float:
        if not day.empty and instrument in day.index:
            value = float(day.at[instrument, price_column])
            if np.isfinite(value) and value > 0:
                return value
        value = float(last_close.get(instrument, np.nan))
        if not np.isfinite(value) or value <= 0:
            return 0.0
        return value

    def _portfolio_value(
        self,
        holdings: dict[str, float],
        cash: float,
        day: pd.DataFrame,
        last_close: dict[str, float],
        *,
        price_column: str,
    ) -> float:
        return cash + sum(
            units * self._price(instrument, day, last_close, price_column)
            for instrument, units in holdings.items()
        )

    def _stamp_rate(self, trade_date: str) -> float:
        if trade_date < self.settings.stamp_duty_change_date:
            return self.settings.stamp_duty_before
        return self.settings.stamp_duty_after

    @staticmethod
    def _trade_record(
        signal_date: str,
        trade_date: str,
        instrument: str,
        side: str,
        status: str,
        requested_value: float,
        gross_value: float,
        linear_cost: float,
        stamp_duty: float,
        reason: str,
        impact_cost: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "instrument": instrument,
            "side": side,
            "status": status,
            "requested_value": requested_value,
            "gross_value": gross_value,
            "linear_cost": linear_cost,
            "stamp_duty": stamp_duty,
            "impact_cost": impact_cost,
            "fill_ratio": (
                gross_value / requested_value if requested_value > 0 else 0.0
            ),
            "reason": reason,
        }
