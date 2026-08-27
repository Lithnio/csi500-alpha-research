from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import ResearchSettings
from csi500_alpha.execution.tradeability import opening_suspensions_by_date
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.universe import benchmark_weights_asof, select_rebalance_dates
from csi500_alpha.risk.model import LedoitWolfRiskModel


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    targets: pd.DataFrame
    optimization: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PortfolioContext:
    exposures: pd.DataFrame | None
    cannot_buy: set[str]
    cannot_sell: set[str]
    max_trade_weights: pd.Series | None


class SmokeEventBacktester:
    """Research event loop proving signal-time, execution and cost boundaries."""

    def __init__(
        self,
        settings: ResearchSettings,
        *,
        risk_model: LedoitWolfRiskModel | None = None,
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
        portfolio_restrictions: pd.DataFrame | None = None,
        suspensions: pd.DataFrame | None = None,
        start_date: str,
        end_date: str,
        rebalance_dates: Sequence[str] | None = None,
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
        restrictions_by_date = self._frames_by_date(
            portfolio_restrictions,
            date_column="trade_date",
            drop_columns={"trade_date"},
        )
        suspended_by_date = self._suspensions_by_date(suspensions)
        liquidity_caps_by_date = self._liquidity_caps_by_date(market_panel)
        index_close = (
            index_bars.set_index("trade_date")["close"].sort_index().reindex(backtest_dates).ffill()
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
        pending: tuple[str, pd.Series] | None = None
        daily_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        optimization_rows: list[dict[str, Any]] = []

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
                signal_date, execution_target = pending
                cash, gross, cost, executed_rows = self._execute_target(
                    signal_date=signal_date,
                    trade_date=trade_date,
                    target=execution_target,
                    holdings=holdings,
                    cash=cash,
                    pre_nav=pre_nav,
                    day=day,
                    last_close=last_close,
                    suspended_instruments=suspended_by_date.get(trade_date, set()),
                )
                daily_gross += gross
                daily_cost += cost
                trade_rows.extend(executed_rows)
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
                benchmark = benchmark_weights_asof(benchmark_weights, trade_date)
                execution_date = backtest_dates[position + 1]
                target: pd.Series | None = None
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
                        liquidity_caps=liquidity_caps_by_date.get(trade_date),
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
                        cannot_buy=portfolio_context.cannot_buy,
                        cannot_sell=portfolio_context.cannot_sell,
                        max_trade_weights=portfolio_context.max_trade_weights,
                    )
                    target = optimization_result.target
                    construction_method = "active_optimizer"
                    diagnostics = optimization_result.diagnostics.copy()
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
                    pending = (trade_date, target)
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
                        }
                        for instrument, weight in target.items()
                        if weight > 1e-10
                    )

        daily = pd.DataFrame(daily_rows)
        benchmark_nav = index_close / float(index_close.iloc[0])
        daily["benchmark_nav"] = daily["trade_date"].map(benchmark_nav)
        daily["portfolio_return"] = daily["nav"].pct_change().fillna(0.0)
        daily["benchmark_return"] = daily["benchmark_nav"].pct_change().fillna(0.0)
        daily["active_return"] = daily["portfolio_return"] - daily["benchmark_return"]
        daily["active_nav"] = (1.0 + daily["active_return"]).cumprod()

        trades = pd.DataFrame(trade_rows)
        targets = pd.DataFrame(target_rows)
        optimization = pd.DataFrame(optimization_rows)
        metrics = self._metrics(daily, trades)
        return BacktestResult(
            daily=daily,
            trades=trades,
            targets=targets,
            optimization=optimization,
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

    def _liquidity_caps_by_date(
        self,
        market_panel: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        if self.optimizer is None or not self.optimizer.settings.liquidity_enabled:
            return {}
        if "amount_cny" not in market_panel:
            raise ValueError("Liquidity constraints require market_panel.amount_cny")
        settings = self.optimizer.settings
        ordered = market_panel[["trade_date", "instrument", "amount_cny"]].copy()
        ordered["amount_cny"] = pd.to_numeric(ordered["amount_cny"], errors="coerce")
        ordered = ordered.sort_values(["instrument", "trade_date"])
        rolling = ordered.groupby("instrument", sort=False)["amount_cny"].transform(
            lambda values: values.rolling(
                settings.adv_lookback,
                min_periods=settings.min_adv_observations,
            ).mean()
        )
        ordered["max_trade_weight"] = (
            rolling
            * settings.max_adv_participation
            / settings.portfolio_aum_cny
        )
        return {
            str(date): group.set_index("instrument")["max_trade_weight"]
            for date, group in ordered.groupby("trade_date", sort=True)
        }

    def _portfolio_context(
        self,
        *,
        decision_date: str,
        names: list[str],
        pre_cash_weight: float,
        exposures: pd.DataFrame | None,
        restrictions: pd.DataFrame | None,
        liquidity_caps: pd.Series | None,
    ) -> PortfolioContext:
        del decision_date
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
            liquidity_caps.reindex(names)
            if liquidity_caps is not None and pre_cash_weight <= 0.5
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
    ) -> tuple[float, float, float, list[dict[str, Any]]]:
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
                day,
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
            impact = self._impact_cost(gross, instrument, day, pre_nav)
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
                day,
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
            + self._impact_cost(executable, instrument, day, pre_nav)
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
            impact = self._impact_cost(gross, instrument, day, pre_nav)
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
        return max(cash, 0.0), gross_total, cost_total, records

    def _execution_capacity(
        self,
        instrument: str,
        day: pd.DataFrame,
        pre_nav: float,
    ) -> tuple[float, str]:
        if self.optimizer is None or not self.optimizer.settings.liquidity_enabled:
            return float("inf"), ""
        if day.empty or instrument not in day.index or "amount_cny" not in day:
            return 0.0, "missing_execution_liquidity"
        amount = float(day.at[instrument, "amount_cny"])
        if not np.isfinite(amount) or amount <= 0:
            return 0.0, "missing_execution_liquidity"
        settings = self.optimizer.settings
        capacity_weight = (
            amount * settings.max_adv_participation / settings.portfolio_aum_cny
        )
        return max(capacity_weight * pre_nav, 0.0), "volume_participation_cap"

    def _impact_cost(
        self,
        gross: float,
        instrument: str,
        day: pd.DataFrame,
        pre_nav: float,
    ) -> float:
        if (
            gross <= 0
            or pre_nav <= 0
            or self.optimizer is None
            or not self.optimizer.settings.liquidity_enabled
        ):
            return 0.0
        amount = float(day.at[instrument, "amount_cny"])
        if not np.isfinite(amount) or amount <= 0:
            return 0.0
        settings = self.optimizer.settings
        notional_cny = gross / pre_nav * settings.portfolio_aum_cny
        participation = notional_cny / amount
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

    @staticmethod
    def _metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
        observations = max(len(daily) - 1, 1)
        years = observations / 252.0
        final_nav = float(daily["nav"].iloc[-1])
        final_benchmark = float(daily["benchmark_nav"].iloc[-1])
        active = daily["active_return"]
        tracking_error = float(active.std(ddof=1) * np.sqrt(252)) if len(active) > 1 else 0.0
        annual_active = float(active.mean() * 252)
        information_ratio = annual_active / tracking_error if tracking_error > 0 else np.nan
        peak = daily["nav"].cummax()
        drawdown = daily["nav"] / peak - 1.0
        costs = (
            float(
                trades["linear_cost"].sum()
                + trades["stamp_duty"].sum()
                + trades["impact_cost"].sum()
            )
            if not trades.empty
            else 0.0
        )
        return {
            "start_date": str(daily["trade_date"].iloc[0]),
            "end_date": str(daily["trade_date"].iloc[-1]),
            "observations": len(daily),
            "final_nav": final_nav,
            "final_benchmark_nav": final_benchmark,
            "total_return": final_nav / float(daily["nav"].iloc[0]) - 1.0,
            "benchmark_total_return": final_benchmark - 1.0,
            "annualized_return": final_nav ** (1.0 / years) - 1.0 if years > 0 else np.nan,
            "annualized_active_return": annual_active,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "max_drawdown": float(drawdown.min()),
            "average_turnover": float(daily["turnover"].fillna(0.0).mean()),
            "transaction_cost": costs,
            "filled_orders": int((trades["status"] == "filled").sum()) if not trades.empty else 0,
            "partial_orders": (
                int((trades["status"] == "partial").sum()) if not trades.empty else 0
            ),
            "executed_orders": (
                int(trades["status"].isin(["filled", "partial"]).sum())
                if not trades.empty
                else 0
            ),
            "blocked_orders": int((trades["status"] == "blocked").sum()) if not trades.empty else 0,
        }
