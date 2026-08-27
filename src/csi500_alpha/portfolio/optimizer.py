from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.risk.model import RiskEstimate


@dataclass(frozen=True)
class OptimizationResult:
    target: pd.Series | None
    diagnostics: dict[str, Any]


class ActivePortfolioOptimizer:
    """Long-only, fully invested optimizer relative to a dynamic benchmark."""

    def __init__(
        self,
        settings: OptimizerSettings,
        research: ResearchSettings,
        risk: RiskSettings,
    ) -> None:
        self.settings = settings
        self.research = research
        self.risk = risk

    def solve(
        self,
        *,
        decision_date: str,
        execution_date: str,
        expected_returns: pd.Series,
        benchmark: pd.Series,
        pre_weights: pd.Series,
        pre_cash_weight: float,
        risk_estimate: RiskEstimate,
        exposures: pd.DataFrame | None = None,
        cannot_buy: set[str] | None = None,
        cannot_sell: set[str] | None = None,
        max_trade_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        names = risk_estimate.covariance.index
        covariance_frame = risk_estimate.covariance
        covariance_valid = (
            not names.empty
            and names.is_unique
            and covariance_frame.columns.equals(names)
            and covariance_frame.shape == (len(names), len(names))
            and np.isfinite(covariance_frame.to_numpy(dtype=float)).all()
        )
        if not covariance_valid:
            return OptimizationResult(
                target=None,
                diagnostics={
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "status": "invalid_input",
                    "action": "hold_pre_trade_portfolio",
                    "error": "invalid_risk_covariance",
                },
            )
        benchmark = pd.to_numeric(
            benchmark.reindex(names, fill_value=0.0),
            errors="coerce",
        )
        benchmark_total = float(benchmark.sum())
        if (
            not np.isfinite(benchmark.to_numpy(dtype=float)).all()
            or (benchmark < 0).any()
            or not np.isfinite(benchmark_total)
            or benchmark_total <= 0
        ):
            return OptimizationResult(
                target=None,
                diagnostics={
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "status": "invalid_input",
                    "action": "hold_pre_trade_portfolio",
                    "error": "invalid_benchmark_weights",
                },
            )
        benchmark /= benchmark_total
        pre_weights = pd.to_numeric(
            pre_weights.reindex(names, fill_value=0.0),
            errors="coerce",
        )
        if (
            not np.isfinite(pre_weights.to_numpy(dtype=float)).all()
            or (pre_weights < 0).any()
            or not np.isfinite(pre_cash_weight)
            or not 0.0 <= pre_cash_weight <= 1.0
        ):
            return OptimizationResult(
                target=None,
                diagnostics={
                    "decision_date": decision_date,
                    "execution_date": execution_date,
                    "status": "invalid_input",
                    "action": "hold_pre_trade_portfolio",
                    "error": "invalid_pre_trade_weights",
                },
            )
        pre_cash_weight = float(pre_cash_weight)
        mu = (
            pd.to_numeric(expected_returns.reindex(names), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        signal_valid = mu.notna()
        mu = mu.fillna(0.0)
        risk_eligible = risk_estimate.eligible.reindex(names, fill_value=False)
        active_eligible = signal_valid & risk_eligible

        exposures_enabled = exposures is not None and not exposures.empty
        matrix: np.ndarray | None = None
        exposure_eligible = pd.Series(True, index=names)
        if exposures_enabled and exposures is not None:
            aligned_exposures = exposures.reindex(index=names)
            aligned_exposures = aligned_exposures.apply(pd.to_numeric, errors="coerce")
            exposure_eligible = aligned_exposures.notna().all(axis=1)
            active_eligible &= exposure_eligible
            matrix = aligned_exposures.fillna(0.0).to_numpy(dtype=float)

        covariance = covariance_frame.to_numpy(dtype=float)
        count = len(names)
        w = cp.Variable(count, name="target_weight")
        b = benchmark.to_numpy(dtype=float)
        pre = pre_weights.to_numpy(dtype=float)
        pre_active = pre - b
        alpha = mu.to_numpy(dtype=float)
        active = w - b
        delta = w - pre
        buy_rate = self.research.linear_cost_bps / 10000.0
        sell_rate = buy_rate + self._stamp_rate(execution_date)
        horizon_covariance = covariance * self.settings.risk_horizon_days

        predicted_alpha = alpha @ active
        risk_penalty = self.settings.risk_aversion * cp.quad_form(
            active,
            cp.psd_wrap(horizon_covariance),
        )
        expected_cost = buy_rate * cp.sum(cp.pos(delta))
        expected_cost += sell_rate * cp.sum(cp.pos(-delta))
        regularization = self.settings.l2_penalty * cp.sum_squares(active)
        objective = cp.Maximize(
            predicted_alpha - risk_penalty - expected_cost - regularization
        )

        turnover = 0.5 * (cp.norm1(delta) + pre_cash_weight)
        configured_turnover_limit = (
            self.settings.initial_turnover_cap
            if pre_cash_weight > 0.5
            else self.settings.turnover_cap
        )
        # A fully invested target must deploy at least the existing cash weight.
        turnover_limit = max(configured_turnover_limit, pre_cash_weight)
        name_upper = np.maximum(self.settings.name_cap, pre)
        active_upper = np.maximum(self.settings.active_cap, pre_active)
        active_lower = np.minimum(-self.settings.active_cap, pre_active)
        constraints: list[cp.Constraint] = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= name_upper,
            active <= active_upper,
            active >= active_lower,
            turnover <= turnover_limit,
        ]
        ineligible_to_buy = ~active_eligible.to_numpy(dtype=bool)
        if ineligible_to_buy.any():
            # A missing signal, short risk history, or missing exposure must never
            # create a new position. Existing holdings may still be reduced using
            # the conservative fallback covariance and normal execution limits.
            # This is important for constituents that leave the benchmark: freezing
            # them at their pre-trade weight would strand stale positions forever.
            constraints.append(w[ineligible_to_buy] <= pre[ineligible_to_buy])

        cannot_buy = cannot_buy or set()
        cannot_sell = cannot_sell or set()
        name_to_position = {name: position for position, name in enumerate(names)}
        buy_positions = [name_to_position[name] for name in cannot_buy if name in name_to_position]
        sell_positions = [
            name_to_position[name] for name in cannot_sell if name in name_to_position
        ]
        if buy_positions:
            constraints.append(w[buy_positions] <= pre[buy_positions])
        if sell_positions:
            constraints.append(w[sell_positions] >= pre[sell_positions])

        if matrix is not None:
            pre_exposure = matrix.T @ pre_active
            exposure_upper = np.maximum(self.settings.exposure_cap, pre_exposure)
            exposure_lower = np.minimum(-self.settings.exposure_cap, pre_exposure)
            constraints.extend(
                [
                    matrix.T @ active <= exposure_upper,
                    matrix.T @ active >= exposure_lower,
                ]
            )
        else:
            pre_exposure = None
            exposure_upper = None
            exposure_lower = None

        trade_caps_enabled = max_trade_weights is not None
        trade_caps: np.ndarray | None = None
        if max_trade_weights is not None:
            aligned_caps = (
                pd.to_numeric(max_trade_weights.reindex(names), errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            trade_caps = aligned_caps.to_numpy(dtype=float)
            constraints.append(cp.abs(delta) <= trade_caps)

        problem = cp.Problem(objective, constraints)
        attempts: list[dict[str, Any]] = []
        solved_with: str | None = None
        for solver in self.settings.solvers:
            try:
                solver_options: dict[str, Any] = {}
                if solver == "OSQP":
                    solver_options = {
                        "eps_abs": self.settings.feasibility_tolerance / 10.0,
                        "eps_rel": self.settings.feasibility_tolerance / 10.0,
                        "max_iter": 100_000,
                        "polishing": True,
                    }
                elif solver == "CLARABEL":
                    solver_options = {
                        "tol_feas": self.settings.feasibility_tolerance / 10.0,
                        "tol_gap_abs": self.settings.feasibility_tolerance / 10.0,
                        "tol_gap_rel": self.settings.feasibility_tolerance / 10.0,
                    }
                problem.solve(
                    solver=solver,
                    warm_start=True,
                    verbose=False,
                    **solver_options,
                )
                attempts.append({"solver": solver, "status": problem.status})
            except cp.error.SolverError as exc:
                attempts.append(
                    {"solver": solver, "status": "solver_error", "error": str(exc)}
                )
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                solved_with = solver
                break

        base_diagnostics: dict[str, Any] = {
            "decision_date": decision_date,
            "execution_date": execution_date,
            "risk_method": risk_estimate.method,
            "risk_observations": risk_estimate.observations,
            "risk_eligible": int(risk_eligible.sum()),
            "active_eligible": int(active_eligible.sum()),
            "exposure_eligible": int(exposure_eligible.sum()),
            "universe_size": count,
            "expected_return_min": (
                float(mu.loc[signal_valid].min()) if signal_valid.any() else None
            ),
            "expected_return_max": (
                float(mu.loc[signal_valid].max()) if signal_valid.any() else None
            ),
            "turnover_limit": turnover_limit,
            "configured_turnover_limit": configured_turnover_limit,
            "exposures_enabled": exposures_enabled,
            "trade_caps_enabled": trade_caps_enabled,
            "constraint_policy": "existing_breaches_may_persist_but_not_worsen",
            "ineligible_position_policy": "cannot_increase_may_reduce",
            "ineligible_to_buy": int(ineligible_to_buy.sum()),
            "ineligible_held": int(
                ((pre > self.settings.feasibility_tolerance) & ineligible_to_buy).sum()
            ),
            "preexisting_name_cap_breaches": int(
                (
                    pre
                    > self.settings.name_cap
                    + self.settings.feasibility_tolerance
                ).sum()
            ),
            "preexisting_active_cap_breaches": int(
                (
                    np.abs(pre_active)
                    > self.settings.active_cap
                    + self.settings.feasibility_tolerance
                ).sum()
            ),
            "preexisting_exposure_cap_breaches": (
                int(
                    (
                        np.abs(pre_exposure)
                        > self.settings.exposure_cap
                        + self.settings.feasibility_tolerance
                    ).sum()
                )
                if pre_exposure is not None
                else 0
            ),
            "attempts": attempts,
        }
        if solved_with is None or w.value is None:
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "status": problem.status or "not_solved",
                    "action": "hold_pre_trade_portfolio",
                },
            )

        target_array = np.asarray(w.value, dtype=float).reshape(-1)
        if target_array.size != count or not np.isfinite(target_array).all():
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "status": "invalid_solver_output",
                    "solver_status": problem.status,
                    "action": "hold_pre_trade_portfolio",
                    "solver": solved_with,
                },
            )
        target_array[np.abs(target_array) < 1e-10] = 0.0
        target_array = np.maximum(target_array, 0.0)
        target_total = float(target_array.sum())
        if not np.isfinite(target_total) or target_total <= 0:
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "status": "invalid_solver_output",
                    "solver_status": problem.status,
                    "action": "hold_pre_trade_portfolio",
                    "solver": solved_with,
                },
            )
        target_array /= target_total
        target = pd.Series(target_array, index=names, name="target_weight")
        active_array = target_array - b
        delta_array = target_array - pre
        realized_turnover = 0.5 * (np.abs(delta_array).sum() + pre_cash_weight)
        ex_ante_te = float(
            np.sqrt(
                max(active_array @ covariance @ active_array, 0.0)
                * self.risk.annualization
            )
        )
        expected_cost_value = float(
            buy_rate * np.maximum(delta_array, 0.0).sum()
            + sell_rate * np.maximum(-delta_array, 0.0).sum()
        )
        violations = {
            "budget": abs(float(target_array.sum()) - 1.0),
            "long_only": max(0.0, -float(target_array.min())),
            "name_cap": max(0.0, float((target_array - name_upper).max())),
            "active_cap": max(
                0.0,
                float((active_array - active_upper).max()),
                float((active_lower - active_array).max()),
            ),
            "turnover": max(0.0, realized_turnover - turnover_limit),
            "ineligible_buy": (
                max(
                    0.0,
                    float(delta_array[ineligible_to_buy].max()),
                )
                if ineligible_to_buy.any()
                else 0.0
            ),
            "cannot_buy": (
                max(
                    0.0,
                    float((target_array[buy_positions] - pre[buy_positions]).max()),
                )
                if buy_positions
                else 0.0
            ),
            "cannot_sell": (
                max(
                    0.0,
                    float((pre[sell_positions] - target_array[sell_positions]).max()),
                )
                if sell_positions
                else 0.0
            ),
            "exposure": (
                max(
                    0.0,
                    float((matrix.T @ active_array - exposure_upper).max()),
                    float((exposure_lower - matrix.T @ active_array).max()),
                )
                if (
                    matrix is not None
                    and matrix.shape[1] > 0
                    and exposure_upper is not None
                    and exposure_lower is not None
                )
                else 0.0
            ),
            "trade_cap": (
                max(
                    0.0,
                    float((np.abs(delta_array) - trade_caps).max()),
                )
                if trade_caps is not None
                else 0.0
            ),
        }
        maximum_violation = max(violations.values())
        if maximum_violation > self.settings.feasibility_tolerance:
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "status": "postsolve_infeasible",
                    "solver_status": problem.status,
                    "action": "hold_pre_trade_portfolio",
                    "solver": solved_with,
                    "turnover": float(realized_turnover),
                    "maximum_violation": maximum_violation,
                    "violations": violations,
                },
            )
        return OptimizationResult(
            target=target,
            diagnostics={
                **base_diagnostics,
                "status": problem.status,
                "action": "trade_to_target",
                "solver": solved_with,
                "objective": float(problem.value),
                "predicted_active_return": float(alpha @ active_array),
                "ex_ante_tracking_error": ex_ante_te,
                "expected_cost": expected_cost_value,
                "turnover": float(realized_turnover),
                "configured_active_cap_breaches_after": int(
                    (
                        np.abs(active_array)
                        > self.settings.active_cap
                        + self.settings.feasibility_tolerance
                    ).sum()
                ),
                "configured_exposure_cap_breaches_after": (
                    int(
                        (
                            np.abs(matrix.T @ active_array)
                            > self.settings.exposure_cap
                            + self.settings.feasibility_tolerance
                        ).sum()
                    )
                    if matrix is not None
                    else 0
                ),
                "maximum_absolute_active_weight": float(np.abs(active_array).max()),
                "maximum_violation": maximum_violation,
                "violations": violations,
            },
        )

    def _stamp_rate(self, execution_date: str) -> float:
        if execution_date < self.research.stamp_duty_change_date:
            return self.research.stamp_duty_before
        return self.research.stamp_duty_after
