from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import cvxpy as cp
import numpy as np
import pandas as pd

from csi500_alpha.config import OptimizerSettings, ResearchSettings, RiskSettings
from csi500_alpha.risk.model import RiskEstimate


@dataclass(frozen=True)
class OptimizationResult:
    target: pd.Series | None
    diagnostics: dict[str, Any]


class _GapProfile(TypedDict):
    score: float
    name_breach_count: int
    active_breach_count: int
    exposure_breach_count: int
    maximum_name_breach: float
    maximum_active_breach: float
    maximum_exposure_breach: float
    active_beta: float | None
    beta_breach: float
    tracking_error: float
    tracking_error_breach: float
    tracking_error_utilization: float
    maximum_breach: float


class ActivePortfolioOptimizer:
    """Long-only benchmark-relative optimizer with lexicographic repair."""

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
        market_beta: pd.Series | None = None,
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
            return self._invalid_result(
                decision_date,
                execution_date,
                "invalid_risk_covariance",
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
            return self._invalid_result(
                decision_date,
                execution_date,
                "invalid_benchmark_weights",
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
            return self._invalid_result(
                decision_date,
                execution_date,
                "invalid_pre_trade_weights",
            )
        pre_cash_weight = float(pre_cash_weight)

        mu = pd.to_numeric(
            expected_returns.reindex(names),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        signal_valid = mu.notna()
        global_signal_inactive = not bool(signal_valid.any())
        if global_signal_inactive:
            # A selector-wide shutdown is a zero-alpha state. It should still
            # repair risk constraints and converge toward the benchmark.
            signal_valid[:] = True
        mu = mu.fillna(0.0)
        risk_eligible = risk_estimate.eligible.reindex(names, fill_value=False)
        active_eligible = signal_valid & risk_eligible

        exposures_enabled = exposures is not None and not exposures.empty
        matrix: np.ndarray | None = None
        exposure_eligible = pd.Series(True, index=names)
        if exposures_enabled and exposures is not None:
            aligned_exposures = exposures.reindex(index=names).apply(
                pd.to_numeric,
                errors="coerce",
            )
            exposure_eligible = aligned_exposures.notna().all(axis=1)
            active_eligible &= exposure_eligible
            matrix = aligned_exposures.fillna(0.0).to_numpy(dtype=float)

        beta_enabled = self.settings.beta_constraint_enabled
        beta_eligible = pd.Series(True, index=names)
        beta_values: np.ndarray | None = None
        if beta_enabled:
            if market_beta is None or market_beta.empty:
                return self._invalid_result(
                    decision_date,
                    execution_date,
                    "missing_market_beta_exposure",
                )
            aligned_beta = pd.to_numeric(
                market_beta.reindex(names),
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            beta_eligible = aligned_beta.notna()
            if not beta_eligible.any():
                return self._invalid_result(
                    decision_date,
                    execution_date,
                    "empty_market_beta_exposure",
                )
            active_eligible &= beta_eligible
            beta_values = aligned_beta.fillna(0.0).to_numpy(dtype=float)

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
        annual_covariance = covariance * self.risk.annualization
        predicted_alpha = alpha @ active
        risk_penalty = self.settings.risk_aversion * cp.quad_form(
            active,
            cp.psd_wrap(horizon_covariance),
        )
        expected_cost = buy_rate * cp.sum(cp.pos(delta))
        expected_cost += sell_rate * cp.sum(cp.pos(-delta))
        regularization = self.settings.l2_penalty * cp.sum_squares(active)
        alpha_objective = cp.Maximize(
            predicted_alpha - risk_penalty - expected_cost - regularization
        )

        configured_turnover_limit = (
            self.settings.initial_turnover_cap
            if pre_cash_weight > 0.5
            else self.settings.turnover_cap
        )
        # A fully invested target must deploy at least the existing cash weight.
        turnover_limit = max(configured_turnover_limit, pre_cash_weight)
        turnover = 0.5 * (cp.norm1(delta) + pre_cash_weight)
        constraints: list[cp.Constraint] = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            turnover <= turnover_limit,
        ]

        ineligible_to_buy = ~active_eligible.to_numpy(dtype=bool)
        inactive_upper = np.maximum(pre, b)
        if ineligible_to_buy.any():
            # Missing alpha or risk evidence cannot create an active overweight;
            # benchmark ownership remains permissible.
            constraints.append(w[ineligible_to_buy] <= inactive_upper[ineligible_to_buy])

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

        # Configured portfolio limits are soft only when hard trading constraints
        # make them infeasible. Gap variables support the strict-feasibility probe
        # and, when needed, the normalized repair problem below.
        name_gap = cp.Variable(count, nonneg=True, name="name_gap")
        active_upper_gap = cp.Variable(
            count,
            nonneg=True,
            name="active_upper_gap",
        )
        active_lower_gap = cp.Variable(
            count,
            nonneg=True,
            name="active_lower_gap",
        )
        constraints.extend(
            [
                w <= self.settings.name_cap + name_gap,
                active <= self.settings.active_cap + active_upper_gap,
                active >= -self.settings.active_cap - active_lower_gap,
            ]
        )
        strict_gap_constraints: list[cp.Constraint] = [
            name_gap == 0.0,
            active_upper_gap == 0.0,
            active_lower_gap == 0.0,
        ]
        repair_terms: list[cp.Expression] = [
            cp.sum(name_gap) / self.settings.name_cap,
            cp.sum(active_upper_gap + active_lower_gap) / self.settings.active_cap,
        ]

        pre_exposure: np.ndarray | None = None
        if matrix is not None:
            exposure_count = matrix.shape[1]
            exposure_upper_gap = cp.Variable(
                exposure_count,
                nonneg=True,
                name="exposure_upper_gap",
            )
            exposure_lower_gap = cp.Variable(
                exposure_count,
                nonneg=True,
                name="exposure_lower_gap",
            )
            exposure_active = matrix.T @ active
            constraints.extend(
                [
                    exposure_active <= self.settings.exposure_cap + exposure_upper_gap,
                    exposure_active >= -self.settings.exposure_cap - exposure_lower_gap,
                ]
            )
            strict_gap_constraints.extend(
                [
                    exposure_upper_gap == 0.0,
                    exposure_lower_gap == 0.0,
                ]
            )
            repair_terms.append(
                cp.sum(exposure_upper_gap + exposure_lower_gap) / self.settings.exposure_cap
            )
            pre_exposure = matrix.T @ pre_active

        pre_active_beta: float | None = None
        if beta_values is not None:
            beta_upper_gap = cp.Variable(nonneg=True, name="beta_upper_gap")
            beta_lower_gap = cp.Variable(nonneg=True, name="beta_lower_gap")
            target_active_beta = beta_values @ active
            constraints.extend(
                [
                    target_active_beta <= self.settings.beta_active_cap + beta_upper_gap,
                    target_active_beta >= -self.settings.beta_active_cap - beta_lower_gap,
                ]
            )
            strict_gap_constraints.extend(
                [
                    beta_upper_gap == 0.0,
                    beta_lower_gap == 0.0,
                ]
            )
            repair_terms.append((beta_upper_gap + beta_lower_gap) / self.settings.beta_active_cap)
            pre_active_beta = float(beta_values @ pre_active)

        tracking_variance_gap = cp.Variable(
            nonneg=True,
            name="tracking_variance_gap",
        )
        tracking_variance = cp.quad_form(
            active,
            cp.psd_wrap(annual_covariance),
        )
        tracking_variance_cap = self.settings.tracking_error_cap**2
        constraints.append(tracking_variance <= tracking_variance_cap + tracking_variance_gap)
        strict_gap_constraints.append(tracking_variance_gap == 0.0)
        repair_terms.append(tracking_variance_gap / tracking_variance_cap)
        repair_score = cp.sum(cp.hstack(repair_terms))

        pre_profile = _configured_gap_profile(
            weights=pre,
            benchmark=b,
            settings=self.settings,
            exposures=matrix,
            beta_values=beta_values,
            covariance=covariance,
            annualization=self.risk.annualization,
        )
        risk_diagnostics = {
            f"risk_{key}": value
            for key, value in risk_estimate.diagnostics.items()
        }
        base_diagnostics: dict[str, Any] = {
            "decision_date": decision_date,
            "execution_date": execution_date,
            "risk_method": risk_estimate.method,
            "beta_method": risk_estimate.beta_method,
            "risk_observations": risk_estimate.observations,
            "risk_eligible": int(risk_eligible.sum()),
            "active_eligible": int(active_eligible.sum()),
            "exposure_eligible": int(exposure_eligible.sum()),
            "beta_constraint_enabled": beta_enabled,
            "beta_eligible": int(beta_eligible.sum()),
            "beta_missing_benchmark_weight": (
                float(benchmark.loc[~beta_eligible].sum()) if beta_enabled else 0.0
            ),
            "universe_size": count,
            "expected_return_min": (
                float(mu.loc[signal_valid].min()) if signal_valid.any() else None
            ),
            "expected_return_max": (
                float(mu.loc[signal_valid].max()) if signal_valid.any() else None
            ),
            "alpha_state": (
                "inactive_zero_alpha" if global_signal_inactive else "active_or_partially_available"
            ),
            "turnover_limit": turnover_limit,
            "configured_turnover_limit": configured_turnover_limit,
            "tracking_error_cap": self.settings.tracking_error_cap,
            "exposures_enabled": exposures_enabled,
            "trade_caps_enabled": trade_caps_enabled,
            "constraint_policy": "lexicographic_minimum_configured_gap",
            "repair_objective": "normalized_l1_constraint_gap",
            "ineligible_position_policy": ("cannot_exceed_pretrade_or_benchmark_may_reduce"),
            "ineligible_to_buy": int(ineligible_to_buy.sum()),
            "ineligible_held": int(
                ((pre > self.settings.feasibility_tolerance) & ineligible_to_buy).sum()
            ),
            "preexisting_name_cap_breaches": pre_profile["name_breach_count"],
            "preexisting_active_cap_breaches": pre_profile["active_breach_count"],
            "preexisting_exposure_cap_breaches": pre_profile["exposure_breach_count"],
            "preexisting_beta_cap_breach": pre_profile["beta_breach"],
            "preexisting_tracking_error": pre_profile["tracking_error"],
            "preexisting_tracking_error_cap_breach": pre_profile["tracking_error_breach"],
            "preexisting_configured_gap_score": pre_profile["score"],
            "preexisting_industry_exposure_count": (
                int(len(pre_exposure)) if pre_exposure is not None else 0
            ),
            "preexisting_active_beta": pre_active_beta,
            **risk_diagnostics,
            "attempts": [],
        }

        # Zero gap is the lexicographic optimum whenever the strict configured
        # limits are feasible. Try that one-problem path first; only pay for the
        # repair and secondary alpha solves when hard trading constraints make
        # a configured limit unattainable.
        strict_problem = cp.Problem(
            alpha_objective,
            [*constraints, *strict_gap_constraints],
        )
        strict_solver, strict_attempts, _ = self._solve_problem(
            strict_problem,
            stage="strict_alpha",
        )
        if strict_solver is not None and w.value is not None:
            raw_target = np.asarray(w.value, dtype=float).reshape(-1)
            result_status = str(strict_problem.status)
            result_action = "trade_to_strict_lexicographic_target"
            solved_with = strict_solver
            repair_solver: str | None = None
            repair_status = "not_required_strict_feasible"
            alpha_status = str(strict_problem.status)
            attempts = strict_attempts
            minimum_gap_score = 0.0
            repair_budget = self.settings.feasibility_tolerance
        else:
            repair_problem = cp.Problem(cp.Minimize(repair_score), constraints)
            repair_solver, repair_attempts, repair_value = self._solve_problem(
                repair_problem,
                stage="repair",
            )
            attempts = [*strict_attempts, *repair_attempts]
            base_diagnostics["attempts"] = attempts
            if repair_solver is None or repair_value is None or w.value is None:
                return OptimizationResult(
                    target=None,
                    diagnostics={
                        **base_diagnostics,
                        "status": repair_problem.status or "not_solved",
                        "action": "hold_pre_trade_portfolio",
                        "strict_alpha_status": strict_problem.status,
                        "repair_status": repair_problem.status,
                    },
                )
            repair_target = np.asarray(w.value, dtype=float).reshape(-1).copy()
            minimum_gap_score = max(0.0, float(repair_value))
            repair_lock_tolerance = max(
                self.settings.feasibility_tolerance,
                abs(minimum_gap_score) * self.settings.feasibility_tolerance,
            )
            repair_budget = minimum_gap_score + repair_lock_tolerance

            alpha_problem = cp.Problem(
                alpha_objective,
                [*constraints, repair_score <= repair_budget],
            )
            alpha_solver, alpha_attempts, _ = self._solve_problem(
                alpha_problem,
                stage="alpha_after_repair",
            )
            attempts.extend(alpha_attempts)
            repair_status = str(repair_problem.status)
            alpha_status = str(alpha_problem.status)
            if alpha_solver is not None and w.value is not None:
                raw_target = np.asarray(w.value, dtype=float).reshape(-1)
                result_status = str(alpha_problem.status)
                result_action = "trade_to_lexicographic_target"
                solved_with = alpha_solver
            else:
                # A valid repair portfolio is safer than retaining a larger breach
                # merely because the secondary alpha problem failed.
                raw_target = repair_target
                result_status = "repair_only"
                result_action = "trade_to_minimum_gap_repair"
                solved_with = repair_solver

        if raw_target.size != count or not np.isfinite(raw_target).all():
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "attempts": attempts,
                    "status": "invalid_solver_output",
                    "action": "hold_pre_trade_portfolio",
                    "strict_alpha_status": strict_problem.status,
                    "repair_status": repair_status,
                    "alpha_status": alpha_status,
                    "solver": solved_with,
                },
            )
        target_array = raw_target.copy()
        target_array[np.abs(target_array) < 1e-10] = 0.0
        target_array = np.maximum(target_array, 0.0)
        target_total = float(target_array.sum())
        if not np.isfinite(target_total) or target_total <= 0:
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "attempts": attempts,
                    "status": "invalid_solver_output",
                    "action": "hold_pre_trade_portfolio",
                    "strict_alpha_status": strict_problem.status,
                    "repair_status": repair_status,
                    "alpha_status": alpha_status,
                    "solver": solved_with,
                },
            )
        target_array /= target_total
        target = pd.Series(target_array, index=names, name="target_weight")
        active_array = target_array - b
        delta_array = target_array - pre
        realized_turnover = 0.5 * (np.abs(delta_array).sum() + pre_cash_weight)
        target_profile = _configured_gap_profile(
            weights=target_array,
            benchmark=b,
            settings=self.settings,
            exposures=matrix,
            beta_values=beta_values,
            covariance=covariance,
            annualization=self.risk.annualization,
        )
        expected_cost_value = float(
            buy_rate * np.maximum(delta_array, 0.0).sum()
            + sell_rate * np.maximum(-delta_array, 0.0).sum()
        )
        predicted_active_return = float(alpha @ active_array)
        risk_penalty_value = float(
            self.settings.risk_aversion * (active_array @ horizon_covariance @ active_array)
        )
        regularization_value = float(self.settings.l2_penalty * (active_array @ active_array))

        # The configured-gap score is dimensionless because every gap is
        # divided by its own cap.  The other post-solve violations are
        # weight-like proportions.  Convert score excess back to a conservative
        # constraint scale before comparing it with ``feasibility_tolerance``.
        repair_gap_score_excess = max(
            0.0,
            float(target_profile["score"]) - repair_budget,
        )
        repair_gap_violation_scale = _repair_gap_violation_scale(
            settings=self.settings,
            exposures_enabled=matrix is not None,
            beta_enabled=beta_values is not None,
        )
        repair_gap_equivalent_violation = (
            repair_gap_score_excess * repair_gap_violation_scale
        )

        gap_diagnostics: dict[str, Any] = {
            "minimum_configured_gap_score": minimum_gap_score,
            "repair_gap_budget": repair_budget,
            "configured_gap_score_after": target_profile["score"],
            "configured_gap_score_excess_from_minimum": max(
                0.0,
                float(target_profile["score"]) - minimum_gap_score,
            ),
            "configured_gap_score_improvement": float(
                pre_profile["score"] - target_profile["score"]
            ),
            "repair_gap_score_excess": repair_gap_score_excess,
            "repair_gap_violation_scale": repair_gap_violation_scale,
            "repair_gap_equivalent_violation": repair_gap_equivalent_violation,
            "configured_name_cap_breaches_after": target_profile["name_breach_count"],
            "configured_active_cap_breaches_after": target_profile[
                "active_breach_count"
            ],
            "configured_exposure_cap_breaches_after": target_profile[
                "exposure_breach_count"
            ],
            "target_active_beta": target_profile["active_beta"],
            "target_portfolio_beta": (
                float(beta_values @ target_array) if beta_values is not None else None
            ),
            "target_benchmark_beta": (
                float(beta_values @ b) if beta_values is not None else None
            ),
            "configured_beta_cap_breach_after": target_profile["beta_breach"],
            "configured_tracking_error_cap_breach_after": target_profile[
                "tracking_error_breach"
            ],
            "maximum_configured_name_cap_breach_after": target_profile[
                "maximum_name_breach"
            ],
            "maximum_configured_active_cap_breach_after": target_profile[
                "maximum_active_breach"
            ],
            "maximum_configured_exposure_cap_breach_after": target_profile[
                "maximum_exposure_breach"
            ],
            "maximum_configured_constraint_breach_after": target_profile[
                "maximum_breach"
            ],
            "maximum_absolute_active_weight": float(np.abs(active_array).max()),
            "strict_configured_constraints_satisfied": bool(
                target_profile["maximum_breach"] <= self.settings.feasibility_tolerance
            ),
        }

        hard_violations = {
            "budget": abs(float(target_array.sum()) - 1.0),
            "long_only": max(0.0, -float(target_array.min())),
            "turnover": max(0.0, realized_turnover - turnover_limit),
            "ineligible_buy": (
                max(
                    0.0,
                    float(
                        (target_array[ineligible_to_buy] - inactive_upper[ineligible_to_buy]).max()
                    ),
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
            "trade_cap": (
                max(
                    0.0,
                    float((np.abs(delta_array) - trade_caps).max()),
                )
                if trade_caps is not None
                else 0.0
            ),
            "repair_gap_lock": repair_gap_equivalent_violation,
        }
        maximum_violation = max(hard_violations.values())
        if maximum_violation > self.settings.feasibility_tolerance:
            return OptimizationResult(
                target=None,
                diagnostics={
                    **base_diagnostics,
                    "attempts": attempts,
                    "status": "postsolve_infeasible",
                    "action": "hold_pre_trade_portfolio",
                    "strict_alpha_status": strict_problem.status,
                    "repair_status": repair_status,
                    "alpha_status": alpha_status,
                    "solver": solved_with,
                    "turnover": float(realized_turnover),
                    **gap_diagnostics,
                    "maximum_violation": maximum_violation,
                    "violations": hard_violations,
                },
            )

        return OptimizationResult(
            target=target,
            diagnostics={
                **base_diagnostics,
                "attempts": attempts,
                "status": result_status,
                "action": result_action,
                "strict_alpha_status": strict_problem.status,
                "repair_status": repair_status,
                "alpha_status": alpha_status,
                "repair_solver": repair_solver,
                "solver": solved_with,
                "objective": (
                    predicted_active_return
                    - risk_penalty_value
                    - expected_cost_value
                    - regularization_value
                ),
                "predicted_active_return": predicted_active_return,
                "ex_ante_tracking_error": target_profile["tracking_error"],
                "tracking_error_utilization": target_profile["tracking_error_utilization"],
                "expected_cost": expected_cost_value,
                "turnover": float(realized_turnover),
                **gap_diagnostics,
                "maximum_violation": maximum_violation,
                "violations": hard_violations,
            },
        )

    def _solve_problem(
        self,
        problem: cp.Problem,
        *,
        stage: str,
    ) -> tuple[str | None, list[dict[str, Any]], float | None]:
        attempts: list[dict[str, Any]] = []
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
                value = problem.solve(
                    solver=solver,
                    warm_start=True,
                    verbose=False,
                    **solver_options,
                )
                attempts.append(
                    {
                        "stage": stage,
                        "solver": solver,
                        "status": problem.status,
                        "objective": float(value) if value is not None else None,
                    }
                )
            except cp.error.SolverError as exc:
                attempts.append(
                    {
                        "stage": stage,
                        "solver": solver,
                        "status": "solver_error",
                        "error": str(exc),
                    }
                )
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return (
                    solver,
                    attempts,
                    float(value) if value is not None else None,
                )
        return None, attempts, None

    @staticmethod
    def _invalid_result(
        decision_date: str,
        execution_date: str,
        error: str,
    ) -> OptimizationResult:
        return OptimizationResult(
            target=None,
            diagnostics={
                "decision_date": decision_date,
                "execution_date": execution_date,
                "status": "invalid_input",
                "action": "hold_pre_trade_portfolio",
                "error": error,
            },
        )

    def _stamp_rate(self, execution_date: str) -> float:
        if execution_date < self.research.stamp_duty_change_date:
            return self.research.stamp_duty_before
        return self.research.stamp_duty_after


def _configured_gap_profile(
    *,
    weights: np.ndarray,
    benchmark: np.ndarray,
    settings: OptimizerSettings,
    exposures: np.ndarray | None,
    beta_values: np.ndarray | None,
    covariance: np.ndarray,
    annualization: int,
) -> _GapProfile:
    active = weights - benchmark
    name_gaps = np.maximum(weights - settings.name_cap, 0.0)
    active_gaps = np.maximum(np.abs(active) - settings.active_cap, 0.0)
    exposure_gaps = (
        np.maximum(np.abs(exposures.T @ active) - settings.exposure_cap, 0.0)
        if exposures is not None
        else np.array([], dtype=float)
    )
    active_beta = float(beta_values @ active) if beta_values is not None else None
    beta_gap = (
        max(0.0, abs(active_beta) - settings.beta_active_cap) if active_beta is not None else 0.0
    )
    tracking_variance = max(
        float(active @ covariance @ active) * annualization,
        0.0,
    )
    tracking_error = float(np.sqrt(tracking_variance))
    tracking_variance_cap = settings.tracking_error_cap**2
    tracking_variance_gap = max(
        0.0,
        tracking_variance - tracking_variance_cap,
    )
    tracking_error_gap = max(
        0.0,
        tracking_error - settings.tracking_error_cap,
    )
    score = (
        float(name_gaps.sum()) / settings.name_cap
        + float(active_gaps.sum()) / settings.active_cap
        + (float(exposure_gaps.sum()) / settings.exposure_cap if exposure_gaps.size else 0.0)
        + beta_gap / settings.beta_active_cap
        + tracking_variance_gap / tracking_variance_cap
    )
    maximum_name_gap = float(name_gaps.max()) if name_gaps.size else 0.0
    maximum_active_gap = float(active_gaps.max()) if active_gaps.size else 0.0
    maximum_exposure_gap = float(exposure_gaps.max()) if exposure_gaps.size else 0.0
    return {
        "score": score,
        "name_breach_count": int((name_gaps > settings.feasibility_tolerance).sum()),
        "active_breach_count": int((active_gaps > settings.feasibility_tolerance).sum()),
        "exposure_breach_count": int((exposure_gaps > settings.feasibility_tolerance).sum()),
        "maximum_name_breach": maximum_name_gap,
        "maximum_active_breach": maximum_active_gap,
        "maximum_exposure_breach": maximum_exposure_gap,
        "active_beta": active_beta,
        "beta_breach": beta_gap,
        "tracking_error": tracking_error,
        "tracking_error_breach": tracking_error_gap,
        "tracking_error_utilization": (tracking_error / settings.tracking_error_cap),
        "maximum_breach": max(
            maximum_name_gap,
            maximum_active_gap,
            maximum_exposure_gap,
            beta_gap,
            tracking_error_gap,
        ),
    }


def _repair_gap_violation_scale(
    *,
    settings: OptimizerSettings,
    exposures_enabled: bool,
    beta_enabled: bool,
) -> float:
    """Return a conservative raw-constraint scale for repair-score excess."""

    scales = [
        settings.name_cap,
        settings.active_cap,
        settings.tracking_error_cap**2,
    ]
    if exposures_enabled:
        scales.append(settings.exposure_cap)
    if beta_enabled:
        scales.append(settings.beta_active_cap)
    return max(scales)
