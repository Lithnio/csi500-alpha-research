from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.execution.backtest import calculate_backtest_metrics, fit_capm
from csi500_alpha.portfolio.audit import summarize_constraint_audits


@dataclass(frozen=True)
class ResearchEvaluation:
    """Shareable evaluation summaries and their auditable tabular evidence."""

    summary: dict[str, Any]
    calibration_bins: pd.DataFrame
    yearly_metrics: pd.DataFrame
    factor_weight_history: pd.DataFrame


@dataclass(frozen=True)
class PortfolioEvaluation:
    """Portfolio-only diagnostics reusable by baseline and stress backtests."""

    summary: dict[str, Any]
    yearly_metrics: pd.DataFrame


def evaluate_portfolio_run(
    *,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    constraint_audits: pd.DataFrame | None = None,
) -> PortfolioEvaluation:
    yearly, yearly_metrics = _yearly_evaluation(daily, trades)
    metrics = calculate_backtest_metrics(daily, trades)
    performance_keys = (
        "relative_active_total_return",
        "annualized_active_return",
        "annualized_active_mean",
        "tracking_error",
        "information_ratio",
        "portfolio_max_drawdown",
        "active_max_drawdown",
        "capm_alpha_annualized",
        "capm_beta",
        "capm_beta_drag_annualized",
        "capm_reconciliation_error",
        "rolling_beta_abs_deviation_p95",
    )
    return PortfolioEvaluation(
        summary={
            "performance": {key: metrics[key] for key in performance_keys},
            "yearly": yearly,
            "execution": _execution_evaluation(daily, trades),
            "constraints": summarize_constraint_audits(
                constraint_audits
                if constraint_audits is not None
                else pd.DataFrame()
            ),
        },
        yearly_metrics=yearly_metrics,
    )


def evaluate_research_run(
    *,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    constraint_audits: pd.DataFrame | None = None,
    optimization: pd.DataFrame | None = None,
    model_fits: pd.DataFrame,
    calibrator_name: str,
    calibrator_params: Mapping[str, Any],
) -> ResearchEvaluation:
    """Evaluate one completed research run without changing its fitted decisions."""

    calibration, calibration_bins = _calibration_evaluation(
        signals,
        labels,
        expected_return_bound=_expected_return_bound(
            calibrator_name,
            calibrator_params,
        ),
    )
    portfolio = evaluate_portfolio_run(
        daily=daily,
        trades=trades,
        constraint_audits=constraint_audits,
    )
    model_weights, factor_weight_history = _model_weight_evaluation(model_fits)
    return ResearchEvaluation(
        summary={
            "calibration": calibration,
            "performance": portfolio.summary["performance"],
            "yearly": portfolio.summary["yearly"],
            "model_weights": model_weights,
            "execution": portfolio.summary["execution"],
            "constraints": portfolio.summary["constraints"],
            "risk": _risk_evaluation(
                optimization if optimization is not None else pd.DataFrame()
            ),
        },
        calibration_bins=calibration_bins,
        yearly_metrics=portfolio.yearly_metrics,
        factor_weight_history=factor_weight_history,
    )


def _risk_evaluation(optimization: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether the configured risk and beta estimators actually ran."""

    empty = {
        "optimization_attempts": 0,
        "risk_methods": {},
        "beta_methods": {},
        "factor_model_attempts": 0,
        "factor_model_fallback_attempts": 0,
        "factor_model_fallback_fraction": None,
        "median_factor_count": None,
        "minimum_beta_observed_fraction": None,
        "maximum_beta_clip_fraction": None,
        "median_factor_covariance_condition_number": None,
    }
    if optimization.empty:
        return empty
    if "risk_method" not in optimization:
        return {**empty, "optimization_attempts": int(len(optimization))}

    methods = optimization["risk_method"].astype("string").fillna("__MISSING__")
    beta_methods = (
        optimization["beta_method"].astype("string").fillna("__MISSING__")
        if "beta_method" in optimization
        else pd.Series("__MISSING__", index=optimization.index, dtype="string")
    )
    factor_attempt = methods.str.startswith("factor_ewma", na=False)
    fallback = methods.str.startswith("factor_ewma_fallback", na=False)
    if "risk_factor_model_fallback" in optimization:
        explicit_fallback = optimization["risk_factor_model_fallback"].fillna(False)
        fallback |= explicit_fallback.astype(bool)
    factor_count = _finite_numeric(
        optimization.get("risk_factor_count", pd.Series(dtype=float))
    ).dropna()
    beta_coverage = _finite_numeric(
        optimization.get("risk_beta_observed_fraction", pd.Series(dtype=float))
    ).dropna()
    beta_clip = _finite_numeric(
        optimization.get("risk_beta_clip_fraction", pd.Series(dtype=float))
    ).dropna()
    condition = _finite_numeric(
        optimization.get(
            "risk_factor_covariance_condition_number",
            pd.Series(dtype=float),
        )
    ).dropna()
    factor_attempt_count = int(factor_attempt.sum())
    fallback_count = int((fallback & factor_attempt).sum())
    return {
        "optimization_attempts": int(len(optimization)),
        "risk_methods": {
            str(key): int(value)
            for key, value in methods.value_counts(dropna=False).sort_index().items()
        },
        "beta_methods": {
            str(key): int(value)
            for key, value in beta_methods.value_counts(dropna=False).sort_index().items()
        },
        "factor_model_attempts": factor_attempt_count,
        "factor_model_fallback_attempts": fallback_count,
        "factor_model_fallback_fraction": (
            float(fallback_count / factor_attempt_count)
            if factor_attempt_count
            else None
        ),
        "median_factor_count": _finite_or_none(factor_count.median()),
        "minimum_beta_observed_fraction": _finite_or_none(beta_coverage.min()),
        "maximum_beta_clip_fraction": _finite_or_none(beta_clip.max()),
        "median_factor_covariance_condition_number": _finite_or_none(
            condition.median()
        ),
    }


def _calibration_evaluation(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    expected_return_bound: float | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    signal_required = {"decision_date", "instrument", "expected_return"}
    label_required = {"decision_date", "instrument", "forward_active_return"}
    _require_columns(signals, signal_required, "evaluation signals")
    _require_columns(labels, label_required, "evaluation labels")
    _require_unique(signals, "evaluation signals")
    _require_unique(labels, "evaluation labels")
    merged = signals[["decision_date", "instrument", "expected_return"]].merge(
        labels[["decision_date", "instrument", "forward_active_return"]],
        on=["decision_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    merged["expected_return"] = _finite_numeric(merged["expected_return"])
    merged["forward_active_return"] = _finite_numeric(
        merged["forward_active_return"]
    )
    predicted = merged["expected_return"].notna()
    realized = predicted & merged["forward_active_return"].notna()
    sample = merged.loc[realized].copy()
    empty_bins = pd.DataFrame(
        columns=[
            "bin",
            "observations",
            "dates",
            "mean_expected_return",
            "mean_realized_active_return",
            "calibration_error",
        ]
    )
    if sample.empty:
        return (
            {
                "predicted_observations": int(predicted.sum()),
                "realized_observations": 0,
                "realized_dates": 0,
                "realized_coverage": 0.0,
                "intercept": None,
                "slope": None,
                "weighted_r_squared": None,
                "mean_daily_rank_ic": None,
                "rank_ic_dates": 0,
                "mean_absolute_error": None,
                "root_mean_squared_error": None,
                "expected_to_realized_volatility_ratio": None,
                "directional_hit_rate": None,
                "boundary_fraction": None,
                "quintile_monotonicity": None,
                "top_minus_bottom_realized_return": None,
            },
            empty_bins,
        )

    weights = _equal_date_weights(sample, "decision_date")
    expected = sample["expected_return"].to_numpy(dtype=float)
    outcome = sample["forward_active_return"].to_numpy(dtype=float)
    intercept, slope, weighted_r_squared = _weighted_linear_fit(
        expected,
        outcome,
        weights,
    )
    errors = outcome - expected
    mean_absolute_error = float(np.sum(weights * np.abs(errors)))
    root_mean_squared_error = float(np.sqrt(np.sum(weights * np.square(errors))))
    expected_mean = float(np.sum(weights * expected))
    outcome_mean = float(np.sum(weights * outcome))
    expected_variance = float(np.sum(weights * np.square(expected - expected_mean)))
    outcome_variance = float(np.sum(weights * np.square(outcome - outcome_mean)))
    volatility_ratio = (
        np.sqrt(expected_variance / outcome_variance)
        if outcome_variance > 0
        else np.nan
    )
    nonzero = np.abs(expected) > 1e-14
    directional_hit_rate = (
        float(np.sum(weights[nonzero] * (np.sign(expected[nonzero]) == np.sign(outcome[nonzero])))
        / np.sum(weights[nonzero]))
        if nonzero.any()
        else np.nan
    )
    rank_ics = _daily_rank_ics(sample)
    calibration_bins = _calibration_bins(sample)
    bin_outcomes = calibration_bins.set_index("bin")[
        "mean_realized_active_return"
    ]
    monotonicity = (
        float(
            pd.Series(bin_outcomes.index, dtype=float).corr(
                pd.Series(bin_outcomes.to_numpy(dtype=float)),
                method="spearman",
            )
        )
        if len(bin_outcomes) >= 3
        else np.nan
    )
    spread = (
        float(bin_outcomes.loc[5] - bin_outcomes.loc[1])
        if {1, 5}.issubset(bin_outcomes.index)
        else np.nan
    )
    boundary_fraction = np.nan
    if expected_return_bound is not None:
        tolerance = max(expected_return_bound * 1e-8, 1e-12)
        boundary_fraction = float(
            np.sum(
                weights
                * np.isclose(
                    np.abs(expected),
                    expected_return_bound,
                    rtol=0.0,
                    atol=tolerance,
                )
            )
        )
    return (
        {
            "predicted_observations": int(predicted.sum()),
            "realized_observations": int(len(sample)),
            "realized_dates": int(sample["decision_date"].nunique()),
            "realized_coverage": float(len(sample) / predicted.sum()),
            "intercept": _finite_or_none(intercept),
            "slope": _finite_or_none(slope),
            "weighted_r_squared": _finite_or_none(weighted_r_squared),
            "mean_daily_rank_ic": _finite_or_none(rank_ics.mean()),
            "rank_ic_dates": int(len(rank_ics)),
            "mean_absolute_error": mean_absolute_error,
            "root_mean_squared_error": root_mean_squared_error,
            "expected_return_volatility": float(np.sqrt(expected_variance)),
            "realized_return_volatility": float(np.sqrt(outcome_variance)),
            "expected_to_realized_volatility_ratio": _finite_or_none(
                volatility_ratio
            ),
            "directional_hit_rate": _finite_or_none(directional_hit_rate),
            "expected_return_bound": expected_return_bound,
            "boundary_fraction": _finite_or_none(boundary_fraction),
            "quintile_monotonicity": _finite_or_none(monotonicity),
            "top_minus_bottom_realized_return": _finite_or_none(spread),
        },
        calibration_bins,
    )


def _yearly_evaluation(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "trade_date",
        "portfolio_return",
        "benchmark_return",
        "active_return",
        "turnover",
    }
    _require_columns(daily, required, "backtest daily")
    rows: list[dict[str, Any]] = []
    cost_by_year: dict[str, float] = {}
    if not trades.empty:
        _require_columns(
            trades,
            {"trade_date", "linear_cost", "stamp_duty", "impact_cost"},
            "backtest trades",
        )
        costs = trades.copy()
        costs["year"] = costs["trade_date"].astype(str).str[:4]
        costs["total_cost"] = sum(
            (_finite_numeric(costs[column]) for column in _cost_columns()),
            start=pd.Series(0.0, index=costs.index),
        )
        cost_by_year = costs.groupby("year")["total_cost"].sum().to_dict()

    prepared = daily.copy()
    prepared["year"] = prepared["trade_date"].astype(str).str[:4]
    for year, frame in prepared.groupby("year", sort=True):
        portfolio = _finite_numeric(frame["portfolio_return"]).fillna(0.0)
        benchmark = _finite_numeric(frame["benchmark_return"]).fillna(0.0)
        active = _finite_numeric(frame["active_return"]).fillna(0.0)
        portfolio_growth = float((1.0 + portfolio).prod())
        benchmark_growth = float((1.0 + benchmark).prod())
        active_total = (
            portfolio_growth / benchmark_growth - 1.0
            if benchmark_growth > 0
            else np.nan
        )
        tracking_error = (
            float(active.std(ddof=1) * np.sqrt(252.0)) if len(active) > 1 else np.nan
        )
        active_growth = float((1.0 + active).prod())
        annualized_active_mean = (
            float(active.mean() * 252.0) if len(active) else np.nan
        )
        annualized_active = (
            active_growth ** (252.0 / len(active)) - 1.0
            if len(active) and active_growth > 0
            else np.nan
        )
        information_ratio = (
            annualized_active_mean / tracking_error
            if np.isfinite(tracking_error) and tracking_error > 0
            else np.nan
        )
        wealth = (1.0 + portfolio).cumprod()
        portfolio_drawdown = wealth / wealth.cummax() - 1.0
        active_wealth = (1.0 + active).cumprod()
        active_drawdown = active_wealth / active_wealth.cummax() - 1.0
        capm_alpha_daily, capm_beta = fit_capm(portfolio, benchmark)
        capm_alpha_annualized = capm_alpha_daily * 252.0
        capm_beta_drag = (
            (capm_beta - 1.0) * float(benchmark.mean()) * 252.0
            if np.isfinite(capm_beta)
            else np.nan
        )
        observations = len(frame)
        active_max_drawdown = (
            float(active_drawdown.min()) if not active_drawdown.empty else np.nan
        )
        rows.append(
            {
                "year": str(year),
                "start_date": str(frame["trade_date"].astype(str).min()),
                "end_date": str(frame["trade_date"].astype(str).max()),
                "observations": observations,
                "total_return": portfolio_growth - 1.0,
                "benchmark_total_return": benchmark_growth - 1.0,
                "active_total_return": active_total,
                "annualized_return": (
                    portfolio_growth ** (252.0 / observations) - 1.0
                    if observations > 0 and portfolio_growth > 0
                    else np.nan
                ),
                "annualized_active_return": annualized_active,
                "annualized_active_mean": annualized_active_mean,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "portfolio_max_drawdown": (
                    float(portfolio_drawdown.min())
                    if not portfolio_drawdown.empty
                    else np.nan
                ),
                "active_max_drawdown": active_max_drawdown,
                "max_drawdown": active_max_drawdown,
                "capm_alpha_annualized": capm_alpha_annualized,
                "capm_beta": capm_beta,
                "capm_beta_drag_annualized": capm_beta_drag,
                "capm_reconciliation_error": (
                    annualized_active_mean
                    - capm_alpha_annualized
                    - capm_beta_drag
                    if np.isfinite(annualized_active_mean)
                    and np.isfinite(capm_alpha_annualized)
                    and np.isfinite(capm_beta_drag)
                    else np.nan
                ),
                "average_turnover": float(
                    _finite_numeric(frame["turnover"]).fillna(0.0).mean()
                ),
                "transaction_cost": float(cost_by_year.get(str(year), 0.0)),
            }
        )
    columns = [
        "year",
        "start_date",
        "end_date",
        "observations",
        "total_return",
        "benchmark_total_return",
        "active_total_return",
        "annualized_return",
        "annualized_active_return",
        "annualized_active_mean",
        "tracking_error",
        "information_ratio",
        "portfolio_max_drawdown",
        "active_max_drawdown",
        "max_drawdown",
        "capm_alpha_annualized",
        "capm_beta",
        "capm_beta_drag_annualized",
        "capm_reconciliation_error",
        "average_turnover",
        "transaction_cost",
    ]
    yearly = pd.DataFrame(rows, columns=columns)
    finite_ir = _finite_numeric(yearly.get("information_ratio", pd.Series(dtype=float))).dropna()
    finite_active = _finite_numeric(
        yearly.get("active_total_return", pd.Series(dtype=float))
    ).dropna()
    return (
        {
            "year_count": int(len(yearly)),
            "positive_active_years": int((finite_active > 0).sum()),
            "positive_active_year_fraction": (
                float((finite_active > 0).mean()) if len(finite_active) else None
            ),
            "minimum_active_total_return": _finite_or_none(finite_active.min()),
            "median_active_total_return": _finite_or_none(finite_active.median()),
            "minimum_information_ratio": _finite_or_none(finite_ir.min()),
            "median_information_ratio": _finite_or_none(finite_ir.median()),
            "information_ratio_dispersion": _finite_or_none(
                finite_ir.std(ddof=0)
            ),
            "worst_max_drawdown": _finite_or_none(
                _finite_numeric(
                    yearly.get("active_max_drawdown", pd.Series(dtype=float))
                ).min()
            ),
            "worst_active_max_drawdown": _finite_or_none(
                _finite_numeric(
                    yearly.get("active_max_drawdown", pd.Series(dtype=float))
                ).min()
            ),
            "worst_portfolio_max_drawdown": _finite_or_none(
                _finite_numeric(
                    yearly.get("portfolio_max_drawdown", pd.Series(dtype=float))
                ).min()
            ),
            "minimum_year_observations": (
                int(yearly["observations"].min()) if not yearly.empty else 0
            ),
        },
        yearly,
    )


def _model_weight_evaluation(
    model_fits: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = [
        "fit_date",
        "model",
        "factor",
        "family",
        "weight_source",
        "raw_parameter",
        "allocation_weight",
        "previous_weight",
        "eligible",
        "mean_directed_ic",
        "hac_standard_error",
        "shrinkage_coefficient",
        "posterior_directed_ic",
        "score_churn",
        "exclusion_reasons",
    ]
    if model_fits.empty:
        return _empty_weight_summary(), pd.DataFrame(columns=columns)
    _require_columns(
        model_fits,
        {"fit_date", "model", "status", "model_parameters"},
        "model fits",
    )
    rows: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, float]] = []
    previous_allocation: dict[str, float] | None = None
    p3_fit_count = 0
    for _, fit in model_fits.iterrows():
        if str(fit["status"]) != "fitted":
            continue
        parameters = _parameter_mapping(fit["model_parameters"])
        source_name, raw_weights = _weight_mapping(parameters)
        if not raw_weights:
            continue
        magnitudes = {name: abs(value) for name, value in raw_weights.items()}
        total = float(sum(magnitudes.values()))
        if not np.isfinite(total) or total <= 0:
            continue
        allocation = {name: value / total for name, value in magnitudes.items()}
        concentration = float(sum(value**2 for value in allocation.values()))
        effective_count = 1.0 / concentration
        union = set(allocation) | set(previous_allocation or {})
        computed_change = (
            float(
                sum(
                    abs(
                        allocation.get(name, 0.0)
                        - (previous_allocation or {}).get(name, 0.0)
                    )
                    for name in union
                )
            )
            if previous_allocation is not None
            else 0.0
        )
        reported_change = _finite_number(
            parameters.get("realized_factor_weight_l1_change")
        )
        factor_change = reported_change if reported_change is not None else computed_change
        settings = parameters.get("settings")
        max_factor_weight = (
            _finite_number(settings.get("max_factor_weight"))
            if isinstance(settings, Mapping)
            else None
        )
        cap_hits = (
            sum(
                value >= max_factor_weight - 1e-6
                for value in allocation.values()
            )
            if max_factor_weight is not None
            else 0
        )
        raw_family_weights = parameters.get("family_weights")
        family_weights = (
            {
                str(name): value
                for name, raw_value in raw_family_weights.items()
                if (value := _finite_number(raw_value)) is not None
            }
            if isinstance(raw_family_weights, Mapping)
            else {}
        )
        max_family_weight = (
            _finite_number(settings.get("max_family_weight"))
            if isinstance(settings, Mapping)
            else None
        )
        family_cap_hits = (
            sum(
                value >= max_family_weight - 1e-6
                for value in family_weights.values()
            )
            if max_family_weight is not None
            else 0
        )
        maximum_family_weight = (
            max(family_weights.values()) if family_weights else np.nan
        )
        is_p3 = parameters.get("method") in {
            "empirical_bayes_ic_shrinkage_convex_synthesis",
            "core_anchored_empirical_bayes_ic_shrinkage",
            "oof_net_residual_sleeve_blend",
            "turnover_budgeted_residual_sleeve_blend",
        }
        p3_fit_count += int(is_p3)
        fit_diagnostics.append(
            {
                "effective_factor_count": effective_count,
                "concentration": concentration,
                "maximum_weight": max(allocation.values()),
                "factor_weight_l1_change": factor_change,
                "cap_hits": float(cap_hits),
                "maximum_family_weight": maximum_family_weight,
                "family_cap_hits": float(family_cap_hits),
            }
        )
        factor_statistics = parameters.get("factor_statistics")
        statistics = factor_statistics if isinstance(factor_statistics, Mapping) else {}
        family_by_factor = _family_by_factor(parameters.get("family_factors"))
        for factor, raw_value in raw_weights.items():
            raw_statistics = statistics.get(factor, {})
            factor_stats = raw_statistics if isinstance(raw_statistics, Mapping) else {}
            reasons = factor_stats.get("exclusion_reasons", [])
            rows.append(
                {
                    "fit_date": str(fit["fit_date"]),
                    "model": str(fit["model"]),
                    "factor": factor,
                    "family": family_by_factor.get(factor),
                    "weight_source": source_name,
                    "raw_parameter": raw_value,
                    "allocation_weight": allocation[factor],
                    "previous_weight": _finite_number(
                        factor_stats.get("previous_weight")
                    ),
                    "eligible": factor_stats.get("eligible"),
                    "mean_directed_ic": _finite_number(
                        factor_stats.get("mean_directed_ic")
                    ),
                    "hac_standard_error": _finite_number(
                        factor_stats.get("hac_standard_error")
                    ),
                    "shrinkage_coefficient": _finite_number(
                        factor_stats.get("shrinkage_coefficient")
                    ),
                    "posterior_directed_ic": _finite_number(
                        factor_stats.get("posterior_directed_ic")
                    ),
                    "score_churn": _finite_number(factor_stats.get("score_churn")),
                    "exclusion_reasons": json.dumps(reasons, ensure_ascii=False),
                }
            )
        previous_allocation = allocation
    history = pd.DataFrame(rows, columns=columns)
    if not fit_diagnostics:
        return _empty_weight_summary(), history
    diagnostics = pd.DataFrame(fit_diagnostics)
    return (
        {
            "audited_fit_count": int(len(diagnostics)),
            "p3_fit_count": p3_fit_count,
            "mean_effective_factor_count": float(
                diagnostics["effective_factor_count"].mean()
            ),
            "minimum_effective_factor_count": float(
                diagnostics["effective_factor_count"].min()
            ),
            "mean_weight_concentration": float(diagnostics["concentration"].mean()),
            "maximum_single_factor_weight": float(
                diagnostics["maximum_weight"].max()
            ),
            "maximum_family_weight": _finite_or_none(
                diagnostics["maximum_family_weight"].max()
            ),
            "mean_factor_weight_l1_change": float(
                diagnostics["factor_weight_l1_change"].mean()
            ),
            "maximum_factor_weight_l1_change": float(
                diagnostics["factor_weight_l1_change"].max()
            ),
            "factor_cap_hits": int(diagnostics["cap_hits"].sum()),
            "family_cap_hits": int(diagnostics["family_cap_hits"].sum()),
        },
        history,
    )


def _execution_evaluation(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    _require_columns(daily, {"turnover", "transaction_cost"}, "backtest daily")
    if trades.empty:
        return {
            "orders": 0,
            "executed_orders": 0,
            "partial_orders": 0,
            "blocked_orders": 0,
            "requested_value": 0.0,
            "executed_value": 0.0,
            "notional_fill_ratio": None,
            "transaction_cost": float(
                _finite_numeric(daily["transaction_cost"]).fillna(0.0).sum()
            ),
            "cost_bps_of_executed_notional": None,
            "average_turnover": float(
                _finite_numeric(daily["turnover"]).fillna(0.0).mean()
            ),
        }
    required = {
        "status",
        "requested_value",
        "gross_value",
        "linear_cost",
        "stamp_duty",
        "impact_cost",
    }
    _require_columns(trades, required, "backtest trades")
    requested = float(_finite_numeric(trades["requested_value"]).fillna(0.0).sum())
    executed = float(_finite_numeric(trades["gross_value"]).fillna(0.0).sum())
    cost = float(
        sum(
            _finite_numeric(trades[column]).fillna(0.0).sum()
            for column in _cost_columns()
        )
    )
    return {
        "orders": int(len(trades)),
        "executed_orders": int(trades["status"].isin(["filled", "partial"]).sum()),
        "partial_orders": int((trades["status"] == "partial").sum()),
        "blocked_orders": int((trades["status"] == "blocked").sum()),
        "requested_value": requested,
        "executed_value": executed,
        "notional_fill_ratio": executed / requested if requested > 0 else None,
        "transaction_cost": cost,
        "cost_bps_of_executed_notional": cost / executed * 10000.0
        if executed > 0
        else None,
        "average_turnover": float(
            _finite_numeric(daily["turnover"]).fillna(0.0).mean()
        ),
    }


def _calibration_bins(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision_date, frame in sample.groupby("decision_date", sort=True):
        if len(frame) < 5 or frame["expected_return"].nunique() < 2:
            continue
        percentile = frame["expected_return"].rank(method="first", pct=True)
        bins = np.ceil(percentile * 5.0).clip(1, 5).astype(int)
        grouped = frame.assign(bin=bins).groupby("bin")
        for bin_number, values in grouped:
            rows.append(
                {
                    "decision_date": str(decision_date),
                    "bin": int(bin_number),
                    "observations": int(len(values)),
                    "mean_expected_return": float(values["expected_return"].mean()),
                    "mean_realized_active_return": float(
                        values["forward_active_return"].mean()
                    ),
                }
            )
    daily_bins = pd.DataFrame(rows)
    columns = [
        "bin",
        "observations",
        "dates",
        "mean_expected_return",
        "mean_realized_active_return",
        "calibration_error",
    ]
    if daily_bins.empty:
        return pd.DataFrame(columns=columns)
    result = (
        daily_bins.groupby("bin", as_index=False)
        .agg(
            observations=("observations", "sum"),
            dates=("decision_date", "nunique"),
            mean_expected_return=("mean_expected_return", "mean"),
            mean_realized_active_return=("mean_realized_active_return", "mean"),
        )
        .sort_values("bin")
    )
    result["calibration_error"] = (
        result["mean_realized_active_return"] - result["mean_expected_return"]
    )
    return result[columns]


def _daily_rank_ics(sample: pd.DataFrame) -> pd.Series:
    values: list[float] = []
    for _, frame in sample.groupby("decision_date", sort=True):
        if len(frame) < 3:
            continue
        if frame["expected_return"].nunique() < 2:
            continue
        rank_ic = float(
            frame["expected_return"].corr(
                frame["forward_active_return"],
                method="spearman",
            )
        )
        if np.isfinite(rank_ic):
            values.append(rank_ic)
    return pd.Series(values, dtype=float)


def _weighted_linear_fit(
    predictor: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(predictor), dtype=float), predictor])
    square_root_weight = np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        design * square_root_weight[:, None],
        outcome * square_root_weight,
        rcond=None,
    )
    fitted = design @ coefficients
    outcome_mean = float(np.sum(weights * outcome))
    residual_sum = float(np.sum(weights * np.square(outcome - fitted)))
    total_sum = float(np.sum(weights * np.square(outcome - outcome_mean)))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    return float(coefficients[0]), float(coefficients[1]), r_squared


def _equal_date_weights(frame: pd.DataFrame, date_column: str) -> np.ndarray:
    counts = frame.groupby(date_column)[date_column].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.sum()


def _expected_return_bound(
    calibrator_name: str,
    params: Mapping[str, Any],
) -> float | None:
    if calibrator_name == "robust_cross_section":
        scale = _finite_number(params.get("target_scale"))
        clip = _finite_number(params.get("score_clip"))
        if scale is not None and clip is not None and scale > 0 and clip > 0:
            return scale * clip
    if calibrator_name == "rolling_ridge":
        bound = _finite_number(params.get("max_abs_expected_return"))
        return bound if bound is not None and bound > 0 else None
    return None


def _weight_mapping(parameters: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    for name in ("factor_weights", "weights", "coefficients"):
        raw = parameters.get(name)
        if not isinstance(raw, Mapping):
            continue
        converted = {
            str(factor): value
            for factor, raw_value in raw.items()
            if (value := _finite_number(raw_value)) is not None
        }
        return name, converted
    return "", {}


def _family_by_factor(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(factor): str(family)
        for family, raw_factors in value.items()
        if isinstance(raw_factors, Sequence) and not isinstance(raw_factors, str)
        for factor in raw_factors
    }


def _parameter_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise ValueError("Model parameters must be a mapping or JSON object")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Model parameters JSON must contain an object")
    return parsed


def _empty_weight_summary() -> dict[str, Any]:
    return {
        "audited_fit_count": 0,
        "p3_fit_count": 0,
        "mean_effective_factor_count": None,
        "minimum_effective_factor_count": None,
        "mean_weight_concentration": None,
        "maximum_single_factor_weight": None,
        "maximum_family_weight": None,
        "mean_factor_weight_l1_change": None,
        "maximum_factor_weight_l1_change": None,
        "factor_cap_hits": 0,
        "family_cap_hits": 0,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if frame.duplicated(["decision_date", "instrument"]).any():
        raise ValueError(f"{name} decision_date/instrument key is not unique")


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _finite_or_none(value: Any) -> float | None:
    return _finite_number(value)


def _cost_columns() -> tuple[str, ...]:
    return "linear_cost", "stamp_duty", "impact_cost"
