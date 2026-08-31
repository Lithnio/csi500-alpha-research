from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import OptimizerSettings


@dataclass(frozen=True)
class ExecutedPortfolioAudit:
    """Post-execution positions and constraint diagnostics for one rebalance."""

    positions: pd.DataFrame
    summary: dict[str, Any]


def audit_executed_portfolio(
    *,
    signal_date: str,
    execution_date: str,
    settings: OptimizerSettings,
    risk_annualization: int,
    pre_weights: pd.Series,
    pre_cash_weight: float,
    actual_weights: pd.Series,
    actual_cash_weight: float,
    benchmark: pd.Series,
    target: pd.Series,
    covariance: pd.DataFrame,
    industry_exposures: pd.DataFrame | None,
    style_exposures: pd.DataFrame | None,
    trade_records: Sequence[Mapping[str, Any]],
    execution_day: pd.DataFrame,
    pre_nav: float,
    risk_method: str | None = None,
    beta_method: str | None = None,
) -> ExecutedPortfolioAudit:
    """Audit realized holdings against configured and repair-policy constraints."""

    names = pd.Index(
        sorted(
            set(pre_weights.index.astype(str))
            | set(actual_weights.index.astype(str))
            | set(benchmark.index.astype(str))
            | set(target.index.astype(str))
        ),
        name="instrument",
    )
    if names.empty:
        raise ValueError("Post-execution audit requires a nonempty portfolio universe")
    pre = _aligned_weights(pre_weights, names, "pre-trade weights")
    actual = _aligned_weights(actual_weights, names, "actual weights")
    benchmark_weights = _aligned_weights(benchmark, names, "benchmark weights")
    target_weights = _aligned_weights(target, names, "target weights")
    benchmark_total = float(benchmark_weights.sum())
    target_total = float(target_weights.sum())
    if benchmark_total <= 0 or target_total <= 0:
        raise ValueError("Benchmark and target weights must have positive totals")
    benchmark_weights /= benchmark_total
    target_weights /= target_total
    for label, cash_weight, weights in (
        ("pre-trade", pre_cash_weight, pre),
        ("actual", actual_cash_weight, actual),
    ):
        if not np.isfinite(cash_weight) or cash_weight < -1e-10:
            raise ValueError(f"{label} cash weight must be finite and nonnegative")
        budget_error = abs(float(weights.sum()) + float(cash_weight) - 1.0)
        if budget_error > 1e-8:
            raise ValueError(f"{label} weights do not reconcile to one: {budget_error}")

    active = actual - benchmark_weights
    target_active = target_weights - benchmark_weights
    tolerance = settings.feasibility_tolerance
    materiality = settings.constraint_materiality_tolerance

    aligned_industry = _aligned_frame(industry_exposures, names)
    industry_actual, _industry_pre, industry_target = _aggregate_exposures(
        aligned_industry,
        actual,
        pre,
        target_weights,
        benchmark_weights,
    )
    aligned_styles = _aligned_frame(style_exposures, names)
    style_actual, _style_pre, style_target = _aggregate_exposures(
        aligned_styles,
        actual,
        pre,
        target_weights,
        benchmark_weights,
    )

    positions = pd.DataFrame(
        {
            "signal_date": str(signal_date),
            "execution_date": str(execution_date),
            "instrument": names,
            "pre_trade_weight": pre.to_numpy(dtype=float),
            "target_weight": target_weights.to_numpy(dtype=float),
            "actual_weight": actual.to_numpy(dtype=float),
            "benchmark_weight": benchmark_weights.to_numpy(dtype=float),
            "active_weight": active.to_numpy(dtype=float),
            "target_deviation": (actual - target_weights).to_numpy(dtype=float),
        }
    )
    if aligned_styles is not None:
        for column in aligned_styles:
            positions[column] = aligned_styles[column].to_numpy(dtype=float)
    positions["industry_bucket"] = _industry_bucket(aligned_industry, names)

    configured_name_breaches = actual - settings.name_cap
    configured_active_breaches = active.abs() - settings.active_cap
    target_configured_name_breaches = target_weights - settings.name_cap
    target_configured_active_breaches = target_active.abs() - settings.active_cap
    configured_industry_breaches = (
        industry_actual.abs() - settings.exposure_cap
        if industry_actual is not None
        else pd.Series(dtype=float)
    )
    target_configured_industry_breaches = (
        industry_target.abs() - settings.exposure_cap
        if industry_target is not None
        else pd.Series(dtype=float)
    )

    # The optimizer has already minimized target gaps. Execution is audited
    # against that target envelope, so post-trade policy violations measure
    # incremental deterioration rather than inherited pre-trade breaches.
    name_upper = np.maximum(settings.name_cap, target_weights)
    active_upper = np.maximum(settings.active_cap, target_active)
    active_lower = np.minimum(-settings.active_cap, target_active)
    name_policy_violation = float(np.maximum(actual - name_upper, 0.0).max())
    active_policy_violation = float(
        max(
            np.maximum(active - active_upper, 0.0).max(),
            np.maximum(active_lower - active, 0.0).max(),
        )
    )

    if industry_actual is not None and industry_target is not None:
        industry_upper = np.maximum(settings.exposure_cap, industry_target)
        industry_lower = np.minimum(-settings.exposure_cap, industry_target)
        industry_policy_violation = float(
            max(
                np.maximum(industry_actual - industry_upper, 0.0).max(),
                np.maximum(industry_lower - industry_actual, 0.0).max(),
            )
        )
        maximum_industry_active_exposure = float(industry_actual.abs().max())
    else:
        industry_policy_violation = 0.0
        maximum_industry_active_exposure = np.nan

    beta = _beta_diagnostics(
        settings=settings,
        aligned_styles=aligned_styles,
        actual=actual,
        target=target_weights,
        benchmark=benchmark_weights,
    )
    configured_turnover_limit = (
        settings.initial_turnover_cap if pre_cash_weight > 0.5 else settings.turnover_cap
    )
    turnover_limit = max(configured_turnover_limit, float(pre_cash_weight))
    actual_turnover = 0.5 * (
        float((actual - pre).abs().sum()) + abs(float(actual_cash_weight) - float(pre_cash_weight))
    )
    turnover_violation = max(0.0, actual_turnover - turnover_limit)
    if settings.liquidity_enabled:
        participation, missing_liquidity = _maximum_ex_ante_participation(
            trade_records=trade_records,
        )
        if missing_liquidity:
            raise ValueError(
                "Executed orders are missing frozen ex-ante ADV audit fields: "
                f"orders={missing_liquidity}"
            )
    else:
        participation, missing_liquidity = np.nan, 0
    realized_participation, missing_realized_volume = _maximum_realized_day_participation(
        settings=settings,
        trade_records=trade_records,
        execution_day=execution_day,
        pre_nav=pre_nav,
    )
    participation_violation = (
        max(0.0, participation - settings.max_adv_participation)
        if np.isfinite(participation)
        else 0.0
    )
    actual_tracking_error = _tracking_error(
        actual - benchmark_weights,
        covariance,
        risk_annualization,
    )
    target_tracking_error = _tracking_error(
        target_weights - benchmark_weights,
        covariance,
        risk_annualization,
    )
    actual_tracking_error_breach = (
        max(0.0, actual_tracking_error - settings.tracking_error_cap)
        if np.isfinite(actual_tracking_error)
        else 0.0
    )
    target_tracking_error_breach = (
        max(0.0, target_tracking_error - settings.tracking_error_cap)
        if np.isfinite(target_tracking_error)
        else 0.0
    )
    tracking_error_policy_violation = (
        max(
            0.0,
            actual_tracking_error - max(settings.tracking_error_cap, target_tracking_error),
        )
        if np.isfinite(actual_tracking_error) and np.isfinite(target_tracking_error)
        else 0.0
    )
    actual_active_risk_utilization = (
        actual_tracking_error / settings.tracking_error_cap
        if np.isfinite(actual_tracking_error)
        else np.nan
    )
    target_active_risk_utilization = (
        target_tracking_error / settings.tracking_error_cap
        if np.isfinite(target_tracking_error)
        else np.nan
    )

    budget_violation = abs(float(actual.sum()) + float(actual_cash_weight) - 1.0)
    long_only_violation = max(0.0, -float(actual.min()), -float(actual_cash_weight))
    policy_violations = {
        "budget": budget_violation,
        "long_only": long_only_violation,
        "name_cap": name_policy_violation,
        "active_cap": active_policy_violation,
        "industry_cap": industry_policy_violation,
        "beta_cap": float(beta["policy_violation"]),
        "tracking_error": tracking_error_policy_violation,
        "turnover": turnover_violation,
        "participation": participation_violation,
    }
    configured_breach_flags = {
        "name_cap": bool((configured_name_breaches > tolerance).any()),
        "active_cap": bool((configured_active_breaches > tolerance).any()),
        "industry_cap": bool((configured_industry_breaches > tolerance).any()),
        "beta_cap": bool(float(beta["configured_breach"]) > tolerance),
        "tracking_error": actual_tracking_error_breach > tolerance,
        "turnover": turnover_violation > tolerance,
        "participation": participation_violation > tolerance,
    }
    target_configured_breach_flags = {
        "name_cap": bool((target_configured_name_breaches > tolerance).any()),
        "active_cap": bool((target_configured_active_breaches > tolerance).any()),
        "industry_cap": bool((target_configured_industry_breaches > tolerance).any()),
        "beta_cap": bool(float(beta["target_configured_breach"]) > tolerance),
        "tracking_error": target_tracking_error_breach > tolerance,
    }
    material_configured_breach_flags = {
        "name_cap": bool((configured_name_breaches > materiality).any()),
        "active_cap": bool((configured_active_breaches > materiality).any()),
        "industry_cap": bool((configured_industry_breaches > materiality).any()),
        "beta_cap": bool(float(beta["configured_breach"]) > materiality),
        "tracking_error": actual_tracking_error_breach > materiality,
        "turnover": turnover_violation > materiality,
        "participation": participation_violation > materiality,
    }
    maximum_configured_breach = max(
        _positive_max(configured_name_breaches),
        _positive_max(configured_active_breaches),
        _positive_max(configured_industry_breaches),
        float(beta["configured_breach"]),
        actual_tracking_error_breach,
        turnover_violation,
        participation_violation,
    )
    maximum_target_configured_breach = max(
        _positive_max(target_configured_name_breaches),
        _positive_max(target_configured_active_breaches),
        _positive_max(target_configured_industry_breaches),
        float(beta["target_configured_breach"]),
        target_tracking_error_breach,
    )
    execution_deteriorations = {
        "name_cap": _gap_deterioration(
            configured_name_breaches,
            target_configured_name_breaches,
        ),
        "active_cap": _gap_deterioration(
            configured_active_breaches,
            target_configured_active_breaches,
        ),
        "industry_cap": _gap_deterioration(
            configured_industry_breaches,
            target_configured_industry_breaches,
        ),
        "beta_cap": max(
            0.0,
            float(beta["configured_breach"]) - float(beta["target_configured_breach"]),
        ),
        "tracking_error": max(
            0.0,
            actual_tracking_error_breach - target_tracking_error_breach,
        ),
    }
    maximum_execution_deterioration = max(execution_deteriorations.values())
    summary = {
        "signal_date": str(signal_date),
        "execution_date": str(execution_date),
        "risk_method": risk_method,
        "beta_method": beta_method,
        "universe_size": int(len(names)),
        "actual_holdings": int((actual > 1e-12).sum()),
        "actual_cash_weight": float(actual_cash_weight),
        "actual_turnover": actual_turnover,
        "turnover_limit": turnover_limit,
        "maximum_name_weight": float(actual.max()),
        "maximum_absolute_active_weight": float(active.abs().max()),
        "maximum_industry_active_exposure": maximum_industry_active_exposure,
        "actual_active_beta": beta["actual_active"],
        "actual_portfolio_beta": beta["actual_portfolio"],
        "benchmark_beta": beta["benchmark"],
        "target_active_beta": beta["target_active"],
        "beta_missing_actual_weight": beta["missing_actual_weight"],
        "beta_missing_benchmark_weight": beta["missing_benchmark_weight"],
        "beta_audit_complete": beta["complete"],
        "actual_tracking_error": actual_tracking_error,
        "target_tracking_error": target_tracking_error,
        "tracking_error_cap": settings.tracking_error_cap,
        "actual_active_risk_utilization": actual_active_risk_utilization,
        "target_active_risk_utilization": target_active_risk_utilization,
        "maximum_ex_ante_adv_participation": participation,
        "maximum_adv_participation": participation,
        "missing_liquidity_executed_orders": missing_liquidity,
        "maximum_realized_day_participation": realized_participation,
        "missing_realized_day_volume_executed_orders": missing_realized_volume,
        "maximum_target_weight_deviation": float((actual - target_weights).abs().max()),
        "configured_name_cap_breaches": int((configured_name_breaches > tolerance).sum()),
        "configured_active_cap_breaches": int((configured_active_breaches > tolerance).sum()),
        "configured_industry_cap_breaches": int((configured_industry_breaches > tolerance).sum()),
        "configured_beta_cap_breach": float(beta["configured_breach"]),
        "configured_tracking_error_cap_breach": actual_tracking_error_breach,
        "target_configured_name_cap_breaches": int(
            (target_configured_name_breaches > tolerance).sum()
        ),
        "target_configured_active_cap_breaches": int(
            (target_configured_active_breaches > tolerance).sum()
        ),
        "target_configured_industry_cap_breaches": int(
            (target_configured_industry_breaches > tolerance).sum()
        ),
        "target_configured_beta_cap_breach": float(beta["target_configured_breach"]),
        "target_configured_tracking_error_cap_breach": (target_tracking_error_breach),
        "target_configured_breached_constraint_count": int(
            sum(target_configured_breach_flags.values())
        ),
        "has_target_configured_breach": any(target_configured_breach_flags.values()),
        "maximum_target_configured_constraint_breach": (maximum_target_configured_breach),
        "configured_breached_constraint_count": int(sum(configured_breach_flags.values())),
        "has_configured_breach": any(configured_breach_flags.values()),
        "maximum_configured_constraint_breach": maximum_configured_breach,
        "has_policy_violation": any(value > tolerance for value in policy_violations.values()),
        "maximum_policy_violation": max(policy_violations.values()),
        "constraint_materiality_tolerance": materiality,
        "has_material_configured_breach": any(material_configured_breach_flags.values()),
        "has_material_policy_violation": any(
            value > materiality for value in policy_violations.values()
        ),
        "has_execution_constraint_deterioration": (maximum_execution_deterioration > tolerance),
        "has_material_execution_constraint_deterioration": (
            maximum_execution_deterioration > materiality
        ),
        "maximum_execution_constraint_deterioration": (maximum_execution_deterioration),
        "policy_violations_json": json.dumps(policy_violations, sort_keys=True),
        "execution_constraint_deteriorations_json": json.dumps(
            execution_deteriorations,
            sort_keys=True,
        ),
        "industry_active_exposures_json": _json_series(industry_actual),
        "target_industry_active_exposures_json": _json_series(industry_target),
        "style_active_exposures_json": _json_series(style_actual),
        "target_style_active_exposures_json": _json_series(style_target),
    }
    return ExecutedPortfolioAudit(positions=positions, summary=summary)


def summarize_constraint_audits(audits: pd.DataFrame) -> dict[str, Any]:
    """Aggregate post-execution audits into run-level metrics."""

    if audits.empty:
        return {
            "post_trade_audit_count": 0,
            "target_configured_breach_fraction": np.nan,
            "post_trade_configured_breach_fraction": np.nan,
            "post_trade_policy_violation_fraction": np.nan,
            "post_trade_material_configured_breach_fraction": np.nan,
            "post_trade_material_policy_violation_fraction": np.nan,
            "post_trade_execution_deterioration_fraction": np.nan,
            "post_trade_material_execution_deterioration_fraction": np.nan,
            "maximum_target_configured_constraint_breach": np.nan,
            "maximum_post_trade_configured_constraint_breach": np.nan,
            "maximum_post_trade_policy_violation": np.nan,
            "maximum_execution_constraint_deterioration": np.nan,
            "maximum_post_trade_active_beta_deviation": np.nan,
            "maximum_post_trade_industry_active_exposure": np.nan,
            "maximum_post_trade_adv_participation": np.nan,
            "maximum_post_trade_realized_day_participation": np.nan,
            "maximum_post_trade_target_weight_deviation": np.nan,
            "maximum_target_tracking_error": np.nan,
            "maximum_post_trade_tracking_error": np.nan,
            "mean_target_active_risk_utilization": np.nan,
            "p95_target_active_risk_utilization": np.nan,
            "maximum_target_active_risk_utilization": np.nan,
            "mean_actual_active_risk_utilization": np.nan,
            "p95_actual_active_risk_utilization": np.nan,
            "maximum_actual_active_risk_utilization": np.nan,
            "beta_audit_complete_fraction": np.nan,
        }
    return {
        "post_trade_audit_count": int(len(audits)),
        "target_configured_breach_fraction": _boolean_fraction(
            audits,
            "has_target_configured_breach",
        ),
        "post_trade_configured_breach_fraction": float(
            audits["has_configured_breach"].astype(bool).mean()
        ),
        "post_trade_policy_violation_fraction": float(
            audits["has_policy_violation"].astype(bool).mean()
        ),
        "post_trade_material_configured_breach_fraction": _boolean_fraction(
            audits,
            "has_material_configured_breach",
        ),
        "post_trade_material_policy_violation_fraction": _boolean_fraction(
            audits,
            "has_material_policy_violation",
        ),
        "post_trade_execution_deterioration_fraction": _boolean_fraction(
            audits,
            "has_execution_constraint_deterioration",
        ),
        "post_trade_material_execution_deterioration_fraction": (
            _boolean_fraction(
                audits,
                "has_material_execution_constraint_deterioration",
            )
        ),
        "maximum_target_configured_constraint_breach": _column_max(
            audits,
            "maximum_target_configured_constraint_breach",
        ),
        "maximum_post_trade_configured_constraint_breach": _column_max(
            audits,
            "maximum_configured_constraint_breach",
        ),
        "maximum_post_trade_policy_violation": _column_max(
            audits,
            "maximum_policy_violation",
        ),
        "maximum_execution_constraint_deterioration": _column_max(
            audits,
            "maximum_execution_constraint_deterioration",
        ),
        "maximum_post_trade_active_beta_deviation": _column_abs_max(
            audits,
            "actual_active_beta",
        ),
        "maximum_post_trade_industry_active_exposure": _column_max(
            audits,
            "maximum_industry_active_exposure",
        ),
        "maximum_post_trade_adv_participation": _column_max(
            audits,
            "maximum_adv_participation",
        ),
        "maximum_post_trade_realized_day_participation": _column_max(
            audits,
            "maximum_realized_day_participation",
        )
        if "maximum_realized_day_participation" in audits
        else np.nan,
        "maximum_post_trade_target_weight_deviation": _column_max(
            audits,
            "maximum_target_weight_deviation",
        ),
        "maximum_target_tracking_error": _column_max(
            audits,
            "target_tracking_error",
        ),
        "maximum_post_trade_tracking_error": _column_max(
            audits,
            "actual_tracking_error",
        ),
        "mean_target_active_risk_utilization": _column_mean(
            audits,
            "target_active_risk_utilization",
        ),
        "p95_target_active_risk_utilization": _column_quantile(
            audits,
            "target_active_risk_utilization",
            0.95,
        ),
        "maximum_target_active_risk_utilization": _column_max(
            audits,
            "target_active_risk_utilization",
        ),
        "mean_actual_active_risk_utilization": _column_mean(
            audits,
            "actual_active_risk_utilization",
        ),
        "p95_actual_active_risk_utilization": _column_quantile(
            audits,
            "actual_active_risk_utilization",
            0.95,
        ),
        "maximum_actual_active_risk_utilization": _column_max(
            audits,
            "actual_active_risk_utilization",
        ),
        "beta_audit_complete_fraction": float(audits["beta_audit_complete"].astype(bool).mean()),
    }


def _aligned_weights(values: pd.Series, names: pd.Index, label: str) -> pd.Series:
    result = pd.to_numeric(values.reindex(names, fill_value=0.0), errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all() or (result < -1e-10).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    return result.clip(lower=0.0).astype(float)


def _aligned_frame(
    frame: pd.DataFrame | None,
    names: pd.Index,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    if not frame.index.is_unique:
        raise ValueError("Portfolio exposure index must be unique")
    return frame.reindex(names).apply(pd.to_numeric, errors="coerce")


def _aggregate_exposures(
    frame: pd.DataFrame | None,
    actual: pd.Series,
    pre: pd.Series,
    target: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    if frame is None or frame.empty:
        return None, None, None
    matrix = frame.fillna(0.0)
    return (
        matrix.T @ (actual - benchmark),
        matrix.T @ (pre - benchmark),
        matrix.T @ (target - benchmark),
    )


def _industry_bucket(frame: pd.DataFrame | None, names: pd.Index) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(pd.NA, index=range(len(names)), dtype="string")
    filled = frame.fillna(0.0)
    maximum = filled.max(axis=1)
    bucket = filled.idxmax(axis=1).astype("string")
    bucket = bucket.where(maximum > 0, pd.NA)
    return bucket.reset_index(drop=True)


def _beta_diagnostics(
    *,
    settings: OptimizerSettings,
    aligned_styles: pd.DataFrame | None,
    actual: pd.Series,
    target: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    if aligned_styles is None or "market_beta_60" not in aligned_styles:
        return {
            "actual_active": np.nan,
            "actual_portfolio": np.nan,
            "benchmark": np.nan,
            "target_active": np.nan,
            "missing_actual_weight": float(actual.sum()),
            "missing_benchmark_weight": float(benchmark.sum()),
            "complete": False,
            "configured_breach": 0.0,
            "target_configured_breach": 0.0,
            "policy_violation": 0.0,
        }
    beta = pd.to_numeric(
        aligned_styles["market_beta_60"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    valid = beta.notna()
    values = beta.fillna(0.0)
    actual_active = float(values @ (actual - benchmark))
    target_active = float(values @ (target - benchmark))
    configured_breach = (
        max(0.0, abs(actual_active) - settings.beta_active_cap)
        if settings.beta_constraint_enabled
        else 0.0
    )
    target_configured_breach = (
        max(0.0, abs(target_active) - settings.beta_active_cap)
        if settings.beta_constraint_enabled
        else 0.0
    )
    if settings.beta_constraint_enabled:
        upper = max(settings.beta_active_cap, target_active)
        lower = min(-settings.beta_active_cap, target_active)
        policy_violation = max(0.0, actual_active - upper, lower - actual_active)
    else:
        policy_violation = 0.0
    missing_actual = float(actual.loc[~valid].sum())
    missing_benchmark = float(benchmark.loc[~valid].sum())
    return {
        "actual_active": actual_active,
        "actual_portfolio": float(values @ actual),
        "benchmark": float(values @ benchmark),
        "target_active": target_active,
        "missing_actual_weight": missing_actual,
        "missing_benchmark_weight": missing_benchmark,
        "complete": missing_actual <= 1e-8 and missing_benchmark <= 1e-8,
        "configured_breach": configured_breach,
        "target_configured_breach": target_configured_breach,
        "policy_violation": policy_violation,
    }


def _tracking_error(
    active: pd.Series,
    covariance: pd.DataFrame,
    annualization: int,
) -> float:
    names = covariance.index
    if (
        covariance.empty
        or not names.is_unique
        or not covariance.columns.equals(names)
        or not np.isfinite(covariance.to_numpy(dtype=float)).all()
    ):
        return np.nan
    vector = active.reindex(names, fill_value=0.0).to_numpy(dtype=float)
    variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
    return float(np.sqrt(max(variance, 0.0) * annualization))


def _maximum_ex_ante_participation(
    *,
    trade_records: Sequence[Mapping[str, Any]],
) -> tuple[float, int]:
    participations: list[float] = []
    missing = 0
    for record in trade_records:
        gross = float(record.get("gross_value", 0.0) or 0.0)
        if gross <= 0:
            continue
        raw_participation = record.get("executed_adv_participation")
        participation = float(raw_participation) if raw_participation is not None else np.nan
        if not np.isfinite(participation) or participation < 0:
            missing += 1
            continue
        participations.append(participation)
    return max(participations, default=0.0), missing


def _maximum_realized_day_participation(
    *,
    settings: OptimizerSettings,
    trade_records: Sequence[Mapping[str, Any]],
    execution_day: pd.DataFrame,
    pre_nav: float,
) -> tuple[float, int]:
    if not settings.liquidity_enabled:
        return np.nan, 0
    if pre_nav <= 0:
        raise ValueError("pre_nav must be positive for liquidity auditing")
    participations: list[float] = []
    missing = 0
    for record in trade_records:
        gross = float(record.get("gross_value", 0.0) or 0.0)
        if gross <= 0:
            continue
        instrument = str(record.get("instrument", ""))
        if instrument not in execution_day.index or "amount_cny" not in execution_day:
            missing += 1
            continue
        amount = float(execution_day.at[instrument, "amount_cny"])
        if not np.isfinite(amount) or amount <= 0:
            missing += 1
            continue
        notional_cny = gross / pre_nav * settings.portfolio_aum_cny
        participations.append(notional_cny / amount)
    return max(participations, default=0.0), missing


def _positive_max(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return max(0.0, float(numeric.max())) if not numeric.empty else 0.0


def _gap_deterioration(actual: pd.Series, target: pd.Series) -> float:
    if actual.empty and target.empty:
        return 0.0
    names = actual.index.union(target.index)
    actual_gap = np.maximum(
        pd.to_numeric(actual.reindex(names), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    target_gap = np.maximum(
        pd.to_numeric(target.reindex(names), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    return max(0.0, float((actual_gap - target_gap).max()))


def _json_series(values: pd.Series | None) -> str:
    if values is None:
        return "{}"
    payload = {str(key): float(value) for key, value in values.items() if np.isfinite(float(value))}
    return json.dumps(payload, sort_keys=True)


def _column_max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def _column_abs_max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna().abs()
    return float(values.max()) if not values.empty else np.nan


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _column_quantile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else np.nan


def _boolean_fraction(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    return float(frame[column].astype(bool).mean())
