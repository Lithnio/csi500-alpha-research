import numpy as np
import pandas as pd
import pytest

from csi500_alpha.config import OptimizerSettings
from csi500_alpha.portfolio.audit import (
    ExecutedPortfolioAudit,
    audit_executed_portfolio,
    summarize_constraint_audits,
)


def _settings() -> OptimizerSettings:
    return OptimizerSettings(
        enabled=True,
        risk_aversion=1.0,
        risk_horizon_days=5,
        l2_penalty=0.01,
        active_cap=0.10,
        name_cap=0.70,
        turnover_cap=0.20,
        initial_turnover_cap=1.0,
        exposure_cap=0.05,
        solvers=("CLARABEL",),
        beta_constraint_enabled=True,
        beta_active_cap=0.05,
        liquidity_enabled=True,
        portfolio_aum_cny=100_000_000.0,
        adv_lookback=20,
        min_adv_observations=10,
        max_adv_participation=0.05,
        impact_bps_at_max_participation=10.0,
    )


def _audit(
    actual_a_weight: float,
    *,
    pre_a_weight: float = 0.50,
    target_a_weight: float = 0.54,
    execution_day_amount_cny: float = 100_000_000.0,
) -> ExecutedPortfolioAudit:
    names = pd.Index(["A", "B"], name="instrument")
    industry = pd.DataFrame(
        {"industry_1": [1.0, 0.0], "industry_2": [0.0, 1.0]},
        index=names,
    )
    styles = pd.DataFrame(
        {
            "market_beta_60": [1.4, 0.6],
            "small_size_z": [1.0, -1.0],
        },
        index=names,
    )
    gross = abs(actual_a_weight - pre_a_weight)
    trade_records = [
        {
            "instrument": "A",
            "gross_value": gross,
            "executed_adv_participation": gross,
        },
        {
            "instrument": "B",
            "gross_value": gross,
            "executed_adv_participation": gross,
        },
    ]
    return audit_executed_portfolio(
        signal_date="20250102",
        execution_date="20250103",
        settings=_settings(),
        risk_annualization=252,
        pre_weights=pd.Series(
            {"A": pre_a_weight, "B": 1.0 - pre_a_weight}
        ),
        pre_cash_weight=0.0,
        actual_weights=pd.Series({"A": actual_a_weight, "B": 1.0 - actual_a_weight}),
        actual_cash_weight=0.0,
        benchmark=pd.Series({"A": 0.50, "B": 0.50}),
        target=pd.Series(
            {"A": target_a_weight, "B": 1.0 - target_a_weight}
        ),
        covariance=pd.DataFrame(np.eye(2) * 0.0001, index=names, columns=names),
        industry_exposures=industry,
        style_exposures=styles,
        trade_records=trade_records,
        execution_day=pd.DataFrame(
            {"amount_cny": [execution_day_amount_cny] * 2},
            index=names,
        ),
        pre_nav=1.0,
    )


def test_post_trade_audit_records_positions_and_passes_repair_policy() -> None:
    audit = _audit(0.54)

    assert audit.positions["actual_weight"].sum() == 1.0
    np.testing.assert_allclose(
        audit.summary["actual_active_beta"],
        0.032,
    )
    assert audit.summary["maximum_industry_active_exposure"] == pytest.approx(0.04)
    assert audit.summary["maximum_adv_participation"] == pytest.approx(0.04)
    assert audit.summary["maximum_realized_day_participation"] == pytest.approx(
        0.04
    )
    assert audit.summary["has_policy_violation"] is False
    assert audit.summary["beta_audit_complete"] is True


def test_post_trade_audit_exposes_execution_constraint_breaches() -> None:
    audit = _audit(0.60)
    summary = summarize_constraint_audits(pd.DataFrame([audit.summary]))

    assert audit.summary["configured_beta_cap_breach"] > 0.0
    assert audit.summary["configured_industry_cap_breaches"] == 2
    assert audit.summary["has_policy_violation"] is True
    assert audit.summary["has_target_configured_breach"] is False
    assert audit.summary["has_material_execution_constraint_deterioration"] is True
    assert summary["post_trade_policy_violation_fraction"] == 1.0
    assert summary["post_trade_material_execution_deterioration_fraction"] == 1.0
    assert summary["maximum_post_trade_active_beta_deviation"] == pytest.approx(0.08)


def test_post_trade_audit_separates_target_gap_from_execution_deterioration() -> None:
    audit = _audit(0.60, pre_a_weight=0.60, target_a_weight=0.60)

    assert audit.summary["has_target_configured_breach"] is True
    assert audit.summary["has_configured_breach"] is True
    assert audit.summary["has_policy_violation"] is False
    assert audit.summary["has_execution_constraint_deterioration"] is False
    assert audit.summary["has_material_execution_constraint_deterioration"] is False
    assert audit.summary["target_active_risk_utilization"] == pytest.approx(
        audit.summary["actual_active_risk_utilization"]
    )


def test_realized_day_volume_is_diagnostic_not_an_execution_policy() -> None:
    audit = _audit(0.54, execution_day_amount_cny=1_000_000.0)

    assert audit.summary["maximum_ex_ante_adv_participation"] == pytest.approx(
        0.04
    )
    assert audit.summary["maximum_realized_day_participation"] == pytest.approx(
        4.0
    )
    assert audit.summary["has_policy_violation"] is False
