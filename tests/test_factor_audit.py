from __future__ import annotations

from pathlib import Path

import pandas as pd

from csi500_alpha.research.diagnostics import compute_factor_diagnostics
from csi500_alpha.research.factor_audit import (
    FactorAuditGates,
    FactorAuditSpec,
    build_factor_audit_tables,
)


def test_v2_factor_audit_plan_resolves_all_candidates() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = FactorAuditSpec.from_yaml(root / "configs" / "factor_audit_v2.yaml")
    config = spec.resolved_config()

    assert spec.audit_id == "csi500-factor-audit-v2"
    assert spec.start_date == "20170103"
    assert spec.end_date == "20251231"
    assert config.workflow.feature_provider.name == "builtin_daily_fundamental"
    assert config.experiment.stage == "walk_forward"


def test_a2_factor_audit_uses_explicit_expanded_provider() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = FactorAuditSpec.from_yaml(
        root / "configs" / "factor_audit_a2_price_volume.yaml"
    )
    config = spec.resolved_config()

    assert spec.audit_id == "csi500-factor-audit-a2-price-volume-v2-validation"
    assert spec.feature_provider_name == "builtin_a2_daily"
    assert config.workflow.feature_provider.name == "builtin_a2_daily"


def test_factor_audit_orients_returns_and_applies_hard_gates() -> None:
    dates = (
        "20240102",
        "20240202",
        "20240304",
        "20250102",
        "20250203",
        "20250303",
    )
    instruments = [f"S{position:02d}" for position in range(10)]
    raw_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    for date in dates:
        for rank, instrument in enumerate(instruments):
            centered = float(rank - 4.5)
            raw_rows.append(
                {
                    "decision_date": date,
                    "instrument": instrument,
                    "industry_code": "I1" if rank < 5 else "I2",
                    "circ_mv_cny": 1_000_000_000.0 + rank,
                    "total_mv_cny": 2_000_000_000.0 + rank,
                    "good": centered,
                    "inverse": -centered,
                    "good__z": centered,
                    "inverse__z": -centered,
                }
            )
            label_rows.append(
                {
                    "decision_date": date,
                    "instrument": instrument,
                    "label_entry_date": date,
                    "label_valid": True,
                    "forward_active_return": centered / 100.0,
                }
            )
            market_rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "amount_cny": 10_000_000.0 + rank * 100_000.0,
                }
            )
        for factor in ("good", "inverse"):
            quality_rows.append(
                {
                    "decision_date": date,
                    "factor": factor,
                    "coverage": 1.0,
                    "active": True,
                    "clipped_fraction": 0.0,
                    "residual_std": 1.0,
                    "industry_coverage": 1.0,
                    "industry_neutralized": True,
                }
            )
    features = pd.DataFrame(raw_rows)
    labels = pd.DataFrame(label_rows)
    quality = pd.DataFrame(quality_rows)
    diagnostics = compute_factor_diagnostics(
        features=features,
        labels=labels,
        feature_quality=quality,
        factor_names=("good", "inverse"),
        directions={"good": 1, "inverse": -1},
    )

    audit = build_factor_audit_tables(
        raw_features=features.drop(columns=["good__z", "inverse__z"]),
        processed_features=features,
        labels=labels,
        feature_quality=quality,
        diagnostics=diagnostics,
        market_panel=pd.DataFrame(market_rows),
        open_dates=dates,
        factor_names=("good", "inverse"),
        directions={"good": 1, "inverse": -1},
        families={"good": "test", "inverse": "test"},
        gates=FactorAuditGates(
            min_mean_coverage=0.80,
            min_active_date_rate=0.80,
            min_ic_dates=4,
            min_audited_years=2,
            min_positive_year_fraction=1.0,
        ),
        label_horizon=5,
        linear_cost_bps=5.0,
        stamp_duty_change_date="20230828",
        stamp_duty_before=0.001,
        stamp_duty_after=0.0005,
        adv_window=2,
        max_adv_participation=0.05,
    )

    assert audit.summary["eligible"].all()
    assert audit.summary["positive_joint_year_fraction"].eq(1.0).all()
    assert audit.summary["median_yearly_q5_minus_q1_net"].gt(0).all()
    assert audit.summary["lookahead_violations"].eq(0).all()
    inverse = audit.rebalance_spreads.loc[
        audit.rebalance_spreads["factor"].eq("inverse")
    ]
    assert inverse["q5_minus_q1_gross"].gt(0).all()
