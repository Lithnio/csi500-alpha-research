from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.incremental_selection import (
    IncrementalAdmissionSettings,
    IncrementalStabilityCostSelector,
)
from csi500_alpha.workflow.selection import StabilityCostSelector, StabilityCostSettings


def test_incremental_selector_keeps_core_testing_domain_and_admits_residual_signal() -> None:
    panel = _incremental_panel()
    core_settings = _selector_settings(min_factors=1, max_factors=1)
    addition_settings = _selector_settings(min_factors=1, max_factors=2)
    selector = IncrementalStabilityCostSelector(
        directions={"core": 1, "incremental": 1, "core_clone": 1},
        families={
            "core": "quality",
            "incremental": "event",
            "core_clone": "momentum",
        },
        settings=IncrementalAdmissionSettings(
            candidate_factors=("incremental", "core_clone"),
            core=core_settings,
            additions=addition_settings,
            min_residual_variance_ratio=0.05,
        ),
    )

    result = selector.select(
        panel,
        ("core", "incremental", "core_clone"),
        as_of_date="20250101",
    )
    standalone = StabilityCostSelector(
        directions={"core": 1},
        families={"core": "quality"},
        settings=core_settings,
    ).select(panel, ("core",), as_of_date="20250101")

    assert result.factor_names == ("core", "incremental")
    diagnostics = result.diagnostics
    assert diagnostics["multiple_testing_domains"] == {
        "core": ["core"],
        "additions": ["incremental", "core_clone"],
    }
    assert diagnostics["core_selection"] == standalone.diagnostics
    residuals = diagnostics["residualization"]
    assert residuals["incremental"]["passed_variance_gate"]
    assert residuals["core_clone"]["residual_variance_ratio"] < 0.05
    clone = diagnostics["factor_diagnostics"]["core_clone"]
    assert "residual_variance_below_minimum" in clone["reasons"]


def test_incremental_selector_additional_candidate_cannot_change_core_q_value() -> None:
    panel = _incremental_panel(include_noise=True)
    common = {
        "directions": {
            "core": 1,
            "incremental": 1,
            "core_clone": 1,
            "noise": 1,
        },
        "families": {
            "core": "quality",
            "incremental": "event",
            "core_clone": "momentum",
            "noise": "noise",
        },
    }
    settings = _selector_settings(min_factors=1, max_factors=2)
    first = IncrementalStabilityCostSelector(
        **common,
        settings=IncrementalAdmissionSettings(
            candidate_factors=("incremental", "core_clone"),
            core=_selector_settings(min_factors=1, max_factors=1),
            additions=settings,
        ),
    ).select(
        panel,
        ("core", "incremental", "core_clone"),
        as_of_date="20250101",
    )
    expanded = IncrementalStabilityCostSelector(
        **common,
        settings=IncrementalAdmissionSettings(
            candidate_factors=("incremental", "core_clone", "noise"),
            core=_selector_settings(min_factors=1, max_factors=1),
            additions=settings,
        ),
    ).select(
        panel,
        ("core", "incremental", "core_clone", "noise"),
        as_of_date="20250101",
    )

    assert first.diagnostics["core_selection"] == expanded.diagnostics[
        "core_selection"
    ]
    first_core = first.diagnostics["factor_diagnostics"]["core"]
    expanded_core = expanded.diagnostics["factor_diagnostics"]["core"]
    assert first_core["bh_q_value"] == expanded_core["bh_q_value"]


def test_incremental_selector_ignores_future_and_unavailable_labels() -> None:
    panel = _incremental_panel()
    selector = IncrementalStabilityCostSelector(
        directions={"core": 1, "incremental": 1, "core_clone": 1},
        families={
            "core": "quality",
            "incremental": "event",
            "core_clone": "momentum",
        },
        settings=IncrementalAdmissionSettings(
            candidate_factors=("incremental", "core_clone"),
            core=_selector_settings(min_factors=1, max_factors=1),
            additions=_selector_settings(min_factors=1, max_factors=2),
        ),
    )
    baseline = selector.select(
        panel,
        ("core", "incremental", "core_clone"),
        as_of_date="20250101",
    )
    future = panel.iloc[:80].copy()
    future["decision_date"] = "20250102"
    future["label_available_date"] = "20250110"
    future["forward_active_return"] = 999.0
    future["core__z"] = np.linspace(-100.0, 100.0, len(future))
    future["incremental__z"] = future["core__z"]
    contaminated = selector.select(
        pd.concat([panel, future], ignore_index=True),
        ("core", "incremental", "core_clone"),
        as_of_date="20250101",
    )

    assert contaminated.factor_names == baseline.factor_names
    assert contaminated.diagnostics == baseline.diagnostics


def test_incremental_selector_registry_validates_nested_contract() -> None:
    registry = default_component_registry()
    selector = registry.create_selector(
        "incremental_stability_cost",
        {
            "candidate_factors": ["limit_up_close_rate_20"],
            "core": {"min_factors": 1, "max_factors": 1},
            "additions": {"min_factors": 1, "max_factors": 1},
        },
        directions={
            "reversal_5d": 1,
            "limit_up_close_rate_20": -1,
        },
    )

    assert selector.name == "incremental_stability_cost"

    with pytest.raises(ConfigurationError, match="candidate_factors"):
        registry.create_selector(
            "incremental_stability_cost",
            {"candidate_factors": "limit_up_close_rate_20"},
            directions={"limit_up_close_rate_20": -1},
        )


def _selector_settings(*, min_factors: int, max_factors: int) -> StabilityCostSettings:
    return StabilityCostSettings(
        min_coverage=0.90,
        min_active_date_rate=0.90,
        min_cross_section=40,
        min_ic_dates=24,
        min_mean_directed_ic=0.01,
        min_direction_consistency=0.60,
        segments=3,
        min_segment_selection_frequency=0.66,
        min_mean_net_quintile_spread=0.0,
        min_net_spread_consistency=0.60,
        min_joint_segment_frequency=0.66,
        min_newey_west_t=1.0,
        max_bh_q_value=0.20,
        min_quintile_monotonicity=0.50,
        max_score_churn=1.0,
        max_abs_correlation=0.90,
        min_factors=min_factors,
        max_factors=max_factors,
        max_per_family=2,
        max_per_cluster=1,
        lookback_dates=36,
        churn_penalty=0.0,
        linear_cost_bps=0.0,
        stamp_duty_before=0.0,
        stamp_duty_after=0.0,
    )


def _incremental_panel(*, include_noise: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    dates = pd.bdate_range("2024-01-02", periods=36).strftime("%Y%m%d")
    instruments = [f"S{position:03d}" for position in range(80)]
    rows: list[dict[str, object]] = []
    for date in dates:
        core = rng.normal(size=len(instruments))
        incremental = rng.normal(size=len(instruments))
        label = 0.70 * core + 0.55 * incremental + rng.normal(
            scale=0.20,
            size=len(instruments),
        )
        clone = core + rng.normal(scale=0.005, size=len(instruments))
        noise = rng.normal(size=len(instruments))
        for position, instrument in enumerate(instruments):
            row: dict[str, object] = {
                "decision_date": str(date),
                "instrument": instrument,
                "label_available_date": str(date),
                "forward_active_return": float(label[position]),
                "core__z": float(core[position]),
                "incremental__z": float(incremental[position]),
                "core_clone__z": float(clone[position]),
            }
            if include_noise:
                row["noise__z"] = float(noise[position])
            rows.append(row)
    return pd.DataFrame(rows)
