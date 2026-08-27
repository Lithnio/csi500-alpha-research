from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.workflow.selection import (
    StabilityCostSelector,
    StabilityCostSettings,
)


def test_stability_cost_selector_rejects_unstable_redundant_and_costly_factors() -> None:
    panel = _selection_panel()
    candidates = ("stable", "duplicate", "costly", "unstable", "wrong_way")
    selector = StabilityCostSelector(
        directions={factor: 1 for factor in candidates},
        families={
            "stable": "quality",
            "duplicate": "quality",
            "costly": "liquidity",
            "unstable": "reversal",
            "wrong_way": "value",
        },
        settings=_settings(),
    )

    selected = selector.select(panel, candidates, as_of_date="20251231")

    assert selected.factor_names == ("stable",)
    diagnostics = selected.diagnostics["factor_diagnostics"]
    assert diagnostics["stable"]["status"] == "selected"
    assert any(
        reason.startswith("correlated_with=stable")
        for reason in diagnostics["duplicate"]["reasons"]
    )
    assert "score_churn_above_maximum" in diagnostics["costly"]["reasons"]
    assert "direction_consistency_below_minimum" in diagnostics["unstable"]["reasons"]
    assert "mean_directed_ic_below_minimum" in diagnostics["wrong_way"]["reasons"]
    assert diagnostics["stable"]["newey_west_t"] > 1.0
    assert diagnostics["stable"]["segment_selection_frequency"] == 1.0


def test_stability_cost_selector_ignores_labels_unavailable_as_of_fit() -> None:
    panel = _selection_panel()
    candidates = ("stable", "duplicate", "costly", "unstable", "wrong_way")
    selector = StabilityCostSelector(
        directions={factor: 1 for factor in candidates},
        families={factor: factor for factor in candidates},
        settings=_settings(),
    )
    baseline = selector.select(panel, candidates, as_of_date="20251231")

    future = panel.iloc[:120].copy()
    future["decision_date"] = "20260105"
    future["label_available_date"] = "20260112"
    future["forward_active_return"] = 999.0
    for factor in candidates:
        future[f"{factor}__z"] = np.linspace(-999.0, 999.0, len(future))
    revised = selector.select(
        pd.concat([panel, future], ignore_index=True),
        candidates,
        as_of_date="20251231",
    )

    assert revised.factor_names == baseline.factor_names
    assert revised.diagnostics == baseline.diagnostics


def test_stability_cost_selector_requires_explicit_label_availability() -> None:
    panel = _selection_panel().drop(columns="label_available_date")
    selector = StabilityCostSelector(
        directions={"stable": 1},
        families={"stable": "quality"},
        settings=_settings(),
    )

    with pytest.raises(ConfigurationError, match="label_available_date"):
        selector.select(panel, ("stable",), as_of_date="20251231")


def test_stability_cost_selector_rejects_invalid_economic_direction() -> None:
    with pytest.raises(ConfigurationError, match="directions must be -1 or 1"):
        StabilityCostSelector(
            directions={"factor": 0},
            families={},
            settings=_settings(),
        )


def test_stability_cost_selector_has_audited_minimum_factor_fallback() -> None:
    candidates = ("a", "b", "c")
    empty = pd.DataFrame(
        columns=[
            "decision_date",
            "instrument",
            "label_available_date",
            "forward_active_return",
            *(f"{factor}__z" for factor in candidates),
        ]
    )
    selector = StabilityCostSelector(
        directions={factor: 1 for factor in candidates},
        families={factor: "one_family" for factor in candidates},
        settings=StabilityCostSettings(
            min_cross_section=5,
            min_ic_dates=2,
            min_factors=2,
            max_factors=2,
        ),
    )

    result = selector.select(empty, candidates, as_of_date="20250101")

    assert len(result.factor_names) == 2
    assert result.diagnostics["fallback_used"]
    factor_diagnostics = result.diagnostics["factor_diagnostics"]
    assert all(
        factor_diagnostics[factor]["status"] == "fallback_selected"
        for factor in result.factor_names
    )
    assert all(
        "selected_by_min_factor_fallback" in factor_diagnostics[factor]["reasons"]
        for factor in result.factor_names
    )


def _settings() -> StabilityCostSettings:
    return StabilityCostSettings(
        min_coverage=0.90,
        min_cross_section=20,
        min_ic_dates=20,
        min_mean_directed_ic=0.02,
        min_direction_consistency=0.70,
        segments=4,
        min_segment_selection_frequency=0.75,
        min_newey_west_t=1.0,
        min_quintile_monotonicity=0.50,
        max_score_churn=0.10,
        max_abs_correlation=0.90,
        min_factors=1,
        max_factors=3,
        max_per_family=2,
        lookback_dates=100,
        churn_penalty=0.05,
    )


def _selection_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=40).strftime("%Y%m%d")
    instruments = [f"S{position:03d}" for position in range(60)]
    latent = np.linspace(-2.0, 2.0, len(instruments))
    duplicate_noise = rng.normal(0.0, 0.08, len(instruments))
    rows: list[dict[str, object]] = []
    for date_position, date in enumerate(dates):
        label = latent + rng.normal(0.0, 0.05, len(instruments))
        costly = latent + rng.normal(0.0, 2.0, len(instruments))
        unstable_sign = 1.0 if date_position < len(dates) / 2 else -1.0
        for position, instrument in enumerate(instruments):
            rows.append(
                {
                    "decision_date": date,
                    "instrument": instrument,
                    "label_available_date": date,
                    "forward_active_return": label[position],
                    "stable__z": latent[position],
                    "duplicate__z": latent[position] + duplicate_noise[position],
                    "costly__z": costly[position],
                    "unstable__z": unstable_sign * latent[position],
                    "wrong_way__z": -latent[position],
                }
            )
    return pd.DataFrame(rows)
