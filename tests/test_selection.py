from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.workflow.selection import (
    StabilityCostSelector,
    StabilityCostSettings,
    _benjamini_hochberg,
    _correlation_clusters,
    _equal_weight_leg_turnover_and_cost,
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
    assert diagnostics["stable"]["mean_net_quintile_spread"] > 0.0
    assert diagnostics["stable"]["joint_segment_frequency"] == 1.0
    assert diagnostics["stable"]["bh_q_value"] <= 0.20
    assert diagnostics["duplicate"]["correlation_cluster"] == diagnostics[
        "stable"
    ]["correlation_cluster"]


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


def test_stability_cost_selector_incremental_cache_matches_fresh_fit() -> None:
    panel = _selection_panel()
    delayed = panel["decision_date"].eq("20240116")
    delayed &= panel["instrument"].isin(
        [f"S{position:03d}" for position in range(10)]
    )
    panel.loc[delayed, "label_available_date"] = "20240215"
    candidates = ("stable", "duplicate", "costly", "unstable", "wrong_way")
    directions = {factor: 1 for factor in candidates}
    families = {factor: factor for factor in candidates}
    incremental = StabilityCostSelector(
        directions=directions,
        families=families,
        settings=_settings(),
    )
    incremental.select(panel, candidates, as_of_date="20240131")
    cached = incremental.select(panel, candidates, as_of_date="20251231")
    fresh = StabilityCostSelector(
        directions=directions,
        families=families,
        settings=_settings(),
    ).select(panel, candidates, as_of_date="20251231")

    assert cached.factor_names == fresh.factor_names
    assert cached.diagnostics == fresh.diagnostics


def test_stability_cost_selector_supports_preregistered_factor_ablation() -> None:
    panel = _selection_panel()
    candidates = ("stable", "duplicate", "costly", "unstable", "wrong_way")
    base = _settings()
    settings = StabilityCostSettings(
        **{
            **base.__dict__,
            "excluded_factors": ("stable",),
        }
    )
    selector = StabilityCostSelector(
        directions={factor: 1 for factor in candidates},
        families={factor: factor for factor in candidates},
        settings=settings,
    )

    result = selector.select(panel, candidates, as_of_date="20251231")

    assert "stable" not in result.diagnostics["factor_diagnostics"]
    assert result.diagnostics["provided_candidate_count"] == 5
    assert result.diagnostics["candidate_count"] == 4
    assert result.diagnostics["excluded_factors"] == ["stable"]


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


def test_stability_cost_selector_fails_closed_without_minimum_factor_fallback() -> None:
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

    assert result.factor_names == ()
    assert not result.diagnostics["fallback_used"]
    assert result.diagnostics["selection_shortfall"]
    assert result.diagnostics["evidence_selected_count"] == 0


def test_stability_cost_selector_closes_a_partial_selection_below_minimum() -> None:
    panel = _selection_panel()
    selector = StabilityCostSelector(
        directions={"stable": 1},
        families={"stable": "quality"},
        settings=StabilityCostSettings(
            min_coverage=0.90,
            min_active_date_rate=0.90,
            min_cross_section=20,
            min_ic_dates=20,
            min_mean_directed_ic=0.02,
            min_direction_consistency=0.70,
            min_segment_selection_frequency=0.75,
            min_net_spread_consistency=0.70,
            min_joint_segment_frequency=0.75,
            min_newey_west_t=1.0,
            min_quintile_monotonicity=0.50,
            max_score_churn=0.10,
            min_factors=2,
            max_factors=2,
        ),
    )

    result = selector.select(panel, ("stable",), as_of_date="20251231")

    assert result.factor_names == ()
    assert result.diagnostics["evidence_selected_factors"] == ["stable"]
    stable = result.diagnostics["factor_diagnostics"]["stable"]
    assert stable["status"] == "rejected"
    assert "selection_count_below_minimum" in stable["reasons"]


def test_stability_cost_selector_rejects_positive_ic_with_negative_net_tail() -> None:
    panel = _selection_panel()
    panel["forward_active_return"] *= 0.01
    selector = StabilityCostSelector(
        directions={"stable": 1},
        families={"stable": "quality"},
        settings=StabilityCostSettings(
            min_coverage=0.90,
            min_active_date_rate=0.90,
            min_cross_section=20,
            min_ic_dates=20,
            min_mean_directed_ic=0.02,
            min_direction_consistency=0.70,
            min_segment_selection_frequency=0.75,
            min_mean_net_quintile_spread=0.0,
            min_net_spread_consistency=0.70,
            min_joint_segment_frequency=0.75,
            min_newey_west_t=1.0,
            min_quintile_monotonicity=0.50,
            max_score_churn=0.10,
            min_factors=1,
            max_factors=1,
            linear_cost_bps=20_000.0,
        ),
    )

    result = selector.select(panel, ("stable",), as_of_date="20251231")

    diagnostic = result.diagnostics["factor_diagnostics"]["stable"]
    assert diagnostic["mean_directed_ic"] > 0.0
    assert diagnostic["mean_net_quintile_spread"] < 0.0
    assert "mean_net_quintile_spread_below_minimum" in diagnostic["reasons"]
    assert result.factor_names == ()


def test_correlation_clusters_are_transitive() -> None:
    factors = ("a", "b", "c")
    columns = [f"{factor}__z" for factor in factors]
    correlation = pd.DataFrame(
        [
            [1.0, 0.90, 0.20],
            [0.90, 1.0, 0.90],
            [0.20, 0.90, 1.0],
        ],
        index=columns,
        columns=columns,
    )

    cluster_by_factor, members = _correlation_clusters(
        factors,
        correlation,
        threshold=0.85,
    )

    assert len(set(cluster_by_factor.values())) == 1
    assert next(iter(members.values())) == factors


def test_benjamini_hochberg_q_values_are_monotone_and_auditable() -> None:
    q_values = _benjamini_hochberg(
        {"a": 0.01, "b": 0.04, "c": 0.03, "missing": np.nan}
    )

    assert q_values["a"] == pytest.approx(0.03)
    assert q_values["b"] == pytest.approx(0.04)
    assert q_values["c"] == pytest.approx(0.04)
    assert np.isnan(q_values["missing"])


@pytest.mark.parametrize(
    ("current", "previous"),
    [
        ({"A", "B"}, None),
        ({"A", "B"}, {"A", "B"}),
        ({"A", "B", "C"}, {"A", "B"}),
        ({"A", "B"}, {"A", "B", "C"}),
        ({"A", "B"}, {"C", "D"}),
        ({"A", "B", "C"}, {"B", "C", "D", "E"}),
    ],
)
def test_equal_weight_turnover_matches_explicit_weight_alignment(
    current: set[str],
    previous: set[str] | None,
) -> None:
    linear_rate = 0.0005
    stamp_rate = 0.001
    current_weights = pd.Series(
        1.0 / len(current),
        index=sorted(current),
        dtype=float,
    )
    previous_weights = (
        pd.Series(
            1.0 / len(previous),
            index=sorted(previous),
            dtype=float,
        )
        if previous
        else pd.Series(dtype=float)
    )
    instruments = current_weights.index.union(previous_weights.index)
    delta = current_weights.reindex(instruments, fill_value=0.0) - (
        previous_weights.reindex(instruments, fill_value=0.0)
    )
    expected_buys = float(delta.clip(lower=0.0).sum())
    expected_sells = float((-delta.clip(upper=0.0)).sum())
    expected_turnover = max(expected_buys, expected_sells)
    expected_cost = (
        linear_rate * (expected_buys + expected_sells)
        + stamp_rate * expected_sells
    )

    turnover, cost = _equal_weight_leg_turnover_and_cost(
        frozenset(current),
        frozenset(previous) if previous else None,
        linear_rate=linear_rate,
        stamp_rate=stamp_rate,
    )

    assert turnover == pytest.approx(expected_turnover, abs=1e-15)
    assert cost == pytest.approx(expected_cost, abs=1e-15)


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
