from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.family_models import FamilyModelSettings
from csi500_alpha.workflow.residual_sleeve import (
    ResidualSleeveBlendAlphaModel,
    ResidualSleeveSettings,
)

CORE_FACTORS = ("core_a", "core_b", "core_c")
CANDIDATE = "candidate"
DIRECTIONS = {**{factor: 1 for factor in CORE_FACTORS}, CANDIDATE: 1}
FAMILIES = {
    "core_a": "core_family_a",
    "core_b": "core_family_b",
    "core_c": "core_family_c",
    CANDIDATE: "candidate_family",
}


def _settings() -> ResidualSleeveSettings:
    core = FamilyModelSettings(
        min_cross_section=10,
        min_training_rows=200,
        min_training_dates=12,
        min_ic_dates=12,
        lookback_dates=60,
        ridge_alpha=1.0,
        min_factor_fraction=0.50,
        min_family_fraction=0.50,
        min_active_factors=3,
        min_active_families=3,
        max_factor_weight=1.0,
        max_family_weight=1.0,
    )
    return ResidualSleeveSettings(
        candidate_factors=(CANDIDATE,),
        core=core,
        min_core_fraction=0.50,
        min_candidate_fraction=1.0,
        min_cross_section=10,
        min_sleeve_dates=24,
        lookback_dates=60,
        oof_segments=4,
        min_oof_train_dates=12,
        min_oof_blocks=2,
        min_oof_positive_block_fraction=0.50,
        oof_target_t=0.50,
        hac_max_lags=2,
        risk_aversion=0.10,
        candidate_anchor_penalty=0.0001,
        candidate_change_penalty=0.0001,
        linear_cost_bps=0.0,
        stamp_duty_before=0.0,
        stamp_duty_after=0.0,
    )


def _panel(*, candidate_strength: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    dates = pd.bdate_range("2020-01-02", periods=60)
    instruments = [f"S{position:03d}" for position in range(40)]
    rows: list[dict[str, object]] = []
    for date in dates:
        decision_date = date.strftime("%Y%m%d")
        available_date = (date + pd.offsets.BDay(2)).strftime("%Y%m%d")
        core = rng.normal(size=(len(instruments), len(CORE_FACTORS)))
        candidate = rng.normal(size=len(instruments))
        core_score = core.mean(axis=1)
        noise = rng.normal(scale=0.01, size=len(instruments))
        label = 0.01 * core_score + candidate_strength * candidate + noise
        for position, instrument in enumerate(instruments):
            rows.append(
                {
                    "decision_date": decision_date,
                    "label_available_date": available_date,
                    "label_entry_date": decision_date,
                    "instrument": instrument,
                    "forward_active_return": label[position],
                    "core_a__z": core[position, 0],
                    "core_b__z": core[position, 1],
                    "core_c__z": core[position, 2],
                    "candidate__z": candidate[position],
                }
            )
    return pd.DataFrame(rows)


def _model(
    settings: ResidualSleeveSettings | None = None,
) -> ResidualSleeveBlendAlphaModel:
    return ResidualSleeveBlendAlphaModel(
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=settings or _settings(),
    )


def test_residual_sleeve_uses_continuous_oof_net_candidate_share() -> None:
    panel = _panel()
    model = _model()

    summary = model.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    parameters = summary.parameters
    evidence = parameters["sleeve_evidence"]
    assert parameters["method"] == "oof_net_residual_sleeve_blend"
    assert evidence["oof_evidence_scope"] == "conditional_on_admitted_candidates"
    assert evidence["oof_evidence_passed"] is True
    assert evidence["oof_block_count"] >= 2
    assert 0.0 < parameters["candidate_share"] <= 1.0
    assert parameters["core_share"] + parameters["candidate_share"] == pytest.approx(
        1.0
    )
    assert parameters["candidate_share"] == pytest.approx(
        parameters["candidate_multiplier"]
        / (1.0 + parameters["candidate_multiplier"])
    )
    assert sum(parameters["factor_weights"].values()) == pytest.approx(1.0)
    assert parameters["residualization"][CANDIDATE]["evaluated_dates"] == 60

    latest = panel.loc[panel["decision_date"].eq(panel["decision_date"].max())]
    score = model.predict(latest)
    assert score.notna().all()
    assert score.index.equals(latest.index)


def test_residual_satellite_can_add_value_without_replacing_core() -> None:
    panel = _panel(candidate_strength=0.0052)
    model = _model(
        replace(
            _settings(),
            candidate_anchor_penalty=0.005,
            oof_target_t=0.0,
        )
    )

    summary = model.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    evidence = summary.parameters["sleeve_evidence"]
    assert evidence["candidate_mean_net_spread"] < evidence["core_mean_net_spread"]
    assert evidence["oof_mean_net_increment"] > 0.0
    assert summary.parameters["candidate_share"] > 0.0


def test_residual_sleeve_core_only_path_exactly_reuses_core_model() -> None:
    panel = _panel(candidate_strength=0.0)
    model = _model()

    summary = model.fit(
        panel,
        CORE_FACTORS,
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    assert summary.parameters["candidate_share"] == 0.0
    assert summary.parameters["candidate_factors"] == []
    latest = panel.loc[panel["decision_date"].eq(panel["decision_date"].max())]
    assert model.core_model is not None
    pd.testing.assert_series_equal(
        model.predict(latest),
        model.core_model.predict(latest),
    )


def test_residual_sleeve_ignores_future_rows_and_labels() -> None:
    panel = _panel()
    base = _model()
    base_summary = base.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    future = panel.tail(40).copy()
    future["decision_date"] = "20220103"
    future["label_available_date"] = "20220105"
    future["label_entry_date"] = "20220103"
    future["forward_active_return"] = 1000.0
    future["candidate__z"] = -1000.0
    expanded = pd.concat([panel, future], ignore_index=True)
    changed = _model()
    changed_summary = changed.fit(
        expanded,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    assert changed_summary.parameters["candidate_share"] == pytest.approx(
        base_summary.parameters["candidate_share"]
    )
    assert (
        changed_summary.parameters["sleeve_evidence"]
        == base_summary.parameters["sleeve_evidence"]
    )


def test_residual_sleeve_refit_state_penalty_is_soft() -> None:
    panel = _panel()
    first = _model()
    first.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )
    penalized = _model(
        replace(
            _settings(),
            candidate_change_penalty=0.1,
        )
    )
    penalized.inherit_refit_state(first)
    summary = penalized.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    assert summary.parameters["previous_candidate_share"] == pytest.approx(
        first.candidate_share
    )
    assert 0.0 <= summary.parameters["candidate_share"] <= 1.0


def test_residual_sleeve_registry_parses_nested_core_settings() -> None:
    registry = default_component_registry()
    model = registry.create_model(
        "residual_sleeve_blend",
        {
            "candidate_factors": ["intraday_strength_20"],
            "core": {
                "min_active_factors": 3,
                "min_active_families": 3,
                "max_factor_weight": 1.0,
                "max_family_weight": 1.0,
            },
        },
        {
            "reversal_5d": -1,
            "momentum_60_20": 1,
            "low_vol_20": -1,
            "intraday_strength_20": -1,
        },
    )

    assert model.name == "residual_sleeve_blend"
    assert model.settings.candidate_factors == ("intraday_strength_20",)
    assert model.settings.core.min_active_factors == 3


def test_oof_minimum_t_is_a_hard_gate_separate_from_full_confidence() -> None:
    panel = _panel()
    model = _model(
        replace(
            _settings(),
            min_oof_t=100.0,
            oof_target_t=101.0,
        )
    )

    summary = model.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    evidence = summary.parameters["sleeve_evidence"]
    assert evidence["oof_mean_net_increment"] > 0.0
    assert evidence["oof_evidence_passed"] is False
    assert evidence["oof_confidence"] == 0.0
    assert summary.parameters["candidate_share"] == 0.0


def test_net_stability_weighting_rejects_recently_reversed_candidate() -> None:
    rng = np.random.default_rng(20260831)
    dates = pd.bdate_range("2020-01-02", periods=60)
    instruments = [f"S{position:03d}" for position in range(40)]
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(dates):
        core = rng.normal(size=(len(instruments), len(CORE_FACTORS)))
        stable = rng.normal(size=len(instruments))
        reversing = rng.normal(size=len(instruments))
        reversing_beta = 0.08 if date_number < 45 else -0.12
        label = (
            0.01 * core.mean(axis=1)
            + 0.05 * stable
            + reversing_beta * reversing
            + rng.normal(scale=0.005, size=len(instruments))
        )
        for position, instrument in enumerate(instruments):
            rows.append(
                {
                    "decision_date": date.strftime("%Y%m%d"),
                    "label_available_date": (
                        date + pd.offsets.BDay(2)
                    ).strftime("%Y%m%d"),
                    "label_entry_date": date.strftime("%Y%m%d"),
                    "instrument": instrument,
                    "forward_active_return": label[position],
                    "core_a__z": core[position, 0],
                    "core_b__z": core[position, 1],
                    "core_c__z": core[position, 2],
                    "candidate__z": stable[position],
                    "reversing__z": reversing[position],
                }
            )
    panel = pd.DataFrame(rows)
    settings = replace(
        _settings(),
        candidate_factors=(CANDIDATE, "reversing"),
        candidate_weighting_method="net_stability",
        candidate_recent_segments=2,
        min_candidate_recent_positive_fraction=0.75,
        min_candidate_weight_t=0.0,
        candidate_weight_full_confidence_t=0.0,
    )
    model = ResidualSleeveBlendAlphaModel(
        directions={**DIRECTIONS, "reversing": 1},
        families={**FAMILIES, "reversing": "candidate_family_b"},
        settings=settings,
    )

    summary = model.fit(
        panel,
        (*CORE_FACTORS, CANDIDATE, "reversing"),
        label_column="forward_active_return",
        as_of_date="20210101",
    )

    weighting = summary.parameters["candidate_weighting"]
    assert weighting["factor_statistics"]["reversing"][
        "recent_positive_fraction"
    ] < 0.75
    assert weighting["normalized_weights"]["reversing"] == 0.0
    assert weighting["normalized_weights"][CANDIDATE] == pytest.approx(1.0)
    evidence = summary.parameters["sleeve_evidence"]
    assert evidence["oof_evidence_scope"].endswith(
        "with_nested_candidate_weighting"
    )
    assert all(
        "candidate_inner_weights" in block
        for block in evidence["oof_blocks"]
    )

    changed = panel.copy()
    last_dates = set(sorted(changed["decision_date"].unique())[-15:])
    changed.loc[
        changed["decision_date"].isin(last_dates),
        "forward_active_return",
    ] *= -5.0
    changed_model = ResidualSleeveBlendAlphaModel(
        directions={**DIRECTIONS, "reversing": 1},
        families={**FAMILIES, "reversing": "candidate_family_b"},
        settings=settings,
    )
    changed_summary = changed_model.fit(
        changed,
        (*CORE_FACTORS, CANDIDATE, "reversing"),
        label_column="forward_active_return",
        as_of_date="20210101",
    )
    changed_blocks = changed_summary.parameters["sleeve_evidence"]["oof_blocks"]
    assert changed_blocks[0]["candidate_inner_weights"] == pytest.approx(
        evidence["oof_blocks"][0]["candidate_inner_weights"]
    )


def test_score_churn_budget_reduces_unconstrained_candidate_share() -> None:
    rng = np.random.default_rng(20260901)
    dates = pd.bdate_range("2020-01-02", periods=30)
    instruments = [f"S{position:03d}" for position in range(40)]
    rows: list[dict[str, str]] = []
    core_values: list[float] = []
    candidate_values: list[float] = []
    stable_core = np.linspace(-1.0, 1.0, len(instruments))
    for date in dates:
        random_candidate = rng.normal(size=len(instruments))
        for position, instrument in enumerate(instruments):
            rows.append(
                {
                    "decision_date": date.strftime("%Y%m%d"),
                    "instrument": instrument,
                }
            )
            core_values.append(float(stable_core[position]))
            candidate_values.append(float(random_candidate[position]))
    sample = pd.DataFrame(rows)
    model = _model(
        replace(
            _settings(),
            max_blend_churn_ratio=1.50,
            turnover_budget_grid_size=21,
        )
    )

    budget = model._apply_score_churn_budget(
        sample,
        pd.Series(core_values, index=sample.index),
        pd.Series(candidate_values, index=sample.index),
        unconstrained_share=0.80,
    )

    assert budget["binding"] is True
    assert budget["candidate_share"] < 0.80
    assert (
        budget["budgeted_blend_score_churn"]
        <= budget["allowed_blend_score_churn"] + 1e-12
    )


def test_turnover_budgeted_registry_variant_parses_a32_settings() -> None:
    registry = default_component_registry()
    model = registry.create_model(
        "turnover_budgeted_residual_sleeve",
        {
            "candidate_factors": ["intraday_strength_20"],
            "candidate_weighting_method": "net_stability",
            "min_oof_t": 1.0,
            "oof_target_t": 2.0,
            "max_blend_churn_ratio": 1.5,
            "core": {
                "min_active_factors": 3,
                "min_active_families": 3,
                "max_factor_weight": 1.0,
                "max_family_weight": 1.0,
            },
        },
        {
            "reversal_5d": -1,
            "momentum_60_20": 1,
            "low_vol_20": -1,
            "intraday_strength_20": -1,
        },
    )

    assert model.name == "turnover_budgeted_residual_sleeve"
    assert model.method == "turnover_budgeted_residual_sleeve_blend"
    assert model.settings.candidate_weighting_method == "net_stability"
    assert model.settings.min_oof_t == 1.0
    assert model.settings.oof_target_t == 2.0
    assert model.settings.max_blend_churn_ratio == 1.5
