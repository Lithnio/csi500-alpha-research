from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.ic_shrinkage import (
    ICShrinkageAlphaModel,
    ICShrinkageSettings,
)
from csi500_alpha.workflow.signals import WalkForwardSignalEngine

DIRECTIONS = {
    "stable": 1,
    "diversifier": 1,
    "costly": 1,
    "bad": 1,
}


def test_ic_shrinkage_fit_enforces_constraints_and_audits_inputs() -> None:
    panel = _synthetic_panel()
    settings = _settings()
    model = ICShrinkageAlphaModel(directions=DIRECTIONS, settings=settings)

    summary = model.fit(
        panel,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date="20250101",
    )

    weights = model.factor_weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.0
    assert max(weights.values()) <= settings.max_factor_weight + 1e-7
    assert weights["bad"] == 0.0
    parameters = summary.parameters
    assert parameters["method"] == "empirical_bayes_ic_shrinkage_convex_synthesis"
    assert parameters["previous_weight_source"] == "equal_weight_initialization"
    assert parameters["optimization"]["status"] in {"optimal", "optimal_inaccurate"}
    assert parameters["optimization"]["constraint_residuals"]["sum_to_one"] < 1e-10
    statistics = parameters["factor_statistics"]
    assert "nonpositive_directed_ic" in statistics["bad"]["exclusion_reasons"]
    for factor in ("stable", "diversifier", "costly"):
        assert statistics[factor]["eligible"]
        assert 0 < statistics[factor]["shrinkage_coefficient"] <= 1
        assert statistics[factor]["ic_dates"] >= settings.min_ic_dates
        assert statistics[factor]["churn_dates"] >= settings.min_churn_dates

    scores = model.predict(panel.iloc[:80])
    assert scores.index.equals(panel.iloc[:80].index)
    assert scores.notna().all()


def test_cost_penalty_reduces_weight_on_high_churn_factor() -> None:
    panel = _synthetic_panel()
    base = replace(
        _settings(),
        shrinkage_enabled=False,
        correlation_penalty=0.0,
        cost_penalty=0.0,
        weight_turnover_penalty=0.0,
        max_factor_weight=0.80,
    )
    without_cost = ICShrinkageAlphaModel(directions=DIRECTIONS, settings=base)
    with_cost = ICShrinkageAlphaModel(
        directions=DIRECTIONS,
        settings=replace(base, cost_penalty=2.0),
    )

    without_summary = without_cost.fit(
        panel,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date="20250101",
    )
    with_summary = with_cost.fit(
        panel,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date="20250101",
    )

    statistics = without_summary.parameters["factor_statistics"]
    assert statistics["costly"]["score_churn"] > statistics["stable"]["score_churn"]
    assert statistics["costly"]["score_churn"] > statistics["diversifier"]["score_churn"]
    assert with_cost.factor_weights["costly"] < without_cost.factor_weights["costly"]
    assert with_summary.parameters["optimization"]["components"]["cost_penalty"] > 0


def test_correlation_and_refit_penalties_change_weights_in_expected_direction() -> None:
    panel = _synthetic_panel()
    rng = np.random.default_rng(7)
    panel["stable_copy__z"] = panel["stable__z"] + rng.normal(
        scale=0.01,
        size=len(panel),
    )
    panel["forward_active_return"] = (
        panel["stable__z"]
        + 0.25 * panel["diversifier__z"]
        + rng.normal(scale=0.05, size=len(panel))
    )
    duplicate_directions = {"stable": 1, "stable_copy": 1, "diversifier": 1}
    unpenalized_settings = replace(
        _settings(),
        shrinkage_enabled=False,
        correlation_penalty=0.0,
        cost_penalty=0.0,
        weight_turnover_penalty=0.0,
        max_factor_weight=0.80,
    )
    unpenalized = ICShrinkageAlphaModel(
        directions=duplicate_directions,
        settings=unpenalized_settings,
    )
    correlation_aware = ICShrinkageAlphaModel(
        directions=duplicate_directions,
        settings=replace(unpenalized_settings, correlation_penalty=1.0),
    )
    for model in (unpenalized, correlation_aware):
        model.fit(
            panel,
            tuple(duplicate_directions),
            label_column="forward_active_return",
            as_of_date="20250101",
        )

    unpenalized_duplicate_weight = (
        unpenalized.factor_weights["stable"]
        + unpenalized.factor_weights["stable_copy"]
    )
    correlation_duplicate_weight = (
        correlation_aware.factor_weights["stable"]
        + correlation_aware.factor_weights["stable_copy"]
    )
    assert correlation_duplicate_weight < unpenalized_duplicate_weight - 0.20

    regime_directions = {"stable": 1, "diversifier": 1, "costly": 1}
    first_regime = _synthetic_panel()
    second_regime = first_regime.copy()
    first_regime["forward_active_return"] = (
        first_regime["stable__z"]
        + 0.25 * first_regime["diversifier__z"]
        + 0.15 * first_regime["costly__z"]
    )
    second_regime["forward_active_return"] = (
        second_regime["diversifier__z"]
        + 0.25 * second_regime["stable__z"]
        + 0.15 * second_regime["costly__z"]
    )
    previous = ICShrinkageAlphaModel(
        directions=regime_directions,
        settings=unpenalized_settings,
    )
    previous.fit(
        first_regime,
        tuple(regime_directions),
        label_column="forward_active_return",
        as_of_date="20250101",
    )
    free_refit = ICShrinkageAlphaModel(
        directions=regime_directions,
        settings=unpenalized_settings,
    )
    free_refit.fit(
        second_regime,
        tuple(regime_directions),
        label_column="forward_active_return",
        as_of_date="20250101",
    )
    stable_refit = ICShrinkageAlphaModel(
        directions=regime_directions,
        settings=replace(unpenalized_settings, weight_turnover_penalty=1.0),
    )
    stable_refit.inherit_refit_state(previous)
    stable_refit.fit(
        second_regime,
        tuple(regime_directions),
        label_column="forward_active_return",
        as_of_date="20250101",
    )

    assert _weight_distance(stable_refit, previous) < _weight_distance(
        free_refit,
        previous,
    )


def test_ic_shrinkage_ignores_future_rows_and_inherits_refit_state() -> None:
    panel = _synthetic_panel()
    as_of_date = "20250101"
    original = ICShrinkageAlphaModel(directions=DIRECTIONS, settings=_settings())
    original.fit(
        panel,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date=as_of_date,
    )
    future = panel.iloc[:80].copy()
    future["decision_date"] = "20250102"
    future["label_available_date"] = "20250103"
    future["forward_active_return"] = 999.0
    contaminated = pd.concat([panel, future], ignore_index=True)
    repeated = ICShrinkageAlphaModel(directions=DIRECTIONS, settings=_settings())
    repeated.fit(
        contaminated,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date=as_of_date,
    )

    assert repeated.factor_weights == pytest.approx(original.factor_weights)

    inherited = ICShrinkageAlphaModel(directions=DIRECTIONS, settings=_settings())
    inherited.inherit_refit_state(original)
    summary = inherited.fit(
        panel,
        tuple(DIRECTIONS),
        label_column="forward_active_return",
        as_of_date=as_of_date,
    )

    assert summary.parameters["previous_weight_source"] == "prior_model"
    assert summary.parameters["realized_factor_weight_l1_change"] < 1e-5


def test_walk_forward_engine_passes_only_previous_model_state() -> None:
    panel = _synthetic_panel()
    dates = sorted(panel["decision_date"].unique())
    registry = default_component_registry()
    settings = _settings()
    engine = WalkForwardSignalEngine(
        selector=registry.create_selector("all", {}),
        model_factory=lambda: registry.create_model(
            "ic_shrinkage",
            {
                "min_cross_section": settings.min_cross_section,
                "min_ic_dates": settings.min_ic_dates,
                "min_churn_dates": settings.min_churn_dates,
                "lookback_dates": settings.lookback_dates,
                "max_factor_weight": settings.max_factor_weight,
                "min_active_factors": settings.min_active_factors,
                "solvers": ["CLARABEL"],
            },
            DIRECTIONS,
        ),
        refit_every=1,
    )

    result = engine.run(
        panel,
        tuple(DIRECTIONS),
        prediction_start=dates[-2],
        prediction_end=dates[-1],
    )

    assert result.model_fits["status"].tolist() == ["fitted", "fitted"]
    parameters = [json.loads(value) for value in result.model_fits["model_parameters"]]
    assert parameters[0]["previous_weight_source"] == "equal_weight_initialization"
    assert parameters[1]["previous_weight_source"] == "prior_model"
    assert result.signals["score"].notna().all()


def test_failed_refit_keeps_last_successful_ic_model() -> None:
    panel = _synthetic_panel()
    dates = sorted(panel["decision_date"].unique())
    calls = 0

    def model_factory() -> ICShrinkageAlphaModel:
        nonlocal calls
        settings = (
            _settings()
            if calls == 0
            else replace(_settings(), min_mean_directed_ic=0.99)
        )
        calls += 1
        return ICShrinkageAlphaModel(directions=DIRECTIONS, settings=settings)

    engine = WalkForwardSignalEngine(
        selector=default_component_registry().create_selector("all", {}),
        model_factory=model_factory,
        refit_every=1,
    )
    result = engine.run(
        panel,
        tuple(DIRECTIONS),
        prediction_start=dates[-2],
        prediction_end=dates[-1],
    )

    assert result.model_fits["status"].tolist() == [
        "fitted",
        "insufficient_training_data",
    ]
    assert result.model_fits["action"].tolist() == [
        "replace_model",
        "keep_previous_model",
    ]
    assert result.signals["model_fit_date"].nunique() == 1
    assert result.signals["model_fit_date"].iloc[0] == dates[-2]
    assert result.signals["score"].notna().all()


def test_ic_shrinkage_registry_rejects_ambiguous_boolean() -> None:
    registry = default_component_registry()

    with pytest.raises(ConfigurationError, match="shrinkage_enabled must be a boolean"):
        registry.create_model(
            "ic_shrinkage",
            {"shrinkage_enabled": "false"},
            DIRECTIONS,
        )


def _settings() -> ICShrinkageSettings:
    return ICShrinkageSettings(
        min_cross_section=60,
        min_ic_dates=24,
        min_churn_dates=20,
        lookback_dates=40,
        correlation_penalty=0.01,
        cost_penalty=0.01,
        weight_turnover_penalty=0.01,
        max_factor_weight=0.70,
        min_active_factors=3,
        solvers=("CLARABEL",),
    )


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260826)
    instruments = np.asarray([f"S{position:03d}" for position in range(80)])
    quality = rng.normal(size=len(instruments))
    diversifier = rng.normal(size=len(instruments))
    diversifier -= quality * float(diversifier @ quality) / float(quality @ quality)
    dates = pd.bdate_range("2024-09-02", periods=42).strftime("%Y%m%d")
    rows: list[dict[str, object]] = []
    for decision_date in dates:
        dynamic = rng.normal(size=len(instruments))
        stable_score = quality + rng.normal(scale=0.08, size=len(instruments))
        diversifier_score = diversifier + rng.normal(scale=0.08, size=len(instruments))
        costly_score = dynamic + rng.normal(scale=0.03, size=len(instruments))
        label = (
            0.45 * quality
            + 0.45 * diversifier
            + 0.85 * dynamic
            + rng.normal(scale=0.20, size=len(instruments))
        )
        bad_score = -label + rng.normal(scale=0.05, size=len(instruments))
        for position, instrument in enumerate(instruments):
            rows.append(
                {
                    "decision_date": str(decision_date),
                    "instrument": str(instrument),
                    "label_available_date": str(decision_date),
                    "forward_active_return": float(label[position]),
                    "stable__z": float(stable_score[position]),
                    "diversifier__z": float(diversifier_score[position]),
                    "costly__z": float(costly_score[position]),
                    "bad__z": float(bad_score[position]),
                }
            )
    return pd.DataFrame(rows)


def _weight_distance(
    left: ICShrinkageAlphaModel,
    right: ICShrinkageAlphaModel,
) -> float:
    names = set(left.factor_weights) | set(right.factor_weights)
    return float(
        sum(
            abs(left.factor_weights.get(name, 0.0) - right.factor_weights.get(name, 0.0))
            for name in names
        )
    )
