from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.research.evaluation import _model_weight_evaluation
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.family_models import (
    FamilyAlphaModel,
    FamilyModelSettings,
)

FACTORS = ("q1", "q2", "v1", "v2", "m1", "m2", "l1", "l2")
DIRECTIONS = {factor: 1 for factor in FACTORS}
FAMILIES = {
    "q1": "quality",
    "q2": "quality",
    "v1": "value",
    "v2": "value",
    "m1": "momentum",
    "m2": "momentum",
    "l1": "liquidity",
    "l2": "liquidity",
}


def test_family_equal_weights_respect_factor_and_family_caps() -> None:
    panel = _family_panel()
    model = FamilyAlphaModel(
        method="direction_equal",
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=_settings(),
    )

    fit = model.fit(
        panel,
        FACTORS,
        label_column="forward_active_return",
        as_of_date="20300101",
    )
    prediction = model.predict(panel.iloc[:80])

    weights = fit.parameters["factor_weights"]
    family_weights = fit.parameters["family_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.20 + 1e-12
    assert max(family_weights.values()) <= 0.35 + 1e-12
    assert set(family_weights) == set(FAMILIES.values())
    assert fit.parameters["effective_factor_count"] >= 5.0
    assert prediction.notna().all()


def test_family_robust_ic_closes_a_reversed_family_before_weighting() -> None:
    panel = _family_panel()
    model = FamilyAlphaModel(
        method="robust_ic",
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=_settings(),
    )

    fit = model.fit(
        panel,
        FACTORS,
        label_column="forward_active_return",
        as_of_date="20300101",
    )

    statistics = fit.parameters["method_statistics"]
    assert statistics["liquidity"]["median_rank_ic"] < 0.0
    assert statistics["liquidity"]["robust_ic_preference"] == 0.0
    assert "liquidity" not in fit.parameters["family_weights"]
    assert set(fit.parameters["family_weights"]) == {
        "quality",
        "value",
        "momentum",
    }
    assert max(fit.parameters["factor_weights"].values()) <= 0.20 + 1e-12


def test_family_ridge_is_registered_and_emits_auditable_capped_weights() -> None:
    panel = _family_panel()
    registry = default_component_registry()
    registered = registry.create_model(
        "family_ridge",
        {
            "min_cross_section": 30,
            "min_training_rows": 500,
            "min_training_dates": 20,
            "min_ic_dates": 20,
            "lookback_dates": 30,
            "ridge_alpha": 1.0,
            "min_active_factors": 6,
            "min_active_families": 3,
            "max_factor_weight": 0.20,
            "max_family_weight": 0.35,
        },
        DIRECTIONS,
    )
    assert registered.name == "family_ridge"
    model = FamilyAlphaModel(
        method="ridge",
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=_settings(),
    )

    fit = model.fit(
        panel,
        FACTORS,
        label_column="forward_active_return",
        as_of_date="20300101",
    )

    parameters = json.loads(json.dumps(fit.parameters))
    assert parameters["method"] == "ridge"
    assert sum(parameters["family_weights"].values()) == pytest.approx(1.0)
    assert max(parameters["family_weights"].values()) <= 0.35 + 1e-12
    assert max(parameters["factor_weights"].values()) <= 0.20 + 1e-12


def test_family_caps_fail_closed_when_selected_factors_cannot_sum_to_one() -> None:
    panel = _family_panel()
    names = ("q1", "q2", "v1", "m1", "l1")
    model = FamilyAlphaModel(
        method="direction_equal",
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=FamilyModelSettings(
            min_cross_section=30,
            min_training_rows=500,
            min_training_dates=20,
            min_ic_dates=20,
            lookback_dates=30,
            min_active_factors=5,
            min_active_families=3,
            max_factor_weight=0.20,
            max_family_weight=0.35,
        ),
    )

    with pytest.raises(InsufficientTrainingData, match="cannot provide unit weight"):
        model.fit(
            panel,
            names,
            label_column="forward_active_return",
            as_of_date="20300101",
        )


def test_family_settings_reject_impossible_minimum_capacity() -> None:
    with pytest.raises(ConfigurationError, match="factor cap cannot provide unit weight"):
        FamilyModelSettings(
            min_active_factors=4,
            min_active_families=3,
            max_factor_weight=0.20,
            max_family_weight=0.35,
        ).validate()


def test_family_weights_flow_into_run_level_weight_audit() -> None:
    panel = _family_panel()
    model = FamilyAlphaModel(
        method="direction_equal",
        directions=DIRECTIONS,
        families=FAMILIES,
        settings=_settings(),
    )
    fit = model.fit(
        panel,
        FACTORS,
        label_column="forward_active_return",
        as_of_date="20300101",
    )
    model_fits = pd.DataFrame(
        {
            "fit_date": ["20300101"],
            "model": [model.name],
            "status": ["fitted"],
            "model_parameters": [json.dumps(fit.parameters)],
        }
    )

    summary, history = _model_weight_evaluation(model_fits)

    assert summary["maximum_single_factor_weight"] <= 0.20 + 1e-12
    assert summary["maximum_family_weight"] <= 0.35 + 1e-12
    assert set(history["family"]) == set(FAMILIES.values())
    assert set(history["weight_source"]) == {"factor_weights"}


def _settings() -> FamilyModelSettings:
    return FamilyModelSettings(
        min_cross_section=30,
        min_training_rows=500,
        min_training_dates=20,
        min_ic_dates=20,
        lookback_dates=30,
        ridge_alpha=1.0,
        min_factor_fraction=0.50,
        min_family_fraction=0.50,
        min_active_factors=6,
        min_active_families=3,
        max_factor_weight=0.20,
        max_family_weight=0.35,
    )


def _family_panel() -> pd.DataFrame:
    rng = np.random.default_rng(27)
    dates = pd.bdate_range("2024-01-02", periods=30).strftime("%Y%m%d")
    instruments = [f"S{position:03d}" for position in range(80)]
    rows: list[dict[str, object]] = []
    for date in dates:
        quality = rng.normal(size=len(instruments))
        value = rng.normal(size=len(instruments))
        momentum = rng.normal(size=len(instruments))
        liquidity = rng.normal(size=len(instruments))
        label = (
            0.012 * quality
            + 0.009 * value
            + 0.006 * momentum
            - 0.010 * liquidity
            + rng.normal(0.0, 0.002, len(instruments))
        )
        factor_values = {
            "q1": quality + rng.normal(0.0, 0.05, len(instruments)),
            "q2": quality + rng.normal(0.0, 0.05, len(instruments)),
            "v1": value + rng.normal(0.0, 0.05, len(instruments)),
            "v2": value + rng.normal(0.0, 0.05, len(instruments)),
            "m1": momentum + rng.normal(0.0, 0.05, len(instruments)),
            "m2": momentum + rng.normal(0.0, 0.05, len(instruments)),
            "l1": liquidity + rng.normal(0.0, 0.05, len(instruments)),
            "l2": liquidity + rng.normal(0.0, 0.05, len(instruments)),
        }
        for position, instrument in enumerate(instruments):
            row: dict[str, object] = {
                "decision_date": date,
                "instrument": instrument,
                "label_available_date": date,
                "forward_active_return": float(label[position]),
            }
            row.update(
                {
                    f"{factor}__z": float(values[position])
                    for factor, values in factor_values.items()
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
