from __future__ import annotations

import numpy as np
import pandas as pd

from csi500_alpha.config import ExperimentSettings
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.samples import ResearchSamplePolicy
from csi500_alpha.workflow.signals import WalkForwardSignalEngine


def _settings(dates: list[str], stage: str) -> ExperimentSettings:
    return ExperimentSettings(
        stage=stage,
        protocol_id="synthetic-v1",
        train_start=dates[2],
        train_end=dates[10],
        validation_start=dates[13],
        validation_end=dates[18],
        test_start=dates[21],
        test_end=dates[25],
        embargo_days=1,
    )


def _panel(dates: list[str]) -> pd.DataFrame:
    rows = []
    for position, decision_date in enumerate(dates):
        for instrument, value in (("A", -1.0), ("B", 0.0), ("C", 1.0)):
            rows.append(
                {
                    "decision_date": decision_date,
                    "instrument": instrument,
                    "f__z": value,
                    "label_available_date": (
                        dates[position + 1] if position + 1 < len(dates) else None
                    ),
                    "forward_active_return": value * 0.01 + position * 0.0001,
                }
            )
    return pd.DataFrame(rows)


def test_embargo_and_label_maturity_purge_boundary_rows() -> None:
    dates = pd.bdate_range("2025-01-02", periods=30).strftime("%Y%m%d").tolist()
    policy = ResearchSamplePolicy(_settings(dates, "validation"), tuple(dates))

    training = policy.select_training(
        _panel(dates),
        as_of_date=dates[13],
        label_column="forward_active_return",
    )

    assert training["decision_date"].min() == dates[2]
    assert training["decision_date"].max() == dates[10]
    assert training["label_available_date"].max() < dates[12]


def test_frozen_test_never_refits_on_test_labels() -> None:
    dates = pd.bdate_range("2025-01-02", periods=30).strftime("%Y%m%d").tolist()
    panel = _panel(dates)
    policy = ResearchSamplePolicy(_settings(dates, "frozen_test"), tuple(dates))
    registry = default_component_registry()

    def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        engine = WalkForwardSignalEngine(
            selector=registry.create_selector("all", {}),
            model_factory=lambda: registry.create_model(
                "ridge",
                {
                    "alpha": 1.0,
                    "min_training_rows": 3,
                    "min_training_dates": 1,
                    "min_factor_fraction": 1.0,
                },
                {"f": 1},
            ),
            refit_every=1,
            sample_policy=policy,
        )
        result = engine.run(
            frame,
            ["f"],
            prediction_start=policy.signal_start,
            prediction_end=policy.signal_end,
        )
        test_signals = result.signals[result.signals["experiment_phase"] == "test"]
        return test_signals.reset_index(drop=True), result.model_fits

    original, fits = run(panel)
    changed = panel.copy()
    changed.loc[
        changed["decision_date"].between(dates[21], dates[25]),
        "forward_active_return",
    ] = 999.0
    revised, _ = run(changed)

    np.testing.assert_allclose(original["score"], revised["score"])
    test_fits = fits[
        (fits["experiment_phase"] == "test") & (fits["status"] == "fitted")
    ]
    assert len(test_fits) == 1
    assert test_fits.iloc[0]["max_training_decision_date"] <= dates[18]


def test_validation_can_reserve_a_future_holdout_boundary() -> None:
    dates = pd.bdate_range("2025-01-02", periods=30).strftime("%Y%m%d").tolist()
    settings = _settings(dates, "validation")
    settings = ExperimentSettings(
        **{
            **settings.__dict__,
            "test_start": "20260105",
            "test_end": "20260630",
        }
    )

    policy = ResearchSamplePolicy(settings, tuple(dates))

    assert policy.evaluation_end == dates[18]
