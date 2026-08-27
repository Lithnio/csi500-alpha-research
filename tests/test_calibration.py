import numpy as np
import pandas as pd

from csi500_alpha.workflow.calibration import WalkForwardReturnCalibrationEngine
from csi500_alpha.workflow.components import default_component_registry


def _signal_and_label_history() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dates = ["20250102", "20250109", "20250116", "20250123", "20250130"]
    signal_rows = []
    label_rows = []
    for position, decision_date in enumerate(dates):
        available = dates[position + 1] if position + 1 < len(dates) else None
        for instrument, score in (("A", -1.0), ("B", 0.0), ("C", 1.0)):
            signal_rows.append(
                {
                    "decision_date": decision_date,
                    "instrument": instrument,
                    "score": score,
                }
            )
            label_rows.append(
                {
                    "decision_date": decision_date,
                    "instrument": instrument,
                    "label_available_date": available,
                    "forward_active_return": score * 0.01 + position * 0.001,
                }
            )
    return pd.DataFrame(signal_rows), pd.DataFrame(label_rows), dates


def test_fixed_calibrator_emits_return_units_without_labels() -> None:
    signals, labels, dates = _signal_and_label_history()
    registry = default_component_registry()
    result = WalkForwardReturnCalibrationEngine(
        calibrator_factory=lambda: registry.create_calibrator(
            "robust_cross_section",
            {"target_scale": 0.02, "score_clip": 3.0},
        ),
        refit_every=2,
    ).run(
        signals,
        labels,
        prediction_start=dates[0],
        prediction_end=dates[-1],
    )

    first = result.signals[result.signals["decision_date"] == dates[0]]
    assert first["expected_return"].notna().all()
    assert np.isclose(first["expected_return"].median(), 0.0)
    assert (result.calibration_fits["status"] == "fitted").all()
    assert result.calibration_fits["training_rows"].eq(0).all()


def test_rolling_calibrator_never_uses_unavailable_labels() -> None:
    signals, labels, dates = _signal_and_label_history()
    registry = default_component_registry()

    def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        result = WalkForwardReturnCalibrationEngine(
            calibrator_factory=lambda: registry.create_calibrator(
                "rolling_ridge",
                {
                    "alpha": 0.0,
                    "min_training_rows": 3,
                    "min_training_dates": 1,
                    "score_clip": 3.0,
                    "label_clip": 0.20,
                    "max_abs_slope": 0.05,
                    "max_abs_expected_return": 0.10,
                },
            ),
            refit_every=1,
        ).run(
            signals,
            frame,
            prediction_start=dates[2],
            prediction_end=dates[-1],
        )
        return result.signals, result.calibration_fits

    original, fits = run(labels)
    changed = labels.copy()
    changed.loc[
        changed["label_available_date"].fillna("").astype(str) >= dates[2],
        "forward_active_return",
    ] = 999.0
    revised, _ = run(changed)

    original_first = original[original["decision_date"] == dates[2]][
        "expected_return"
    ].reset_index(drop=True)
    revised_first = revised[revised["decision_date"] == dates[2]][
        "expected_return"
    ].reset_index(drop=True)
    pd.testing.assert_series_equal(original_first, revised_first)
    fitted = fits[fits["status"] == "fitted"]
    assert not fitted.empty
    assert (
        fitted["max_label_available_date"].astype(str)
        < fitted["fit_date"].astype(str)
    ).all()
