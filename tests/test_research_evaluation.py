from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.research.evaluation import evaluate_research_run


def test_research_evaluation_aggregates_calibration_years_weights_and_execution() -> None:
    signals, labels = _signals_and_labels()
    daily = _daily_returns()
    trades = _trades()
    model_fits = _model_fits()

    result = evaluate_research_run(
        signals=signals,
        labels=labels,
        daily=daily,
        trades=trades,
        optimization=_optimization(),
        model_fits=model_fits,
        calibrator_name="robust_cross_section",
        calibrator_params={"target_scale": 0.01, "score_clip": 3.0},
    )

    calibration = result.summary["calibration"]
    assert calibration["realized_observations"] == len(signals)
    assert calibration["realized_dates"] == 4
    assert calibration["intercept"] == pytest.approx(0.001)
    assert calibration["slope"] == pytest.approx(0.8)
    assert calibration["weighted_r_squared"] == pytest.approx(1.0)
    assert calibration["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert calibration["quintile_monotonicity"] == pytest.approx(1.0)
    assert calibration["top_minus_bottom_realized_return"] > 0
    assert calibration["boundary_fraction"] == 0.0
    assert result.calibration_bins["bin"].tolist() == [1, 2, 3, 4, 5]

    yearly = result.summary["yearly"]
    assert yearly["year_count"] == 2
    assert yearly["positive_active_year_fraction"] == 1.0
    assert yearly["minimum_active_total_return"] == pytest.approx(
        result.yearly_metrics["active_total_return"].min()
    )
    assert yearly["median_active_total_return"] == pytest.approx(
        result.yearly_metrics["active_total_return"].median()
    )
    assert result.yearly_metrics["year"].tolist() == ["2024", "2025"]
    assert result.yearly_metrics["transaction_cost"].sum() == pytest.approx(
        trades[["linear_cost", "stamp_duty", "impact_cost"]].sum().sum()
    )

    weights = result.summary["model_weights"]
    assert weights["audited_fit_count"] == 2
    assert weights["p3_fit_count"] == 2
    assert weights["factor_cap_hits"] == 2
    assert weights["minimum_effective_factor_count"] == pytest.approx(1.923076923)
    assert len(result.factor_weight_history) == 4
    assert set(result.factor_weight_history["factor"]) == {"a", "b"}

    execution = result.summary["execution"]
    assert execution["orders"] == 3
    assert execution["executed_orders"] == 2
    assert execution["partial_orders"] == 1
    assert execution["blocked_orders"] == 1
    assert execution["notional_fill_ratio"] == pytest.approx(0.5)
    assert execution["cost_bps_of_executed_notional"] > 0

    risk = result.summary["risk"]
    assert risk["optimization_attempts"] == 3
    assert risk["factor_model_attempts"] == 3
    assert risk["factor_model_fallback_attempts"] == 1
    assert risk["factor_model_fallback_fraction"] == pytest.approx(1.0 / 3.0)
    assert risk["minimum_beta_observed_fraction"] == pytest.approx(0.91)
    assert risk["maximum_beta_clip_fraction"] == pytest.approx(0.04)


def test_research_evaluation_rejects_duplicate_signal_keys() -> None:
    signals, labels = _signals_and_labels()
    duplicated = pd.concat([signals, signals.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="evaluation signals.*not unique"):
        evaluate_research_run(
            signals=duplicated,
            labels=labels,
            daily=_daily_returns(),
            trades=_trades(),
            model_fits=_model_fits(),
            calibrator_name="robust_cross_section",
            calibrator_params={"target_scale": 0.01, "score_clip": 3.0},
        )


def _signals_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    dates = ("20241205", "20241212", "20250109", "20250116")
    expected = np.linspace(-0.02, 0.02, 10)
    for decision_date in dates:
        for position, value in enumerate(expected):
            instrument = f"S{position:02d}"
            rows.append(
                {
                    "decision_date": decision_date,
                    "instrument": instrument,
                    "expected_return": float(value),
                }
            )
            label_rows.append(
                {
                    "decision_date": decision_date,
                    "instrument": instrument,
                    "forward_active_return": float(0.001 + 0.8 * value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(label_rows)


def _optimization() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_method": [
                "factor_ewma",
                "factor_ewma",
                "factor_ewma_fallback:ledoit_wolf",
            ],
            "beta_method": ["ewma_shrunk_to_one"] * 3,
            "risk_factor_model_fallback": [False, False, True],
            "risk_factor_count": [36, 35, 0],
            "risk_beta_observed_fraction": [0.95, 0.91, 0.94],
            "risk_beta_clip_fraction": [0.02, 0.04, 0.03],
            "risk_factor_covariance_condition_number": [100.0, 120.0, np.nan],
        }
    )


def _daily_returns() -> pd.DataFrame:
    dates = (
        "20241202",
        "20241203",
        "20241204",
        "20241205",
        "20250106",
        "20250107",
        "20250108",
        "20250109",
    )
    benchmark = np.asarray([0.0, 0.001, -0.001, 0.001, 0.0, -0.001, 0.001, 0.001])
    active = np.asarray([0.0, 0.001, 0.0005, 0.0007, 0.0, 0.0008, 0.0004, 0.0006])
    portfolio = (1.0 + benchmark) * (1.0 + active) - 1.0
    nav = np.cumprod(1.0 + portfolio)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    active_nav = nav / benchmark_nav
    return pd.DataFrame(
        {
            "trade_date": dates,
            "nav": nav,
            "benchmark_nav": benchmark_nav,
            "active_nav": active_nav,
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
            "active_return": active,
            "turnover": [0.0, 0.2, 0.0, 0.1, 0.0, 0.3, 0.0, 0.2],
            "transaction_cost": [0.0, 0.0001, 0.0, 0.0001, 0.0, 0.0002, 0.0, 0.0001],
        }
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20241203", "20250107", "20250109"],
            "status": ["filled", "partial", "blocked"],
            "requested_value": [0.2, 0.3, 0.1],
            "gross_value": [0.2, 0.1, 0.0],
            "linear_cost": [0.0001, 0.00005, 0.0],
            "stamp_duty": [0.0, 0.00002, 0.0],
            "impact_cost": [0.00001, 0.00002, 0.0],
        }
    )


def _model_fits() -> pd.DataFrame:
    parameters = []
    for weights, change in (({"a": 0.6, "b": 0.4}, 0.2), ({"a": 0.4, "b": 0.6}, 0.4)):
        parameters.append(
            json.dumps(
                {
                    "method": "empirical_bayes_ic_shrinkage_convex_synthesis",
                    "settings": {"max_factor_weight": 0.6},
                    "weights": weights,
                    "realized_factor_weight_l1_change": change,
                    "factor_statistics": {
                        factor: {
                            "previous_weight": 0.5,
                            "eligible": True,
                            "mean_directed_ic": 0.03,
                            "hac_standard_error": 0.01,
                            "shrinkage_coefficient": 0.9,
                            "posterior_directed_ic": 0.027,
                            "score_churn": 0.1,
                            "exclusion_reasons": [],
                        }
                        for factor in weights
                    },
                }
            )
        )
    return pd.DataFrame(
        {
            "fit_date": ["20241205", "20250109"],
            "model": ["ic_shrinkage", "ic_shrinkage"],
            "status": ["fitted", "fitted"],
            "model_parameters": parameters,
        }
    )
