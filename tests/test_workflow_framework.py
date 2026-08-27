from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csi500_alpha.config import (
    AppConfig,
    ComponentSettings,
    DateSettings,
    ExperimentSettings,
    FeatureSettings,
    OptimizerSettings,
    ResearchSettings,
    RiskSettings,
    WorkflowSettings,
)
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.pipeline import _assert_stage_artifact_boundaries
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.contracts import FeatureBuildContext
from csi500_alpha.workflow.orchestrator import (
    ResearchWorkflow,
    _industry_exposures,
    _name_history_restrictions,
)
from csi500_alpha.workflow.signals import WalkForwardSignalEngine


def test_name_history_builds_point_in_time_st_buy_restrictions() -> None:
    features = pd.DataFrame(
        {
            "decision_date": ["20250103", "20250103", "20250110", "20250110"],
            "instrument": ["A", "B", "A", "B"],
        }
    )
    history = pd.DataFrame(
        {
            "instrument": ["A", "B"],
            "name": ["*ST甲", "乙"],
            "start_date": ["20250102", "20200101"],
            "end_date": ["20250105", None],
            "announcement_date": ["20250101", "20191231"],
            "is_st": [True, False],
        }
    )

    restrictions = _name_history_restrictions(features, history)

    assert restrictions[["trade_date", "instrument"]].to_dict("records") == [
        {"trade_date": "20250103", "instrument": "A"}
    ]
    assert restrictions["cannot_buy"].all()
    assert not restrictions["cannot_sell"].any()


def test_industry_exposures_include_explicit_missing_bucket() -> None:
    features = pd.DataFrame(
        {
            "decision_date": ["20250103", "20250103"],
            "instrument": ["A", "B"],
            "industry_code": ["I1", None],
        }
    )

    exposures = _industry_exposures(features)

    exposure_columns = [
        column for column in exposures.columns if column not in {"trade_date", "instrument"}
    ]
    assert len(exposure_columns) == 2
    assert exposures[exposure_columns].notna().all().all()
    assert (exposures[exposure_columns].sum(axis=1) == 1.0).all()


def test_walk_forward_ridge_never_uses_unavailable_labels() -> None:
    dates = ["20250102", "20250109", "20250116", "20250123"]
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
                    "forward_active_return": 0.01 * value + position * 0.001,
                }
            )
    panel = pd.DataFrame(rows)
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
        )
        result = engine.run(
            frame,
            ["f"],
            prediction_start=dates[2],
            prediction_end=dates[3],
        )
        return result.signals, result.model_fits

    original_signals, fits = run(panel)
    changed = panel.copy()
    changed.loc[changed["decision_date"] >= dates[2], "forward_active_return"] = 999.0
    changed_signals, _ = run(changed)

    pd.testing.assert_series_equal(original_signals["score"], changed_signals["score"])
    assert (fits["status"] == "fitted").all()
    assert (
        fits["max_label_available_date"].astype(str) < fits["fit_date"].astype(str)
    ).all()


class ToyFactorProvider:
    name = "toy"
    factor_names = ("toy",)
    directions = {"toy": 1}

    def build_raw(self, context: FeatureBuildContext) -> pd.DataFrame:
        eligible = [
            date
            for date in context.open_dates
            if context.start_date <= date <= context.end_date
        ][:: context.rebalance_every]
        instruments = ("A", "B", "C", "D")
        base_scores = (1.0, -0.7, 0.2, -0.1)
        rows = []
        for position, decision_date in enumerate(eligible):
            for rank, (instrument, score) in enumerate(
                zip(instruments, base_scores, strict=True)
            ):
                rows.append(
                    {
                        "decision_date": decision_date,
                        "instrument": instrument,
                        "benchmark_weight": 0.25,
                        "circ_mv_cny": float((rank + 1) * 1_000_000),
                        "pb": 1.0,
                        "industry_code": None,
                        "toy": score * (-1.0 if position % 2 else 1.0),
                    }
                )
        return pd.DataFrame(rows)


def test_custom_factor_provider_runs_through_optimizer_and_event_backtest() -> None:
    dates = pd.bdate_range("2025-01-02", periods=18).strftime("%Y%m%d").tolist()
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    config = replace(
        base,
        dates=DateSettings(
            raw_start=dates[0],
            backtest_start=dates[8],
            end=dates[-1],
        ),
        research=ResearchSettings(
            factor_window=1,
            rebalance_every=2,
            top_n=2,
            initial_cash=1.0,
            linear_cost_bps=5.0,
            stamp_duty_change_date="20230828",
            stamp_duty_before=0.001,
            stamp_duty_after=0.0005,
            price_limit_tolerance=1e-6,
        ),
        risk=RiskSettings(
            lookback=6,
            min_history=4,
            annualization=252,
            missing_annual_volatility=0.8,
            variance_floor=1e-8,
            return_clip=0.2,
        ),
        optimizer=OptimizerSettings(
            enabled=True,
            risk_aversion=1.0,
            risk_horizon_days=2,
            l2_penalty=0.001,
            active_cap=0.20,
            name_cap=0.60,
            turnover_cap=0.50,
            initial_turnover_cap=1.0,
            exposure_cap=0.10,
            solvers=("CLARABEL",),
        ),
        features=FeatureSettings(
            label_horizon=2,
            min_factor_coverage=0.80,
            mad_clip=5.0,
            industry_coverage_threshold=0.95,
            industry_transition_date="20211213",
        ),
        workflow=WorkflowSettings(
            feature_start=dates[2],
            portfolio_start=dates[8],
            refit_every=2,
            factor_names=(),
            feature_provider=ComponentSettings("toy", {}),
            selector=ComponentSettings("all", {}),
            model=ComponentSettings(
                "direction_equal_weight",
                {"min_factor_fraction": 1.0},
            ),
            calibrator=ComponentSettings(
                "robust_cross_section",
                {"target_scale": 0.02, "score_clip": 3.0},
            ),
        ),
        experiment=ExperimentSettings(
            stage="validation",
            protocol_id="toy-stage-boundary-v1",
            train_start=dates[2],
            train_end=dates[7],
            validation_start=dates[8],
            validation_end=dates[12],
            test_start=dates[13],
            test_end=dates[17],
            embargo_days=1,
        ),
    )
    rows = []
    for position, date in enumerate(dates):
        for rank, instrument in enumerate(("A", "B", "C", "D")):
            price = 10.0 + rank * 2.0 + position * (0.05 + rank * 0.01)
            rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                }
            )
    stock_bars = pd.DataFrame(rows)
    keys = stock_bars[["trade_date", "instrument"]]
    tables = {
        "calendar": pd.DataFrame({"trade_date": dates, "is_open": 1}),
        "benchmark_weights": pd.DataFrame(
            {
                "snapshot_date": ["20241231"] * 4,
                "instrument": ["A", "B", "C", "D"],
                "weight": [0.25] * 4,
            }
        ),
        "index_bars": pd.DataFrame(
            {
                "trade_date": dates,
                "open": np.linspace(100.0, 101.7, len(dates)),
                "close": np.linspace(100.0, 101.7, len(dates)),
            }
        ),
        "stock_bars": stock_bars,
        "adjustments": keys.assign(adj_factor=1.0),
        "price_limits": keys.assign(
            up_limit=stock_bars["open"].to_numpy() * 1.10,
            down_limit=stock_bars["open"].to_numpy() * 0.90,
        ),
        "daily_characteristics": pd.DataFrame(),
    }
    registry = default_component_registry()
    registry.register_feature_provider("toy", lambda params: ToyFactorProvider())

    result = ResearchWorkflow(config, registry=registry).run(tables)
    _assert_stage_artifact_boundaries(config, result)

    assert result.factor_names == ("toy",)
    assert result.signals["score"].notna().all()
    assert result.signals["expected_return"].notna().all()
    assert (result.model_fits["status"] == "fitted").all()
    assert (result.calibration_fits["status"] == "fitted").all()
    assert not result.backtest.daily.empty
    assert not result.backtest.optimization.empty
    assert (result.backtest.optimization["active_eligible"] > 0).all()
    assert result.backtest.optimization["status"].isin(
        ["optimal", "optimal_inaccurate"]
    ).all()
    stage_end = dates[12]
    assert result.raw_features["decision_date"].max() <= stage_end
    assert result.processed.features["decision_date"].max() <= stage_end
    assert result.labels["decision_date"].max() <= stage_end
    assert result.labels["label_available_date"].dropna().max() <= stage_end
    assert result.diagnostics.ic_by_date["decision_date"].max() <= stage_end
    assert result.signals["decision_date"].max() <= stage_end
    assert result.evaluation_signals["decision_date"].max() <= stage_end

    revised_tables = {name: frame.copy() for name, frame in tables.items()}
    future_stock = revised_tables["stock_bars"]["trade_date"].astype(str) > stage_end
    revised_tables["stock_bars"].loc[
        future_stock,
        ["open", "high", "low", "close"],
    ] *= 100.0
    future_index = revised_tables["index_bars"]["trade_date"].astype(str) > stage_end
    revised_tables["index_bars"].loc[future_index, ["open", "close"]] *= 100.0

    revised = ResearchWorkflow(config, registry=registry).run(revised_tables)

    pd.testing.assert_frame_equal(result.labels, revised.labels)
    pd.testing.assert_frame_equal(
        result.diagnostics.ic_by_date,
        revised.diagnostics.ic_by_date,
    )
    pd.testing.assert_frame_equal(
        result.evaluation_signals,
        revised.evaluation_signals,
    )
    assert result.backtest.metrics == revised.backtest.metrics

    future_feature = result.raw_features.iloc[[0]].copy()
    future_feature["decision_date"] = dates[13]
    invalid = replace(
        result,
        raw_features=pd.concat(
            [result.raw_features, future_feature],
            ignore_index=True,
        ),
    )
    with pytest.raises(ConfigurationError, match="artifacts cross stage_end"):
        _assert_stage_artifact_boundaries(config, invalid)
