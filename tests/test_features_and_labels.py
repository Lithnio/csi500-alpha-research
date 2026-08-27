from dataclasses import replace

import numpy as np
import pandas as pd

from csi500_alpha.config import FeatureSettings
from csi500_alpha.features.builder import build_raw_factor_panel, process_factor_panel
from csi500_alpha.features.catalog import (
    DIRECTIONS,
    FACTOR_CATALOG,
    FACTOR_NAMES,
    FAMILIES,
)
from csi500_alpha.features.labels import build_forward_labels


def _feature_settings() -> FeatureSettings:
    return FeatureSettings(
        label_horizon=5,
        min_factor_coverage=0.8,
        mad_clip=5.0,
        industry_coverage_threshold=0.95,
        industry_transition_date="20211213",
    )


def test_factor_catalog_v2_has_complete_auditable_metadata() -> None:
    assert len(FACTOR_CATALOG) == 25
    assert len(set(FACTOR_NAMES)) == len(FACTOR_NAMES)
    assert set(DIRECTIONS.values()) == {-1, 1}
    assert set(FAMILIES) == set(FACTOR_NAMES)
    assert "abnormal_turnover_reversal" in FACTOR_NAMES
    for definition in FACTOR_CATALOG:
        assert definition.formula_version
        assert definition.input_fields
        assert definition.availability
        assert definition.family
        assert definition.missing_rule
        assert definition.formula
        assert definition.hypothesis


def _factor_inputs() -> tuple[
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.bdate_range("2024-01-02", periods=280).strftime("%Y%m%d").tolist()
    instruments = ("A", "B", "C")
    slopes = {"A": 0.001, "B": 0.002, "C": -0.001}
    market_rows = []
    characteristic_rows = []
    for position, date in enumerate(dates):
        for instrument in instruments:
            close = 100.0 * np.exp(slopes[instrument] * position)
            market_rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adjusted_open": close,
                    "adjusted_close": close,
                    "amount_cny": 1_000_000.0 + position,
                }
            )
            characteristic_rows.append(
                {
                    "trade_date": date,
                    "instrument": instrument,
                    "turnover_rate": 1.0 + 0.001 * position,
                    "turnover_rate_f": 1.2 + 0.001 * position,
                    "circ_mv_cny": {"A": 1e9, "B": 2e9, "C": 3e9}[instrument],
                    "total_mv_cny": {"A": 2e9, "B": 4e9, "C": 6e9}[instrument],
                    "pb": {"A": 1.0, "B": 2.0, "C": 3.0}[instrument],
                }
            )
    weights = pd.DataFrame(
        {
            "snapshot_date": ["20240101"] * 3,
            "instrument": list(instruments),
            "weight": [1 / 3] * 3,
        }
    )
    index_returns = 0.001 + 0.0005 * np.sin(np.arange(len(dates)) / 7.0)
    index_bars = pd.DataFrame(
        {
            "trade_date": dates,
            "close": 100.0 * np.exp(np.cumsum(index_returns)),
        }
    )
    return (
        dates,
        pd.DataFrame(market_rows),
        pd.DataFrame(characteristic_rows),
        weights,
        index_bars,
    )


def test_raw_factor_formulas_and_future_data_invariance() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    decision_date = dates[260]
    original = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
    )
    row = original.set_index("instrument").loc["A"]
    assert np.isclose(row["reversal_5d"], -0.005)
    assert np.isclose(row["reversal_1d"], -0.001)
    assert np.isclose(row["reversal_10d"], -0.010)
    assert np.isclose(row["reversal_20d"], -0.020)
    assert np.isclose(row["momentum_20_5"], 0.015)
    assert np.isclose(row["momentum_60_20"], 0.040)
    assert np.isclose(row["momentum_120_20"], 0.100)
    assert np.isclose(row["momentum_250_20"], 0.230)
    assert np.isclose(row["free_float_ratio"], np.log(0.5))

    changed_market = market.copy()
    changed_market.loc[changed_market["trade_date"] > decision_date, "adjusted_close"] *= 50
    changed_characteristics = characteristics.copy()
    changed_characteristics.loc[
        changed_characteristics["trade_date"] > decision_date,
        "circ_mv_cny",
    ] *= 100
    changed_index = index_bars.copy()
    changed_index.loc[changed_index["trade_date"] > decision_date, "close"] *= 100
    revised = build_raw_factor_panel(
        market_panel=changed_market,
        daily_characteristics=changed_characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=changed_index,
    )
    pd.testing.assert_frame_equal(
        original[["instrument", *FACTOR_NAMES]],
        revised[["instrument", *FACTOR_NAMES]],
    )


def test_labels_use_next_open_and_horizon_end_open() -> None:
    dates = pd.bdate_range("2025-01-02", periods=10).strftime("%Y%m%d").tolist()
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "instrument": "A",
            "adjusted_open": np.arange(10.0, 20.0),
        }
    )
    index = pd.DataFrame(
        {"trade_date": dates, "open": np.arange(100.0, 110.0)}
    )
    features = pd.DataFrame({"decision_date": [dates[1]], "instrument": ["A"]})
    labels = build_forward_labels(
        features=features,
        market_panel=market,
        index_bars=index,
        open_dates=dates,
        horizon=5,
    )
    expected_stock = 17.0 / 12.0 - 1.0
    expected_index = 107.0 / 102.0 - 1.0
    assert labels.loc[0, "label_entry_date"] == dates[2]
    assert labels.loc[0, "label_end_date"] == dates[7]
    assert np.isclose(labels.loc[0, "forward_stock_return"], expected_stock)
    assert np.isclose(
        labels.loc[0, "forward_active_return"],
        expected_stock - expected_index,
    )
    assert bool(labels.loc[0, "entry_tradeable"])
    assert bool(labels.loc[0, "exit_tradeable"])
    assert labels.loc[0, "label_status"] == "valid"
    assert bool(labels.loc[0, "label_valid"])


def test_zero_range_session_has_neutral_close_location() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    decision_date = dates[260]
    zero_range = market["instrument"].eq("A")
    market.loc[zero_range, "high"] = market.loc[zero_range, "close"]
    market.loc[zero_range, "low"] = market.loc[zero_range, "close"]

    features = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
    )

    assert features.set_index("instrument").loc["A", "close_location_20"] == 0.0


def test_label_flags_opening_suspension_but_not_later_intraday_pause() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8).strftime("%Y%m%d").tolist()
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "instrument": "A",
            "adjusted_open": np.arange(10.0, 18.0),
        }
    )
    index = pd.DataFrame({"trade_date": dates, "open": np.arange(100.0, 108.0)})
    features = pd.DataFrame({"decision_date": [dates[0]], "instrument": ["A"]})
    suspensions = pd.DataFrame(
        {
            "trade_date": [dates[1], dates[3]],
            "instrument": ["A", "A"],
            "suspend_timing": ["10:07-10:17", "09:30-10:00"],
            "suspend_type": ["S", "S"],
        }
    )

    labels = build_forward_labels(
        features=features,
        market_panel=market,
        index_bars=index,
        open_dates=dates,
        horizon=2,
        suspensions=suspensions,
    )

    assert bool(labels.loc[0, "entry_tradeable"])
    assert not bool(labels.loc[0, "exit_tradeable"])
    assert labels.loc[0, "label_status"] == "exit_suspended_at_open"
    assert not bool(labels.loc[0, "label_valid"])
    assert np.isnan(labels.loc[0, "forward_stock_return"])


def test_industry_missing_bucket_keeps_stable_neutralization() -> None:
    instruments = [f"S{number:02d}" for number in range(10)]
    rows = []
    for position, instrument in enumerate(instruments, start=1):
        row = {
            "decision_date": "20250110",
            "instrument": instrument,
            "benchmark_weight": 0.1,
            "circ_mv_cny": float(np.exp(position)),
            "pb": 1.0,
            "industry_code": "I1" if position <= 5 else "I2",
        }
        if position == 10:
            row["industry_code"] = None
        row.update({factor: float(position + 0.1 * ((-1) ** position)) for factor in FACTOR_NAMES})
        rows.append(row)

    processed = process_factor_panel(
        pd.DataFrame(rows),
        replace(_feature_settings(), industry_coverage_threshold=0.90),
    )

    assert processed.quality["industry_neutralized"].all()
    assert np.isclose(processed.quality["industry_coverage"].iloc[0], 0.9)
    assert processed.features.filter(like="__z").notna().all().all()


def test_cross_sectional_processing_is_date_local_and_size_neutral() -> None:
    instruments = [f"S{number:02d}" for number in range(10)]
    rows = []
    for date, shift in (("20250110", 0.0), ("20250117", 1000.0)):
        for position, instrument in enumerate(instruments, start=1):
            row = {
                "decision_date": date,
                "instrument": instrument,
                "benchmark_weight": 0.1,
                "circ_mv_cny": float(np.exp(position)),
                "pb": 1.0,
                "industry_code": None,
            }
            value = position + 0.1 * ((-1) ** position) + shift
            row.update({factor: float(value) for factor in FACTOR_NAMES})
            rows.append(row)
    raw = pd.DataFrame(rows)
    both = process_factor_panel(raw, _feature_settings()).features
    first_only = process_factor_panel(
        raw[raw["decision_date"] == "20250110"].copy(),
        _feature_settings(),
    ).features
    columns = [f"{factor}__z" for factor in FACTOR_NAMES]
    pd.testing.assert_frame_equal(
        both.loc[both["decision_date"] == "20250110", columns].reset_index(drop=True),
        first_only[columns].reset_index(drop=True),
    )
    neutralized = both[both["decision_date"] == "20250110"]
    correlation = neutralized["reversal_5d__z"].corr(
        np.log(neutralized["circ_mv_cny"])
    )
    assert abs(correlation) < 1e-8
