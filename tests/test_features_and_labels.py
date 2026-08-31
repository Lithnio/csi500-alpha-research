from dataclasses import replace

import numpy as np
import pandas as pd

from csi500_alpha.config import FeatureSettings
from csi500_alpha.features.builder import build_raw_factor_panel, process_factor_panel
from csi500_alpha.features.catalog import (
    A2_DAILY_FACTOR_CATALOG,
    A2_DAILY_FACTOR_NAMES,
    A3_ALL_DAILY_FACTOR_NAMES,
    A3_DAILY_FACTOR_CATALOG,
    A3_DAILY_FACTOR_NAMES,
    ALL_DAILY_FACTOR_NAMES,
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


def test_a2_daily_catalog_is_versioned_without_changing_frozen_a0() -> None:
    assert len(FACTOR_CATALOG) == 25
    assert len(A2_DAILY_FACTOR_CATALOG) == 7
    assert set(FACTOR_NAMES).isdisjoint(A2_DAILY_FACTOR_NAMES)
    assert len(ALL_DAILY_FACTOR_NAMES) == 32
    for definition in A2_DAILY_FACTOR_CATALOG:
        assert definition.formula_version.startswith("a2-v")
        assert definition.input_fields
        assert definition.availability
        assert definition.missing_rule
        assert definition.formula
        assert definition.hypothesis


def test_a3_daily_catalog_is_opt_in_without_changing_a2() -> None:
    assert len(A3_DAILY_FACTOR_CATALOG) == 6
    assert set(A3_DAILY_FACTOR_NAMES).isdisjoint(ALL_DAILY_FACTOR_NAMES)
    assert len(A3_ALL_DAILY_FACTOR_NAMES) == 38
    for definition in A3_DAILY_FACTOR_CATALOG:
        assert definition.formula_version.startswith("a3-v")
        assert definition.input_fields
        assert definition.availability
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
                    "volume_shares": 1_000_000.0 + 1000.0 * position,
                    "amount_cny": 1_000_000.0 + position,
                    "up_limit": close * 1.10,
                    "down_limit": close * 0.90,
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
            "benchmark_close": 100.0 * np.exp(np.cumsum(index_returns)),
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
    changed_index.loc[
        changed_index["trade_date"] > decision_date,
        "benchmark_close",
    ] *= 100
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


def test_a2_daily_factors_are_point_in_time_and_leave_a0_opt_in() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    decision_date = dates[260]
    baseline = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
    )
    assert set(A2_DAILY_FACTOR_NAMES).isdisjoint(baseline.columns)

    expanded = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=ALL_DAILY_FACTOR_NAMES,
    )
    assert set(A2_DAILY_FACTOR_NAMES).issubset(expanded.columns)
    assert expanded[list(A2_DAILY_FACTOR_NAMES)].notna().all().all()
    assert expanded["limit_up_close_rate_20"].eq(0.0).all()
    assert expanded["failed_limit_up_rate_20"].eq(0.0).all()

    changed_market = market.copy()
    changed_market.loc[changed_market["trade_date"] > decision_date, "adjusted_close"] *= 5
    changed = build_raw_factor_panel(
        market_panel=changed_market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=ALL_DAILY_FACTOR_NAMES,
    )
    pd.testing.assert_frame_equal(
        expanded[["instrument", *A2_DAILY_FACTOR_NAMES]],
        changed[["instrument", *A2_DAILY_FACTOR_NAMES]],
    )


def test_a3_daily_factors_are_point_in_time_and_leave_a2_frozen() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    decision_date = dates[260]
    a2 = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=ALL_DAILY_FACTOR_NAMES,
    )
    assert set(A3_DAILY_FACTOR_NAMES).isdisjoint(a2.columns)

    expanded = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=A3_ALL_DAILY_FACTOR_NAMES,
    )
    assert expanded[list(A3_DAILY_FACTOR_NAMES)].notna().all().all()
    assert expanded["limit_down_close_rate_20"].eq(0.0).all()
    assert expanded["failed_limit_down_rate_20"].eq(0.0).all()
    row = expanded.set_index("instrument").loc["A"]
    assert np.isclose(row["limit_adjusted_momentum_120_20"], 0.100)
    assert row["alpha006_open_volume_corr_10"] < -0.99
    assert np.isclose(row["high_price_momentum_250_20"], 0.230 * 2.0 / 3.0)
    assert np.isclose(row["overnight_intraday_divergence_20"], 0.001)

    changed_market = market.copy()
    future = changed_market["trade_date"] > decision_date
    changed_market.loc[future, "adjusted_open"] *= 7.0
    changed_market.loc[future, "volume_shares"] *= 11.0
    changed = build_raw_factor_panel(
        market_panel=changed_market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=A3_ALL_DAILY_FACTOR_NAMES,
    )
    pd.testing.assert_frame_equal(
        expanded[["instrument", *A3_DAILY_FACTOR_NAMES]],
        changed[["instrument", *A3_DAILY_FACTOR_NAMES]],
    )


def test_limit_adjusted_momentum_excludes_limit_close_and_following_session() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    decision_date = dates[260]
    limit_date = dates[180]
    affected = market["instrument"].eq("A") & market["trade_date"].ge(limit_date)
    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_close",
        "up_limit",
        "down_limit",
    ]
    market.loc[affected, price_columns] *= 1.10
    limit_row = market["instrument"].eq("A") & market["trade_date"].eq(limit_date)
    market.loc[limit_row, "up_limit"] = market.loc[limit_row, "close"]

    result = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=decision_date,
        end_date=decision_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=("limit_adjusted_momentum_120_20",),
    )

    value = result.set_index("instrument").loc[
        "A",
        "limit_adjusted_momentum_120_20",
    ]
    assert np.isclose(value, 0.098)


def test_high_turnover_return_requires_full_60_plus_20_history() -> None:
    dates, market, characteristics, weights, index_bars = _factor_inputs()
    early_date = dates[70]
    mature_date = dates[80]

    early = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=early_date,
        end_date=early_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=("high_turnover_return_20",),
    )
    mature = build_raw_factor_panel(
        market_panel=market,
        daily_characteristics=characteristics,
        benchmark_weights=weights,
        open_dates=dates,
        start_date=mature_date,
        end_date=mature_date,
        rebalance_every=5,
        index_bars=index_bars,
        factor_names=("high_turnover_return_20",),
    )

    assert early["high_turnover_return_20"].isna().all()
    assert mature["high_turnover_return_20"].notna().all()


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
        {"trade_date": dates, "benchmark_open": np.arange(100.0, 110.0)}
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
        (1.0 + expected_stock) / (1.0 + expected_index) - 1.0,
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
    index = pd.DataFrame(
        {"trade_date": dates, "benchmark_open": np.arange(100.0, 108.0)}
    )
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
