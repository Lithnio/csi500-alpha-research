import pandas as pd
import pytest

from csi500_alpha.execution.liquidity import build_trailing_adv_snapshots


def test_trailing_adv_snapshot_is_invariant_to_future_volume() -> None:
    panel = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250103", "20250106"],
            "instrument": "A",
            "amount_cny": [100.0, 200.0, 300.0],
        }
    )
    baseline = build_trailing_adv_snapshots(
        panel,
        lookback=2,
        min_observations=2,
    )
    changed = panel.copy()
    changed.loc[changed["trade_date"] == "20250106", "amount_cny"] = 1_000_000.0
    rerun = build_trailing_adv_snapshots(
        changed,
        lookback=2,
        min_observations=2,
    )

    assert baseline["20250103"].adv_cny["A"] == 150.0
    assert baseline["20250103"].observation_count["A"] == 2
    pd.testing.assert_series_equal(
        baseline["20250103"].adv_cny,
        rerun["20250103"].adv_cny,
    )
    assert baseline["20250106"].adv_cny["A"] != rerun["20250106"].adv_cny["A"]


def test_liquidity_snapshot_converts_adv_to_portfolio_weight_cap() -> None:
    panel = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250103"],
            "instrument": "A",
            "amount_cny": [100_000_000.0, 200_000_000.0],
        }
    )
    snapshot = build_trailing_adv_snapshots(
        panel,
        lookback=2,
        min_observations=2,
    )["20250103"]

    caps = snapshot.max_trade_weights(
        portfolio_aum_cny=100_000_000.0,
        max_adv_participation=0.05,
    )

    assert caps["A"] == pytest.approx(0.075)


def test_liquidity_snapshot_rejects_duplicate_market_keys() -> None:
    duplicate = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250102"],
            "instrument": ["A", "A"],
            "amount_cny": [100.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="key is not unique"):
        build_trailing_adv_snapshots(
            duplicate,
            lookback=2,
            min_observations=1,
        )
