import numpy as np
import pandas as pd
import pandas.testing as pdt

from csi500_alpha.research.factors import compute_reversal_5d
from csi500_alpha.research.universe import benchmark_weights_asof


def test_benchmark_snapshot_is_available_only_after_snapshot_date() -> None:
    weights = pd.DataFrame(
        {
            "snapshot_date": ["20250131", "20250131", "20250228", "20250228"],
            "instrument": ["A", "B", "A", "C"],
            "weight": [0.5, 0.5, 0.4, 0.6],
        }
    )
    assert benchmark_weights_asof(weights, "20250131").empty
    january = benchmark_weights_asof(weights, "20250203")
    assert set(january.index) == {"A", "B"}
    assert np.isclose(january.sum(), 1.0)


def test_future_price_change_does_not_change_past_factor() -> None:
    dates = pd.bdate_range("2025-01-02", periods=10).strftime("%Y%m%d").tolist()
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "instrument": ["A"] * len(dates),
            "adjusted_close": np.arange(10.0, 20.0),
        }
    )
    baseline = compute_reversal_5d(panel, dates, window=5)
    changed = panel.copy()
    changed.loc[changed["trade_date"] == dates[-1], "adjusted_close"] = 9999.0
    rerun = compute_reversal_5d(changed, dates, window=5)

    cutoff = dates[-2]
    left = baseline[baseline["trade_date"] <= cutoff].reset_index(drop=True)
    right = rerun[rerun["trade_date"] <= cutoff].reset_index(drop=True)
    pdt.assert_frame_equal(left, right)

