import numpy as np
import pandas as pd
import pytest

from csi500_alpha.data.benchmark import (
    BenchmarkEventSpec,
    active_membership_asof,
    materialize_benchmark_membership,
)
from csi500_alpha.errors import DataQualityError
from csi500_alpha.research.universe import benchmark_weight_state_asof


def _calendar(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": dates, "is_open": 1})


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": "000905.SH",
            "snapshot_date": ["20250131"] * 2 + ["20250228"] * 2,
            "instrument": ["A", "B", "A", "C"],
            "weight": [0.6, 0.4, 0.55, 0.45],
        }
    )


def _replacement() -> BenchmarkEventSpec:
    return BenchmarkEventSpec(
        event_id="temporary-20250217",
        confirmation_snapshot_date="20250228",
        published_date="20250214",
        effective_from="20250217",
        event_type="temporary",
        notice_id=1,
        additions=("C",),
        removals=("B",),
    )


def test_effective_membership_switches_on_announcement_effective_date() -> None:
    events, intervals = materialize_benchmark_membership(
        _weights(),
        _calendar(
            "20250131",
            "20250203",
            "20250214",
            "20250217",
            "20250228",
            "20250303",
        ),
        registry=(_replacement(),),
    )

    assert set(active_membership_asof(intervals, "20250214")["instrument"]) == {
        "A",
        "B",
    }
    assert set(active_membership_asof(intervals, "20250217")["instrument"]) == {
        "A",
        "C",
    }
    replacement = events[events["event_id"] == "temporary-20250217"]
    assert set(replacement["action"]) == {"add", "remove"}


def test_effective_weights_use_frozen_proxy_until_snapshot_is_available() -> None:
    _, intervals = materialize_benchmark_membership(
        _weights(),
        _calendar(
            "20250131",
            "20250203",
            "20250214",
            "20250217",
            "20250228",
            "20250303",
        ),
        registry=(_replacement(),),
    )

    event_day = benchmark_weight_state_asof(_weights(), "20250217", intervals)
    after_snapshot = benchmark_weight_state_asof(_weights(), "20250303", intervals)

    assert set(event_day.weights.index) == {"A", "C"}
    assert event_day.proxy_instruments == ("C",)
    assert event_day.weight_source == "tushare_snapshot_with_event_proxy"
    assert np.isclose(event_day.weights["C"], 0.4)
    assert not after_snapshot.proxy_instruments
    assert after_snapshot.weight_source == "tushare_snapshot"
    assert np.isclose(after_snapshot.weights["C"], 0.45)


def test_changed_snapshot_requires_audited_event() -> None:
    with pytest.raises(DataQualityError, match="without an audited effective-date"):
        materialize_benchmark_membership(
            _weights(),
            _calendar("20250131", "20250203", "20250228", "20250303"),
            registry=(),
        )


def test_multiple_events_can_reconcile_one_month_end_snapshot() -> None:
    weights = pd.DataFrame(
        {
            "index_code": "000905.SH",
            "snapshot_date": ["20250829"] * 3 + ["20250930"] * 3,
            "instrument": ["A", "B", "D", "A", "C", "E"],
            "weight": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2],
        }
    )
    registry = (
        BenchmarkEventSpec(
            event_id="temporary-20250915",
            confirmation_snapshot_date="20250930",
            published_date="20250912",
            effective_from="20250915",
            event_type="temporary",
            notice_id=1,
            additions=("C",),
            removals=("B",),
        ),
        BenchmarkEventSpec(
            event_id="merger-20250929",
            confirmation_snapshot_date="20250930",
            published_date="20250926",
            effective_from="20250929",
            event_type="temporary",
            notice_id=2,
            additions=("E",),
            removals=("D",),
        ),
    )

    _, intervals = materialize_benchmark_membership(
        weights,
        _calendar(
            "20250829",
            "20250901",
            "20250912",
            "20250915",
            "20250926",
            "20250929",
            "20250930",
            "20251008",
        ),
        registry=registry,
    )

    assert set(active_membership_asof(intervals, "20250915")["instrument"]) == {
        "A",
        "C",
        "D",
    }
    assert set(active_membership_asof(intervals, "20250929")["instrument"]) == {
        "A",
        "C",
        "E",
    }
