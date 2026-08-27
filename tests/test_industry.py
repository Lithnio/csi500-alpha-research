import pandas as pd

from csi500_alpha.data.normalize import normalize_industry_membership
from csi500_alpha.research.industry import (
    industry_asof,
    industry_coverage_by_date,
    overlapping_memberships,
)


def test_industry_intervals_are_point_in_time_and_end_date_is_exclusive() -> None:
    membership = pd.DataFrame(
        {
            "taxonomy": ["SW2021", "SW2021"],
            "instrument": ["A", "A"],
            "l1_code": ["OLD", "NEW"],
            "in_date": ["20200101", "20250105"],
            "out_date": ["20250105", None],
        }
    )
    before = industry_asof(membership, "20250104", transition_date="20211213")
    boundary = industry_asof(membership, "20250105", transition_date="20211213")
    assert before["A"] == "OLD"
    assert boundary["A"] == "NEW"
    assert overlapping_memberships(membership).empty


def test_overlapping_industry_intervals_are_detected() -> None:
    membership = pd.DataFrame(
        {
            "taxonomy": ["SW2021", "SW2021"],
            "instrument": ["A", "A"],
            "l1_code": ["ONE", "TWO"],
            "in_date": ["20200101", "20240101"],
            "out_date": [None, None],
        }
    )
    assert len(overlapping_memberships(membership)) == 1


def test_vendor_exit_events_are_reconciled_into_non_overlapping_intervals() -> None:
    raw = pd.DataFrame(
        {
            "taxonomy": ["SW2021", "SW2021", "SW2021"],
            "l1_code": ["OLD", "MIDDLE", "CURRENT"],
            "l1_name": ["old", "middle", "current"],
            "l2_code": [None, None, None],
            "l2_name": [None, None, None],
            "l3_code": [None, None, None],
            "l3_name": [None, None, None],
            "ts_code": ["A", "A", "A"],
            "name": ["asset", "asset", "asset"],
            "in_date": ["20000101", "20000101", "20000104"],
            "out_date": ["20130701", "20211213", None],
            "is_new": ["N", "N", "Y"],
        }
    )

    membership = normalize_industry_membership([raw])

    assert membership["in_date"].tolist() == ["20000101", "20130701", "20211213"]
    assert membership["out_date"].iloc[:2].tolist() == ["20130701", "20211213"]
    assert pd.isna(membership["out_date"].iloc[-1])
    assert membership.iloc[-1]["raw_in_date"] == "20000104"
    assert overlapping_memberships(membership).empty


def test_industry_coverage_uses_strictly_prior_benchmark_snapshot() -> None:
    membership = pd.DataFrame(
        {
            "taxonomy": ["SW2021"],
            "instrument": ["A"],
            "l1_code": ["ONE"],
            "in_date": ["20200101"],
            "out_date": [None],
        }
    )
    weights = pd.DataFrame(
        {
            "snapshot_date": ["20250102", "20250103", "20250103"],
            "instrument": ["A", "A", "B"],
        }
    )

    coverage = industry_coverage_by_date(
        membership,
        weights,
        ["20250102", "20250103", "20250106"],
        transition_date="20211213",
    )

    assert coverage["decision_date"].tolist() == ["20250103", "20250106"]
    assert coverage["snapshot_date"].tolist() == ["20250102", "20250103"]
    assert coverage["coverage"].tolist() == [1.0, 0.5]
    assert coverage.iloc[-1]["missing_instruments"] == ("B",)
