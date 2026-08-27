from __future__ import annotations

import pandas as pd


def taxonomy_for_date(decision_date: str, transition_date: str) -> str:
    return "SW2014" if str(decision_date) < str(transition_date) else "SW2021"


def industry_asof(
    membership: pd.DataFrame,
    decision_date: str,
    *,
    transition_date: str,
) -> pd.Series:
    """Return the latest active SW level-one membership known on a decision date."""
    if membership.empty:
        return pd.Series(dtype="object", name="l1_code")
    taxonomy = taxonomy_for_date(decision_date, transition_date)
    out_date = membership["out_date"].fillna("").astype(str)
    active = membership[
        (membership["taxonomy"] == taxonomy)
        & (membership["in_date"].astype(str) <= str(decision_date))
        & ((out_date == "") | (out_date > str(decision_date)))
    ]
    if active.empty:
        return pd.Series(dtype="object", name="l1_code")
    latest = active.sort_values("in_date").drop_duplicates("instrument", keep="last")
    result = latest.set_index("instrument")["l1_code"].sort_index()
    result.name = "l1_code"
    return result


def industry_coverage_by_date(
    membership: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    decision_dates: list[str] | tuple[str, ...] | pd.Series,
    *,
    transition_date: str,
) -> pd.DataFrame:
    """Measure point-in-time industry coverage for the latest prior benchmark snapshot."""

    columns = (
        "decision_date",
        "snapshot_date",
        "taxonomy",
        "members",
        "known_members",
        "missing_members",
        "coverage",
        "missing_instruments",
    )
    if benchmark_weights.empty:
        return pd.DataFrame(columns=columns)

    grouped_members = {
        str(snapshot_date): frozenset(frame["instrument"].astype(str))
        for snapshot_date, frame in benchmark_weights.groupby("snapshot_date", sort=True)
    }
    snapshots = sorted(grouped_members)
    snapshot_position = -1
    rows: list[dict[str, object]] = []
    for decision_date in sorted({str(value) for value in decision_dates}):
        while (
            snapshot_position + 1 < len(snapshots)
            and snapshots[snapshot_position + 1] < decision_date
        ):
            snapshot_position += 1
        if snapshot_position < 0:
            continue
        snapshot_date = snapshots[snapshot_position]
        members = grouped_members[snapshot_date]
        if not members:
            continue
        industries = industry_asof(
            membership,
            decision_date,
            transition_date=transition_date,
        )
        known = members.intersection(set(industries.index.astype(str)))
        missing = tuple(sorted(members.difference(known)))
        rows.append(
            {
                "decision_date": decision_date,
                "snapshot_date": snapshot_date,
                "taxonomy": taxonomy_for_date(decision_date, transition_date),
                "members": len(members),
                "known_members": len(known),
                "missing_members": len(missing),
                "coverage": len(known) / len(members),
                "missing_instruments": missing,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def overlapping_memberships(membership: pd.DataFrame) -> pd.DataFrame:
    """Identify overlapping level-one intervals within a taxonomy and instrument."""
    records: list[dict[str, str]] = []
    if membership.empty:
        return pd.DataFrame(columns=["taxonomy", "instrument", "in_date", "previous_out"])
    grouped = membership.sort_values("in_date").groupby(
        ["taxonomy", "instrument"],
        sort=False,
    )
    for (taxonomy, instrument), frame in grouped:
        previous_end = ""
        for row in frame.itertuples(index=False):
            in_date = str(row.in_date)
            if previous_end and in_date < previous_end:
                records.append(
                    {
                        "taxonomy": str(taxonomy),
                        "instrument": str(instrument),
                        "in_date": in_date,
                        "previous_out": previous_end,
                    }
                )
            out_date = "" if pd.isna(row.out_date) else str(row.out_date)
            current_end = out_date or "99999999"
            previous_end = max(previous_end, current_end)
    return pd.DataFrame(records)
