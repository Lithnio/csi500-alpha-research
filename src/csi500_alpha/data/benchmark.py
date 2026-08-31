from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import pandas as pd
import yaml

from csi500_alpha.errors import DataQualityError

BENCHMARK_MEMBERSHIP_CONTRACT_VERSION = "csi500-effective-membership-v1"
CSI_ANNOUNCEMENT_URL = "https://www.csindex.com.cn/#/about/newsDetail?id={notice_id}"
TUSHARE_INDEX_WEIGHT_URL = "https://tushare.pro/document/2?doc_id=96"

EVENT_COLUMNS = [
    "index_code",
    "event_id",
    "published_date",
    "effective_from",
    "event_type",
    "action",
    "instrument",
    "confirmation_snapshot_date",
    "source",
]

INTERVAL_COLUMNS = [
    "index_code",
    "instrument",
    "effective_from",
    "effective_to",
    "entry_event_id",
    "entry_event_type",
    "entry_published_date",
    "entry_source",
    "entry_weight_proxy",
    "weight_proxy_source",
    "exit_event_id",
    "exit_event_type",
    "exit_published_date",
    "exit_source",
]


@dataclass(frozen=True)
class BenchmarkEventSpec:
    event_id: str
    confirmation_snapshot_date: str
    published_date: str
    effective_from: str
    event_type: str
    notice_id: int
    additions: tuple[str, ...] = ()
    removals: tuple[str, ...] = ()

    @property
    def source(self) -> str:
        return CSI_ANNOUNCEMENT_URL.format(notice_id=self.notice_id)


def load_csi500_event_registry() -> tuple[BenchmarkEventSpec, ...]:
    """Load the audited CSI 500 event schedule bundled with the package."""

    resource = files("csi500_alpha").joinpath(
        "data/reference/csi500_membership_events.yaml"
    )
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract_version") != (
        BENCHMARK_MEMBERSHIP_CONTRACT_VERSION
    ):
        raise DataQualityError("CSI 500 membership-event registry contract is invalid")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise DataQualityError("CSI 500 membership-event registry is empty")

    specs: list[BenchmarkEventSpec] = []
    for position, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise DataQualityError(
                f"CSI 500 membership event {position} must be a mapping"
            )
        try:
            specs.append(
                BenchmarkEventSpec(
                    event_id=str(raw["event_id"]),
                    confirmation_snapshot_date=str(raw["confirmation_snapshot_date"]),
                    published_date=str(raw["published_date"]),
                    effective_from=str(raw["effective_from"]),
                    event_type=str(raw["event_type"]),
                    notice_id=int(raw["notice_id"]),
                    additions=tuple(str(value) for value in raw.get("additions", ())),
                    removals=tuple(str(value) for value in raw.get("removals", ())),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataQualityError(
                f"CSI 500 membership event {position} is malformed"
            ) from exc

    identifiers = [spec.event_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise DataQualityError("CSI 500 membership event identifiers are not unique")
    return tuple(specs)


def materialize_benchmark_membership(
    weights: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    registry: tuple[BenchmarkEventSpec, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point-in-time membership events and non-overlapping active intervals.

    Tushare supplies one benchmark-weight snapshot per month.  The bundled event
    registry supplies the announcement date and first trading session on which a
    regular or temporary CSI 500 change is active.  Snapshot deltas provide the
    constituent codes and independently confirm the aggregate event result.
    """

    required_weights = {"index_code", "snapshot_date", "instrument", "weight"}
    missing_weights = sorted(required_weights.difference(weights.columns))
    if missing_weights:
        raise DataQualityError(
            f"Benchmark weights are missing membership columns: {missing_weights}"
        )
    if weights.empty:
        raise DataQualityError("Benchmark weights are empty")
    if weights["index_code"].dropna().astype(str).nunique() != 1:
        raise DataQualityError("Benchmark membership requires exactly one index code")

    open_dates = sorted(
        calendar.loc[calendar["is_open"] == 1, "trade_date"].astype(str).unique()
    )
    if not open_dates:
        raise DataQualityError("Benchmark membership requires a nonempty open calendar")
    open_date_set = set(open_dates)

    normalized = weights.copy()
    normalized["snapshot_date"] = normalized["snapshot_date"].astype(str)
    normalized["instrument"] = normalized["instrument"].astype(str)
    normalized["weight"] = pd.to_numeric(normalized["weight"], errors="raise")
    if normalized.duplicated(["snapshot_date", "instrument"]).any():
        raise DataQualityError("Benchmark weights contain duplicate snapshot members")

    index_code = str(normalized["index_code"].iloc[0])
    snapshot_dates = sorted(normalized["snapshot_date"].unique())
    snapshot_members = {
        date: frozenset(
            normalized.loc[normalized["snapshot_date"] == date, "instrument"]
        )
        for date in snapshot_dates
    }
    baseline_snapshot = snapshot_dates[0]
    baseline_effective = _next_open_date(open_dates, baseline_snapshot)
    if baseline_effective is None:
        raise DataQualityError(
            "No trading session exists after the first benchmark-weight snapshot"
        )

    event_rows: list[dict[str, Any]] = []
    baseline_event_id = f"baseline-{baseline_snapshot}"
    for instrument in sorted(snapshot_members[baseline_snapshot]):
        event_rows.append(
            {
                "index_code": index_code,
                "event_id": baseline_event_id,
                "published_date": baseline_snapshot,
                "effective_from": baseline_effective,
                "event_type": "baseline_snapshot",
                "action": "add",
                "instrument": instrument,
                "confirmation_snapshot_date": baseline_snapshot,
                "source": TUSHARE_INDEX_WEIGHT_URL,
            }
        )

    event_specs = registry if registry is not None else load_csi500_event_registry()
    specs_by_snapshot: dict[str, list[BenchmarkEventSpec]] = {}
    for spec in event_specs:
        specs_by_snapshot.setdefault(spec.confirmation_snapshot_date, []).append(spec)

    previous_members = snapshot_members[baseline_snapshot]
    for snapshot_date in snapshot_dates[1:]:
        current_members = snapshot_members[snapshot_date]
        additions = set(current_members.difference(previous_members))
        removals = set(previous_members.difference(current_members))
        previous_members = current_members
        if not additions and not removals:
            continue
        if len(additions) != len(removals):
            raise DataQualityError(
                "Benchmark snapshot transition is asymmetric: "
                f"snapshot={snapshot_date}, additions={len(additions)}, "
                f"removals={len(removals)}"
            )

        specs = sorted(
            specs_by_snapshot.get(snapshot_date, ()),
            key=lambda spec: (spec.effective_from, spec.event_id),
        )
        if not specs:
            raise DataQualityError(
                "Benchmark membership changed without an audited effective-date event: "
                f"snapshot={snapshot_date}, additions={sorted(additions)[:10]}, "
                f"removals={sorted(removals)[:10]}"
            )

        assigned: tuple[tuple[BenchmarkEventSpec, set[str], set[str]], ...]
        if len(specs) == 1 and not specs[0].additions and not specs[0].removals:
            assigned = ((specs[0], additions, removals),)
        else:
            assigned = tuple(
                (spec, set(spec.additions), set(spec.removals)) for spec in specs
            )
            assigned_additions = set().union(*(item[1] for item in assigned))
            assigned_removals = set().union(*(item[2] for item in assigned))
            if assigned_additions != additions or assigned_removals != removals:
                raise DataQualityError(
                    "Audited benchmark events do not reconcile to the monthly snapshot: "
                    f"snapshot={snapshot_date}, "
                    f"missing_additions={sorted(additions - assigned_additions)}, "
                    f"extra_additions={sorted(assigned_additions - additions)}, "
                    f"missing_removals={sorted(removals - assigned_removals)}, "
                    f"extra_removals={sorted(assigned_removals - removals)}"
                )

        for spec, event_additions, event_removals in assigned:
            _validate_event_spec(
                spec,
                snapshot_date=snapshot_date,
                additions=event_additions,
                removals=event_removals,
                open_dates=open_date_set,
            )
            for action, instruments in (
                ("remove", event_removals),
                ("add", event_additions),
            ):
                for instrument in sorted(instruments):
                    event_rows.append(
                        {
                            "index_code": index_code,
                            "event_id": spec.event_id,
                            "published_date": spec.published_date,
                            "effective_from": spec.effective_from,
                            "event_type": spec.event_type,
                            "action": action,
                            "instrument": instrument,
                            "confirmation_snapshot_date": snapshot_date,
                            "source": spec.source,
                        }
                    )

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS).sort_values(
        ["effective_from", "event_id", "action", "instrument"]
    ).reset_index(drop=True)
    intervals = _events_to_intervals(normalized, events)
    return events, intervals


def active_membership_asof(intervals: pd.DataFrame, date: str) -> pd.DataFrame:
    """Return one active interval row per constituent on ``date``."""

    if intervals.empty:
        return intervals.copy()
    effective_to = intervals["effective_to"].fillna("").astype(str)
    active = intervals[
        (intervals["effective_from"].astype(str) <= str(date))
        & (effective_to.eq("") | (effective_to > str(date)))
    ]
    return active.sort_values("instrument").reset_index(drop=True)


def _next_open_date(open_dates: list[str], date: str) -> str | None:
    position = bisect_right(open_dates, str(date))
    return open_dates[position] if position < len(open_dates) else None


def _validate_event_spec(
    spec: BenchmarkEventSpec,
    *,
    snapshot_date: str,
    additions: set[str],
    removals: set[str],
    open_dates: set[str],
) -> None:
    if spec.confirmation_snapshot_date != snapshot_date:
        raise DataQualityError(f"Benchmark event {spec.event_id} has the wrong snapshot")
    if spec.published_date > spec.effective_from:
        raise DataQualityError(
            f"Benchmark event {spec.event_id} was published after it became effective"
        )
    if spec.effective_from not in open_dates:
        raise DataQualityError(
            f"Benchmark event {spec.event_id} is not effective on an open date"
        )
    if spec.effective_from > snapshot_date:
        raise DataQualityError(
            f"Benchmark event {spec.event_id} is confirmed before it becomes effective"
        )
    if not additions or len(additions) != len(removals):
        raise DataQualityError(
            f"Benchmark event {spec.event_id} must have symmetric nonempty changes"
        )
    if additions.intersection(removals):
        raise DataQualityError(
            f"Benchmark event {spec.event_id} adds and removes the same instrument"
        )
    if spec.event_type not in {"regular", "temporary"}:
        raise DataQualityError(
            f"Benchmark event {spec.event_id} has unsupported type {spec.event_type}"
        )


def _events_to_intervals(
    weights: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_rows: dict[str, int] = {}
    proxy_weights: dict[str, float] = {}
    snapshot_dates = sorted(weights["snapshot_date"].astype(str).unique())

    grouped = events.groupby("event_id", sort=False)
    ordered_event_ids = (
        events[["event_id", "effective_from"]]
        .drop_duplicates()
        .sort_values(["effective_from", "event_id"])["event_id"]
    )
    for event_id in ordered_event_ids:
        event = grouped.get_group(event_id)
        metadata = event.iloc[0]
        effective_from = str(metadata["effective_from"])
        proxy_weights = _refresh_proxy_weights(
            weights,
            snapshot_dates,
            active=set(active_rows),
            current=proxy_weights,
            date=effective_from,
        )
        additions = sorted(
            event.loc[event["action"] == "add", "instrument"].astype(str)
        )
        removals = sorted(
            event.loc[event["action"] == "remove", "instrument"].astype(str)
        )

        if str(metadata["event_type"]) == "baseline_snapshot":
            if active_rows or removals:
                raise DataQualityError("Benchmark baseline event is malformed")
            baseline_snapshot = str(metadata["confirmation_snapshot_date"])
            baseline = (
                weights.loc[weights["snapshot_date"].astype(str) == baseline_snapshot]
                .set_index("instrument")["weight"]
                .astype(float)
            )
            proxy_weights = (baseline / baseline.sum()).to_dict()
        else:
            missing_removals = sorted(set(removals).difference(active_rows))
            duplicate_additions = sorted(set(additions).intersection(active_rows))
            if missing_removals or duplicate_additions:
                raise DataQualityError(
                    "Benchmark event is inconsistent with the active membership: "
                    f"event={event_id}, missing_removals={missing_removals}, "
                    f"duplicate_additions={duplicate_additions}"
                )
            removed_weight = sum(proxy_weights[instrument] for instrument in removals)
            for instrument in removals:
                row = rows[active_rows.pop(instrument)]
                row["effective_to"] = effective_from
                row["exit_event_id"] = str(event_id)
                row["exit_event_type"] = str(metadata["event_type"])
                row["exit_published_date"] = str(metadata["published_date"])
                row["exit_source"] = str(metadata["source"])
                proxy_weights.pop(instrument)
            addition_proxy = removed_weight / len(additions)
            for instrument in additions:
                proxy_weights[instrument] = addition_proxy

        for instrument in additions:
            row = {
                "index_code": str(metadata["index_code"]),
                "instrument": instrument,
                "effective_from": effective_from,
                "effective_to": None,
                "entry_event_id": str(event_id),
                "entry_event_type": str(metadata["event_type"]),
                "entry_published_date": str(metadata["published_date"]),
                "entry_source": str(metadata["source"]),
                "entry_weight_proxy": float(proxy_weights[instrument]),
                "weight_proxy_source": (
                    "tushare_snapshot"
                    if str(metadata["event_type"]) == "baseline_snapshot"
                    else "reallocated_outgoing_event_weight"
                ),
                "exit_event_id": None,
                "exit_event_type": None,
                "exit_published_date": None,
                "exit_source": None,
            }
            active_rows[instrument] = len(rows)
            rows.append(row)

        total = sum(proxy_weights.values())
        if total <= 0:
            raise DataQualityError(f"Benchmark event {event_id} produced zero proxy weight")
        proxy_weights = {instrument: value / total for instrument, value in proxy_weights.items()}

    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS).sort_values(
        ["instrument", "effective_from"]
    ).reset_index(drop=True)


def _refresh_proxy_weights(
    weights: pd.DataFrame,
    snapshot_dates: list[str],
    *,
    active: set[str],
    current: dict[str, float],
    date: str,
) -> dict[str, float]:
    if not active:
        return current
    eligible = [snapshot for snapshot in snapshot_dates if snapshot < date]
    if not eligible:
        return current
    snapshot_date = eligible[-1]
    snapshot = (
        weights.loc[weights["snapshot_date"].astype(str) == snapshot_date]
        .set_index("instrument")["weight"]
        .astype(float)
    )
    refreshed: dict[str, float] = {}
    for instrument in active:
        if instrument in snapshot.index:
            refreshed[instrument] = float(snapshot.loc[instrument])
        elif instrument in current:
            refreshed[instrument] = float(current[instrument])
        else:
            raise DataQualityError(
                f"No point-in-time proxy weight exists for active member {instrument}"
            )
    total = sum(refreshed.values())
    if total <= 0:
        raise DataQualityError("Benchmark proxy weights have a nonpositive total")
    return {instrument: value / total for instrument, value in refreshed.items()}
