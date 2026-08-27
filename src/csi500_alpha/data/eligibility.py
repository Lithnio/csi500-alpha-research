from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.client import TushareClient
from csi500_alpha.data.downloader import DatePartition, annual_partitions
from csi500_alpha.data.normalize import normalize_name_history, normalize_suspensions
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import DataQualityError
from csi500_alpha.utils import utc_now

ELIGIBILITY_CONTRACT_VERSION = "csi500-tushare-eligibility-v3"


@dataclass(frozen=True)
class EligibilityDownloadPlan:
    instruments: int
    partitions: tuple[DatePartition, ...]
    estimated_requests: int
    theoretical_minimum_minutes: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruments": self.instruments,
            "partitions": [partition.to_dict() for partition in self.partitions],
            "estimated_requests": self.estimated_requests,
            "theoretical_minimum_minutes": self.theoretical_minimum_minutes,
            "outputs": ["name_history", "resumptions"],
        }


@dataclass(frozen=True)
class EligibilityDownloadSummary:
    paths: dict[str, Path]
    rows: dict[str, int]
    cache_hits: int
    network_requests: int
    quality_path: Path
    progress_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": {name: str(path) for name, path in self.paths.items()},
            "rows": self.rows,
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "quality_path": str(self.quality_path),
            "progress_path": str(self.progress_path),
        }


def build_eligibility_download_plan(config: AppConfig) -> EligibilityDownloadPlan:
    instruments = _benchmark_instruments(config)
    partitions = annual_partitions(config.dates.raw_start, config.dates.end)
    requests = len(instruments) + len(partitions)
    return EligibilityDownloadPlan(
        instruments=len(instruments),
        partitions=partitions,
        estimated_requests=requests,
        theoretical_minimum_minutes=(
            requests * config.source.effective_min_request_interval_seconds / 60.0
        ),
    )


class EligibilityDownloader:
    """Resumable supplemental downloader for historical names and resumptions."""

    def __init__(self, config: AppConfig, client: TushareClient) -> None:
        self.config = config
        self.client = client
        self.cache_hits = 0
        self.network_requests = 0

    def _fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
        result = self.client.fetch(*args, **kwargs)  # type: ignore[arg-type]
        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.network_requests += 1
        return result.frame

    def run(
        self,
        *,
        force: bool = False,
        refresh_names_from: str | None = None,
    ) -> EligibilityDownloadSummary:
        self.cache_hits = 0
        self.network_requests = 0
        plan = build_eligibility_download_plan(self.config)
        instruments = _benchmark_instruments(self.config)
        effective_refresh_names_from = (
            refresh_names_from or self.config.download.eligibility_refresh_start
        )
        refresh_name_instruments = (
            frozenset(
                _benchmark_instruments_for_refresh(
                    self.config,
                    start_date=effective_refresh_names_from,
                )
            )
            if effective_refresh_names_from is not None
            else frozenset()
        )
        name_cache_tag = (
            f"{self.config.download.reference_cache_tag}.eligibility"
            if self.config.download.reference_cache_tag is not None
            else (
                f"{self.config.paths.dataset}-{self.config.dates.end}.eligibility"
                if effective_refresh_names_from is not None
                else None
            )
        )
        progress_path = self.config.paths.quality_root / "eligibility-download-progress.json"
        quality_path = self.config.paths.quality_root / "eligibility-data-quality.json"
        progress: dict[str, Any] = {
            "status": "running",
            "started_at": utc_now(),
            "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "plan": plan.to_dict(),
            "name_history": {"completed_instruments": 0, "total_instruments": len(instruments)},
            "resumptions": {"completed_partitions": 0, "total_partitions": len(plan.partitions)},
            "request_policy": {
                "force_all": force,
                "configured_refresh_names_from": (
                    self.config.download.eligibility_refresh_start
                ),
                "refresh_names_from": effective_refresh_names_from,
                "explicit_name_refresh": refresh_names_from is not None,
                "refresh_name_instruments": len(refresh_name_instruments),
                "name_cache_tag": name_cache_tag,
            },
        }
        write_json_atomic(progress, progress_path)

        name_frames: list[pd.DataFrame] = []
        resumption_frames: list[pd.DataFrame] = []
        try:
            for position, instrument in enumerate(instruments, start=1):
                name_frames.append(
                    self._fetch(
                        "namechange",
                        params={"ts_code": instrument},
                        fields=(
                            "ts_code",
                            "name",
                            "start_date",
                            "end_date",
                            "ann_date",
                            "change_reason",
                        ),
                        force=(
                            force
                            or (
                                refresh_names_from is not None
                                and instrument in refresh_name_instruments
                            )
                        ),
                        cache_tag=(
                            name_cache_tag
                            if instrument in refresh_name_instruments
                            else None
                        ),
                    )
                )
                if position % 50 == 0 or position == len(instruments):
                    progress["name_history"]["completed_instruments"] = position
                    progress["cache_hits"] = self.cache_hits
                    progress["network_requests"] = self.network_requests
                    write_json_atomic(progress, progress_path)

            for position, partition in enumerate(plan.partitions, start=1):
                resumption_frames.append(
                    self._fetch(
                        "suspend_d",
                        params={
                            "start_date": partition.start_date,
                            "end_date": partition.end_date,
                            "suspend_type": "R",
                        },
                        fields=("ts_code", "trade_date", "suspend_timing", "suspend_type"),
                        force=force,
                    )
                )
                progress["resumptions"]["completed_partitions"] = position
                progress["cache_hits"] = self.cache_hits
                progress["network_requests"] = self.network_requests
                write_json_atomic(progress, progress_path)
        except Exception as exc:
            progress.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "cache_hits": self.cache_hits,
                    "network_requests": self.network_requests,
                }
            )
            write_json_atomic(progress, progress_path)
            raise

        instrument_set = set(instruments)
        source_name_history = normalize_name_history(name_frames)
        name_normalization = dict(
            source_name_history.attrs.get("normalization", {})
        )
        name_history = restrict_name_history_to_window(
            source_name_history,
            start_date=self.config.dates.raw_start,
            end_date=self.config.dates.end,
        )
        name_history = name_history[
            name_history["instrument"].astype(str).isin(instrument_set)
        ].reset_index(drop=True)
        resumptions = normalize_suspensions(resumption_frames)
        resumptions = resumptions[
            resumptions["instrument"].astype(str).isin(instrument_set)
            & resumptions["trade_date"].astype(str).between(
                self.config.dates.raw_start,
                self.config.dates.end,
            )
        ].reset_index(drop=True)

        validation = validate_eligibility_data(
            name_history,
            resumptions,
            instruments,
            start_date=self.config.dates.raw_start,
            end_date=self.config.dates.end,
        )
        paths = {
            "name_history": self.config.paths.silver_root / "name_history.parquet",
            "resumptions": self.config.paths.silver_root / "resumptions.parquet",
        }
        fingerprints = {
            "name_history": write_parquet_atomic(name_history, paths["name_history"]),
            "resumptions": write_parquet_atomic(resumptions, paths["resumptions"]),
        }
        quality = {
            "status": "success" if validation["passed"] else "failed",
            "created_at": utc_now(),
            "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "date_range": {
                "start_date": self.config.dates.raw_start,
                "end_date": self.config.dates.end,
            },
            "rows": {"name_history": len(name_history), "resumptions": len(resumptions)},
            "normalization": {
                **name_normalization,
                "source_name_history_rows": len(source_name_history),
                "research_window_name_history_rows": len(name_history),
                "excluded_name_history_rows": len(source_name_history) - len(name_history),
                "window_policy": "intersect_and_clip_to_configured_date_range",
            },
            "validation": validation,
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "request_policy": {
                "force_all": force,
                "configured_refresh_names_from": (
                    self.config.download.eligibility_refresh_start
                ),
                "refresh_names_from": effective_refresh_names_from,
                "explicit_name_refresh": refresh_names_from is not None,
                "refresh_name_instruments": len(refresh_name_instruments),
                "name_cache_tag": name_cache_tag,
            },
            "paths": {name: str(path) for name, path in paths.items()},
            "fingerprints": fingerprints,
        }
        write_json_atomic(quality, quality_path)
        progress.update(
            {
                "status": quality["status"],
                "finished_at": utc_now(),
                "quality_path": str(quality_path),
                "cache_hits": self.cache_hits,
                "network_requests": self.network_requests,
            }
        )
        write_json_atomic(progress, progress_path)
        if not validation["passed"]:
            raise DataQualityError(
                "Eligibility supplemental data failed quality checks; "
                f"see {quality_path}"
            )
        return EligibilityDownloadSummary(
            paths=paths,
            rows={"name_history": len(name_history), "resumptions": len(resumptions)},
            cache_hits=self.cache_hits,
            network_requests=self.network_requests,
            quality_path=quality_path,
            progress_path=progress_path,
        )


def _benchmark_instruments(config: AppConfig) -> tuple[str, ...]:
    path = config.paths.silver_root / "benchmark_weights.parquet"
    if not path.exists():
        raise DataQualityError(
            "Eligibility download requires an existing benchmark_weights.parquet snapshot"
        )
    weights = pd.read_parquet(path, columns=["instrument"])
    instruments = tuple(sorted(weights["instrument"].dropna().astype(str).unique()))
    if not instruments:
        raise DataQualityError("Eligibility download found no benchmark instruments")
    return instruments


def _benchmark_instruments_for_refresh(
    config: AppConfig,
    *,
    start_date: str,
) -> tuple[str, ...]:
    """Refresh names for the last pre-window snapshot and later constituents."""

    if (
        len(start_date) != 8
        or not start_date.isdigit()
        or not config.dates.raw_start <= start_date <= config.dates.end
    ):
        raise DataQualityError(
            "refresh_names_from must be YYYYMMDD within the configured data range"
        )
    path = config.paths.silver_root / "benchmark_weights.parquet"
    weights = pd.read_parquet(path, columns=["snapshot_date", "instrument"])
    snapshot_dates = weights["snapshot_date"].fillna("").astype(str)
    prior_dates = snapshot_dates[snapshot_dates < start_date]
    relevant = snapshot_dates >= start_date
    if not prior_dates.empty:
        relevant |= snapshot_dates.eq(str(prior_dates.max()))
    instruments = tuple(
        sorted(weights.loc[relevant, "instrument"].dropna().astype(str).unique())
    )
    if not instruments:
        raise DataQualityError(
            "No benchmark instruments overlap the requested name-refresh window"
        )
    return instruments


def validate_eligibility_data(
    name_history: pd.DataFrame,
    resumptions: pd.DataFrame,
    instruments: tuple[str, ...],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    name_duplicates = int(
        name_history.duplicated(["instrument", "start_date", "end_date", "name"]).sum()
    )
    invalid_name_intervals = int(
        (
            name_history["start_date"].isna()
            | (
                name_history["end_date"].notna()
                & (name_history["end_date"].astype(str) < name_history["start_date"].astype(str))
            )
        ).sum()
    )
    covered = set(name_history["instrument"].astype(str))
    missing = sorted(set(instruments).difference(covered))
    name_coverage = len(covered.intersection(instruments)) / len(instruments)
    resumption_duplicates = int(
        resumptions.duplicated(["trade_date", "instrument", "suspend_type"]).sum()
    )
    invalid_resumption_types = int(
        (~resumptions["suspend_type"].astype(str).eq("R")).sum()
    )
    interval_overlaps = _overlapping_name_intervals(name_history)
    announcements = name_history["announcement_date"].fillna("").astype(str)
    missing_st_announcements = int(
        (name_history["is_st"].fillna(False).astype(bool) & announcements.eq("")).sum()
    )
    outside_window = 0
    outside_resumptions = 0
    if start_date is not None and end_date is not None:
        starts = name_history["start_date"].fillna("").astype(str)
        ends = name_history["end_date"].fillna("").astype(str)
        outside_window = int(
            (
                starts.lt(start_date)
                | starts.gt(end_date)
                | ends.eq("")
                | ends.lt(start_date)
                | ends.gt(end_date)
            ).sum()
        )
        outside_resumptions = int(
            (~resumptions["trade_date"].astype(str).between(start_date, end_date)).sum()
        )
    passed = (
        name_coverage >= 0.99
        and name_duplicates == 0
        and invalid_name_intervals == 0
        and interval_overlaps.empty
        and missing_st_announcements == 0
        and outside_window == 0
        and resumption_duplicates == 0
        and invalid_resumption_types == 0
        and outside_resumptions == 0
    )
    return {
        "passed": passed,
        "name_history": {
            "benchmark_instruments": len(instruments),
            "covered_instruments": len(covered.intersection(instruments)),
            "coverage": name_coverage,
            "missing_instruments": len(missing),
            "missing_sample": missing[:20],
            "duplicates": name_duplicates,
            "invalid_intervals": invalid_name_intervals,
            "overlapping_intervals": int(len(interval_overlaps)),
            "overlap_sample": interval_overlaps.head(20).to_dict("records"),
            "outside_configured_window": outside_window,
            "missing_st_announcements": missing_st_announcements,
            "st_intervals": int(name_history["is_st"].sum()),
        },
        "resumptions": {
            "rows": len(resumptions),
            "duplicates": resumption_duplicates,
            "invalid_types": invalid_resumption_types,
            "outside_configured_window": outside_resumptions,
        },
    }


def restrict_name_history_to_window(
    name_history: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Create a deterministic research-window dimension from full name history."""

    if start_date > end_date:
        raise ValueError("Name-history window start must not be after end")
    if name_history.empty:
        return name_history.copy()
    result = name_history.copy()
    starts = result["start_date"].fillna("").astype(str)
    ends = result["end_date"].fillna("").astype(str)
    invalid = starts.eq("") | (ends.ne("") & ends.lt(starts))
    intersects = starts.le(end_date) & (ends.eq("") | ends.ge(start_date))
    result = result.loc[invalid | intersects].copy()
    starts = result["start_date"].fillna("").astype(str)
    ends = result["end_date"].fillna("").astype(str)
    valid = starts.ne("") & (ends.eq("") | ends.ge(starts))
    result.loc[valid & starts.lt(start_date), "start_date"] = start_date
    result.loc[valid & (ends.eq("") | ends.gt(end_date)), "end_date"] = end_date
    keys = ["instrument", "start_date", "end_date", "name"]
    return (
        result.drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _overlapping_name_intervals(name_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "instrument",
        "start_date",
        "end_date",
        "previous_start_date",
        "previous_end_date",
    ]
    rows: list[dict[str, str]] = []
    ordered = name_history.sort_values(["instrument", "start_date", "end_date"])
    for instrument, group in ordered.groupby("instrument", sort=False):
        previous_start = ""
        previous_end = ""
        for row in group.itertuples(index=False):
            start = "" if pd.isna(row.start_date) else str(row.start_date)
            end = "99999999" if pd.isna(row.end_date) else str(row.end_date)
            if previous_end and start and start <= previous_end:
                rows.append(
                    {
                        "instrument": str(instrument),
                        "start_date": start,
                        "end_date": end,
                        "previous_start_date": previous_start,
                        "previous_end_date": previous_end,
                    }
                )
            if not previous_end or end > previous_end:
                previous_start = start
                previous_end = end
    return pd.DataFrame(rows, columns=columns)
