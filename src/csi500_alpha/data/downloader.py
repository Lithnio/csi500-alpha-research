from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.benchmark import (
    BENCHMARK_MEMBERSHIP_CONTRACT_VERSION,
    materialize_benchmark_membership,
)
from csi500_alpha.data.client import TushareClient
from csi500_alpha.data.normalize import (
    normalize_adjustments,
    normalize_calendar,
    normalize_daily_basic,
    normalize_index_bars,
    normalize_industry_classification,
    normalize_industry_membership,
    normalize_instrument_master,
    normalize_limits,
    normalize_stock_bars,
    normalize_suspensions,
    normalize_weights,
)
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import DataQualityError
from csi500_alpha.research.industry import industry_coverage_by_date
from csi500_alpha.utils import canonical_json, iter_months, sha256_file, sha256_text, utc_now

LOGGER = logging.getLogger(__name__)

PARTITION_CONTRACT_VERSION = "csi500-tushare-silver-v3"
SNAPSHOT_CONTRACT_VERSION = "csi500-tushare-silver-v5"
RECOMMENDED_FREE_BYTES = 5 * 1024**3
INDUSTRY_SUPPLEMENT_ESTIMATED_INSTRUMENTS = 250


@dataclass(frozen=True)
class DatePartition:
    partition_id: str
    start_date: str
    end_date: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadPlan:
    start_date: str
    end_date: str
    partitions: tuple[DatePartition, ...]
    daily_apis: tuple[str, ...]
    estimated_business_days: int
    estimated_requests: int
    effective_min_request_interval_seconds: float
    theoretical_minimum_minutes: float
    free_bytes: int
    recommended_free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "partitions": [partition.to_dict() for partition in self.partitions],
            "daily_apis": list(self.daily_apis),
            "estimated_business_days": self.estimated_business_days,
            "estimated_requests": self.estimated_requests,
            "effective_min_request_interval_seconds": (self.effective_min_request_interval_seconds),
            "theoretical_minimum_minutes": self.theoretical_minimum_minutes,
            "free_bytes": self.free_bytes,
            "recommended_free_bytes": self.recommended_free_bytes,
            "disk_preflight_passed": self.free_bytes >= self.recommended_free_bytes,
        }


@dataclass(frozen=True)
class PartitionSummary:
    partition_id: str
    start_date: str
    end_date: str
    status: str
    rows: dict[str, int]
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "rows": self.rows,
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True)
class DownloadSummary:
    paths: dict[str, Path]
    rows: dict[str, int]
    cache_hits: int
    network_requests: int
    partitions: tuple[PartitionSummary, ...]
    progress_path: Path
    snapshot_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "partitions": [partition.to_dict() for partition in self.partitions],
            "paths": {name: str(path) for name, path in self.paths.items()},
            "progress_path": str(self.progress_path),
            "snapshot_path": str(self.snapshot_path),
        }


@dataclass(frozen=True)
class _ReferenceData:
    tables: dict[str, pd.DataFrame]
    open_dates: tuple[str, ...]
    instruments: frozenset[str]


def annual_partitions(start_date: str, end_date: str) -> tuple[DatePartition, ...]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    partitions: list[DatePartition] = []
    for year in range(start.year, end.year + 1):
        partition_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        partition_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        partitions.append(
            DatePartition(
                partition_id=str(year),
                start_date=partition_start.strftime("%Y%m%d"),
                end_date=partition_end.strftime("%Y%m%d"),
            )
        )
    return tuple(partitions)


def build_download_plan(config: AppConfig) -> DownloadPlan:
    daily_apis = ["daily", "adj_factor", "stk_limit"]
    if config.download.include_daily_basic:
        daily_apis.append("daily_basic")
    if config.download.include_suspensions:
        daily_apis.append("suspend_d")

    estimated_business_days = len(pd.bdate_range(config.dates.raw_start, config.dates.end))
    monthly_weight_requests = sum(1 for _ in iter_months(config.dates.raw_start, config.dates.end))
    reference_requests = 3 + monthly_weight_requests
    if config.download.include_instrument_master:
        reference_requests += 4
    if config.download.include_industry:
        # Two membership calls (current and historical) for roughly 32 L1
        # industries per taxonomy, plus the classification call itself.
        reference_requests += len(config.download.industry_taxonomies) * 65
        if config.download.supplement_industry_by_instrument:
            # The actual second-stage request count is coverage-driven. This planning
            # allowance assumes 250 instruments need two narrow calls each.
            reference_requests += 2 * INDUSTRY_SUPPLEMENT_ESTIMATED_INSTRUMENTS
    estimated_requests = reference_requests + estimated_business_days * len(daily_apis)
    effective_interval = config.source.effective_min_request_interval_seconds
    free_bytes = shutil.disk_usage(config.paths.data_root.parent).free
    return DownloadPlan(
        start_date=config.dates.raw_start,
        end_date=config.dates.end,
        partitions=annual_partitions(config.dates.raw_start, config.dates.end),
        daily_apis=tuple(daily_apis),
        estimated_business_days=estimated_business_days,
        estimated_requests=estimated_requests,
        effective_min_request_interval_seconds=effective_interval,
        theoretical_minimum_minutes=estimated_requests * effective_interval / 60.0,
        free_bytes=free_bytes,
        recommended_free_bytes=RECOMMENDED_FREE_BYTES,
    )


class SmokeDownloader:
    """Resumable Tushare-to-Silver downloader with annual materialization gates."""

    def __init__(self, config: AppConfig, client: TushareClient) -> None:
        self.config = config
        self.client = client
        self.cache_hits = 0
        self.network_requests = 0
        self.response_limit_events: list[dict[str, Any]] = []
        self.industry_supplement_audit: dict[str, Any] = {}

    def _fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
        result = self.client.fetch(*args, **kwargs)  # type: ignore[arg-type]
        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.network_requests += 1
        return result.frame

    @staticmethod
    def _assert_not_at_limit(api_name: str, frame: pd.DataFrame, limit: int) -> None:
        if len(frame) >= limit:
            raise DataQualityError(
                f"Tushare response may be truncated: api={api_name}, rows={len(frame)}, "
                f"documented_limit={limit}"
            )

    def run(
        self,
        *,
        force: bool = False,
        refresh_reference: bool = False,
    ) -> DownloadSummary:
        self.cache_hits = 0
        self.network_requests = 0
        self.response_limit_events = []
        self.industry_supplement_audit = {}
        plan = build_download_plan(self.config)
        if plan.free_bytes < plan.recommended_free_bytes:
            raise DataQualityError(
                "Data download disk preflight failed: "
                f"free_bytes={plan.free_bytes}, "
                f"recommended_free_bytes={plan.recommended_free_bytes}"
            )

        progress_path = self.config.paths.quality_root / "download-progress.json"
        snapshot_path = self.config.paths.quality_root / "snapshot-manifest.json"
        progress = self._initial_progress(plan)
        progress["request_policy"] = {
            "force_all": force,
            "refresh_mutable_reference": refresh_reference,
            "reference_cache_tag": self.config.download.reference_cache_tag,
        }
        write_json_atomic(progress, progress_path)

        try:
            reference = self._fetch_reference_data(
                force=force,
                refresh_mutable=refresh_reference,
            )
            progress["reference"] = {
                "status": "success",
                "rows": {name: int(len(frame)) for name, frame in reference.tables.items()},
                "instrument_count": len(reference.instruments),
                "open_dates": len(reference.open_dates),
                "industry_supplement": self.industry_supplement_audit,
            }
            progress["actual_open_dates"] = len(reference.open_dates)
            write_json_atomic(progress, progress_path)
        except Exception as exc:
            progress["status"] = "failed"
            progress["reference"] = self._failure_payload(exc)
            progress["finished_at"] = utc_now()
            write_json_atomic(progress, progress_path)
            raise

        instrument_hash = sha256_text("\n".join(sorted(reference.instruments)))
        partition_summaries: list[PartitionSummary] = []
        for partition in plan.partitions:
            entry = progress["partitions"][partition.partition_id]
            entry["status"] = "running"
            entry["started_at"] = utc_now()
            write_json_atomic(progress, progress_path)
            contract_hash = self._partition_contract_hash(partition, instrument_hash)
            try:
                reused = (
                    None
                    if force
                    else self._load_valid_partition(
                        partition,
                        contract_hash,
                        reference.open_dates,
                    )
                )
                if reused is not None:
                    summary = reused
                else:
                    summary = self._download_partition(
                        partition,
                        reference.open_dates,
                        reference.instruments,
                        contract_hash,
                        force=force,
                    )
                partition_summaries.append(summary)
                progress["partitions"][partition.partition_id] = {
                    **summary.to_dict(),
                    "finished_at": utc_now(),
                }
                write_json_atomic(progress, progress_path)
            except Exception as exc:
                progress["status"] = "failed"
                progress["partitions"][partition.partition_id] = {
                    **partition.to_dict(),
                    **self._failure_payload(exc),
                    "finished_at": utc_now(),
                }
                progress["finished_at"] = utc_now()
                write_json_atomic(progress, progress_path)
                raise

        paths, rows, fingerprints = self._materialize_snapshot(
            reference.tables,
            plan.partitions,
        )
        snapshot_contract_hash = sha256_text(
            canonical_json(
                {
                    "contract_version": SNAPSHOT_CONTRACT_VERSION,
                    "start_date": plan.start_date,
                    "end_date": plan.end_date,
                    "instrument_hash": instrument_hash,
                    "source": self._source_contract(),
                    "download": asdict(self.config.download),
                }
            )
        )
        snapshot = {
            "status": "success",
            "created_at": utc_now(),
            "contract_version": SNAPSHOT_CONTRACT_VERSION,
            "contract_hash": snapshot_contract_hash,
            "dataset": self.config.paths.dataset,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "instrument_count": len(reference.instruments),
            "instrument_hash": instrument_hash,
            "source": self._source_contract(),
            "download": asdict(self.config.download),
            "rows": rows,
            "fingerprints": fingerprints,
            "paths": {name: str(path) for name, path in paths.items()},
            "partitions": [summary.to_dict() for summary in partition_summaries],
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "request_policy": {
                "force_all": force,
                "refresh_mutable_reference": refresh_reference,
                "reference_cache_tag": self.config.download.reference_cache_tag,
            },
            "response_limit_events": self.response_limit_events,
            "industry_supplement": self.industry_supplement_audit,
            "request_manifest": str(self.config.paths.data_root / "manifest.sqlite"),
        }
        write_json_atomic(snapshot, snapshot_path)
        progress["status"] = "success"
        progress["finished_at"] = utc_now()
        progress["cache_hits"] = self.cache_hits
        progress["network_requests"] = self.network_requests
        progress["snapshot_path"] = str(snapshot_path)
        write_json_atomic(progress, progress_path)
        return DownloadSummary(
            paths=paths,
            rows=rows,
            cache_hits=self.cache_hits,
            network_requests=self.network_requests,
            partitions=tuple(partition_summaries),
            progress_path=progress_path,
            snapshot_path=snapshot_path,
        )

    def _fetch_reference_data(
        self,
        *,
        force: bool,
        refresh_mutable: bool,
    ) -> _ReferenceData:
        cfg = self.config
        reference_cache_tag = cfg.download.reference_cache_tag
        calendar_raw = self._fetch(
            "trade_cal",
            params={
                "exchange": cfg.source.exchange,
                "start_date": cfg.dates.raw_start,
                "end_date": cfg.dates.end,
            },
            fields=("exchange", "cal_date", "is_open", "pretrade_date"),
            force=force,
        )
        calendar = normalize_calendar(calendar_raw)
        open_dates = tuple(calendar.loc[calendar["is_open"] == 1, "trade_date"].astype(str))
        if not open_dates:
            raise DataQualityError("No open trading dates were returned for the download range")

        weight_frames: list[pd.DataFrame] = []
        for start_date, end_date in iter_months(cfg.dates.raw_start, cfg.dates.end):
            query_start = max(start_date, cfg.dates.raw_start)
            query_end = min(end_date, cfg.dates.end)
            weight_frames.append(
                self._fetch(
                    "index_weight",
                    params={
                        "index_code": cfg.source.index_code,
                        "start_date": query_start,
                        "end_date": query_end,
                    },
                    fields=("index_code", "con_code", "trade_date", "weight"),
                    force=force,
                )
            )
        weights = normalize_weights(weight_frames)
        instruments = frozenset(weights["instrument"].astype(str))
        if not instruments:
            raise DataQualityError("No benchmark constituents were downloaded")
        membership_events, membership_intervals = materialize_benchmark_membership(
            weights,
            calendar,
        )

        index_raw = self._fetch(
            "index_daily",
            params={
                "ts_code": cfg.source.index_code,
                "start_date": cfg.dates.raw_start,
                "end_date": cfg.dates.end,
            },
            fields=(
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
            ),
            force=force,
        )
        self._assert_not_at_limit("index_daily", index_raw, 8000)
        total_return_index_raw = self._fetch(
            "index_daily",
            params={
                "ts_code": cfg.source.total_return_index_code,
                "start_date": cfg.dates.raw_start,
                "end_date": cfg.dates.end,
            },
            fields=(
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
            ),
            force=force,
        )
        self._assert_not_at_limit(
            "index_daily[total_return]",
            total_return_index_raw,
            8000,
        )
        index_bars = normalize_index_bars(index_raw, total_return_index_raw)

        tables: dict[str, pd.DataFrame] = {
            "calendar": calendar,
            "benchmark_weights": weights,
            "benchmark_membership_events": membership_events,
            "benchmark_membership_intervals": membership_intervals,
            "index_bars": index_bars,
        }
        if cfg.download.include_instrument_master:
            master_frames: list[pd.DataFrame] = []
            for list_status in ("L", "D", "P", "G"):
                frame = self._fetch(
                    "stock_basic",
                    params={"exchange": "", "list_status": list_status},
                    fields=(
                        "ts_code",
                        "symbol",
                        "name",
                        "market",
                        "exchange",
                        "list_status",
                        "list_date",
                        "delist_date",
                    ),
                    force=force or refresh_mutable,
                    cache_tag=reference_cache_tag,
                )
                self._assert_not_at_limit("stock_basic", frame, 6000)
                master_frames.append(frame)
            instrument_master = normalize_instrument_master(master_frames)
            tables["instrument_master"] = instrument_master[
                instrument_master["instrument"].isin(instruments)
            ].reset_index(drop=True)

        if cfg.download.include_industry:
            classifications, memberships = self._fetch_industry(
                instruments,
                weights,
                open_dates,
                force=force or refresh_mutable,
                cache_tag=reference_cache_tag,
            )
            tables["industry_classification"] = classifications
            tables["industry_membership"] = memberships

        return _ReferenceData(
            tables=tables,
            open_dates=open_dates,
            instruments=instruments,
        )

    def _fetch_industry(
        self,
        instruments: frozenset[str],
        benchmark_weights: pd.DataFrame,
        open_dates: tuple[str, ...],
        *,
        force: bool,
        cache_tag: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        classification_frames: list[pd.DataFrame] = []
        membership_frames: list[pd.DataFrame] = []
        classification_codes: dict[str, frozenset[str]] = {}
        member_fields = (
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "l3_code",
            "l3_name",
            "ts_code",
            "name",
            "in_date",
            "out_date",
            "is_new",
        )
        for taxonomy in self.config.download.industry_taxonomies:
            classification = self._fetch(
                "index_classify",
                params={"level": "L1", "src": taxonomy},
                fields=(
                    "index_code",
                    "industry_name",
                    "parent_code",
                    "level",
                    "industry_code",
                    "is_pub",
                    "src",
                ),
                force=force,
                cache_tag=cache_tag,
            ).assign(taxonomy=taxonomy)
            if classification.empty:
                raise DataQualityError(
                    f"No L1 industry classifications were returned for {taxonomy}"
                )
            classification_frames.append(classification)
            l1_codes = frozenset(classification["index_code"].astype(str).unique())
            classification_codes[taxonomy] = l1_codes
            for l1_code in sorted(l1_codes):
                for is_new in ("Y", "N"):
                    membership = self._fetch(
                        "index_member_all",
                        params={"l1_code": l1_code, "is_new": is_new},
                        fields=member_fields,
                        force=force,
                        cache_tag=cache_tag,
                    )
                    self._assert_not_at_limit("index_member_all", membership, 2000)
                    membership_frames.append(membership.assign(taxonomy=taxonomy))
        classifications = normalize_industry_classification(classification_frames)
        memberships = normalize_industry_membership(membership_frames)
        memberships = memberships[memberships["instrument"].isin(instruments)].reset_index(
            drop=True
        )

        decision_dates = tuple(
            date for date in open_dates if date >= self.config.dates.backtest_start
        )
        baseline_coverage = industry_coverage_by_date(
            memberships,
            benchmark_weights,
            decision_dates,
            transition_date=self.config.features.industry_transition_date,
        )
        missing_instruments = self._missing_industry_instruments(baseline_coverage)
        supplemental_frames: list[pd.DataFrame] = []
        requested_calls = 0
        response_rows = 0
        if self.config.download.supplement_industry_by_instrument and missing_instruments:
            LOGGER.info(
                "Industry coverage supplement: querying %d missing instruments",
                len(missing_instruments),
            )
            for position, instrument in enumerate(missing_instruments, start=1):
                for is_new in ("Y", "N"):
                    supplemental = self._fetch(
                        "index_member_all",
                        params={"ts_code": instrument, "is_new": is_new},
                        fields=member_fields,
                        force=force,
                        cache_tag=cache_tag,
                    )
                    self._assert_not_at_limit(
                        "index_member_all[ts_code]",
                        supplemental,
                        2000,
                    )
                    requested_calls += 1
                    response_rows += len(supplemental)
                    supplemental_frames.append(supplemental)
                if position % 50 == 0 or position == len(missing_instruments):
                    LOGGER.info(
                        "Industry coverage supplement: completed %d/%d instruments",
                        position,
                        len(missing_instruments),
                    )

        if supplemental_frames:
            supplemental_raw = pd.concat(supplemental_frames, ignore_index=True)
            for taxonomy, l1_codes in classification_codes.items():
                scoped = supplemental_raw[supplemental_raw["l1_code"].astype(str).isin(l1_codes)]
                if not scoped.empty:
                    membership_frames.append(scoped.assign(taxonomy=taxonomy))
            memberships = normalize_industry_membership(membership_frames)
            memberships = memberships[memberships["instrument"].isin(instruments)].reset_index(
                drop=True
            )

        final_coverage = industry_coverage_by_date(
            memberships,
            benchmark_weights,
            decision_dates,
            transition_date=self.config.features.industry_transition_date,
        )
        remaining_missing = self._missing_industry_instruments(final_coverage)
        self.industry_supplement_audit = {
            "enabled": self.config.download.supplement_industry_by_instrument,
            "strategy": "l1_bulk_then_missing_ts_code",
            "baseline": self._industry_coverage_summary(baseline_coverage),
            "requested_instruments": len(missing_instruments)
            if self.config.download.supplement_industry_by_instrument
            else 0,
            "requested_calls": requested_calls,
            "response_rows": response_rows,
            "final": self._industry_coverage_summary(final_coverage),
            "remaining_missing_instruments": len(remaining_missing),
            "remaining_missing_sample": list(remaining_missing[:20]),
        }
        return classifications, memberships

    @staticmethod
    def _missing_industry_instruments(coverage: pd.DataFrame) -> tuple[str, ...]:
        if coverage.empty:
            return ()
        missing: set[str] = set()
        for values in coverage["missing_instruments"]:
            missing.update(str(value) for value in values)
        return tuple(sorted(missing))

    @staticmethod
    def _industry_coverage_summary(coverage: pd.DataFrame) -> dict[str, Any]:
        if coverage.empty:
            return {
                "days_checked": 0,
                "minimum_coverage": 0.0,
                "minimum_date": None,
                "minimum_missing_members": None,
            }
        minimum = coverage.loc[coverage["coverage"].idxmin()]
        return {
            "days_checked": len(coverage),
            "minimum_coverage": float(minimum["coverage"]),
            "minimum_date": str(minimum["decision_date"]),
            "minimum_taxonomy": str(minimum["taxonomy"]),
            "minimum_missing_members": int(minimum["missing_members"]),
        }

    def _download_partition(
        self,
        partition: DatePartition,
        all_open_dates: tuple[str, ...],
        instruments: frozenset[str],
        contract_hash: str,
        *,
        force: bool,
    ) -> PartitionSummary:
        open_dates = tuple(
            date for date in all_open_dates if partition.start_date <= date <= partition.end_date
        )
        if not open_dates:
            raise DataQualityError(f"No open dates in annual partition {partition.partition_id}")
        daily_frames: list[pd.DataFrame] = []
        adjustment_frames: list[pd.DataFrame] = []
        limit_frames: list[pd.DataFrame] = []
        basic_frames: list[pd.DataFrame] = []
        suspension_frames: list[pd.DataFrame] = []
        limit_event_start = len(self.response_limit_events)

        for position, trade_date in enumerate(open_dates, start=1):
            daily = self._fetch(
                "daily",
                params={"trade_date": trade_date},
                fields=(
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "vol",
                    "amount",
                ),
                force=force,
            )
            self._assert_not_at_limit("daily", daily, 6000)
            scoped_daily = daily[daily["ts_code"].isin(instruments)]
            daily_frames.append(scoped_daily)

            adjustment = self._fetch(
                "adj_factor",
                params={"trade_date": trade_date},
                fields=("ts_code", "trade_date", "adj_factor"),
                force=force,
            )
            self._assert_not_at_limit("adj_factor", adjustment, 6000)
            adjustment_frames.append(adjustment[adjustment["ts_code"].isin(instruments)])

            limits = self._fetch(
                "stk_limit",
                params={"trade_date": trade_date},
                fields=("ts_code", "trade_date", "up_limit", "down_limit"),
                force=force,
            )
            scoped_limits = limits[limits["ts_code"].isin(instruments)]
            self._verify_scoped_limit_response(
                raw=limits,
                scoped=scoped_limits,
                required_codes=set(scoped_daily["ts_code"].astype(str)),
                trade_date=trade_date,
            )
            limit_frames.append(scoped_limits)

            if self.config.download.include_daily_basic:
                basic = self._fetch(
                    "daily_basic",
                    params={"trade_date": trade_date},
                    fields=(
                        "ts_code",
                        "trade_date",
                        "turnover_rate",
                        "turnover_rate_f",
                        "pb",
                        "total_mv",
                        "circ_mv",
                    ),
                    force=force,
                )
                self._assert_not_at_limit("daily_basic", basic, 6000)
                basic_frames.append(basic[basic["ts_code"].isin(instruments)])

            if self.config.download.include_suspensions:
                suspension = self._fetch(
                    "suspend_d",
                    params={"trade_date": trade_date, "suspend_type": "S"},
                    fields=("ts_code", "trade_date", "suspend_timing", "suspend_type"),
                    force=force,
                )
                suspension_frames.append(suspension[suspension["ts_code"].isin(instruments)])

            if position % 20 == 0 or position == len(open_dates):
                LOGGER.info(
                    "Partition %s: downloaded %d/%d open dates",
                    partition.partition_id,
                    position,
                    len(open_dates),
                )

        partition_limit_events = self.response_limit_events[limit_event_start:]
        if partition_limit_events:
            LOGGER.info(
                "Partition %s: %d stk_limit responses reached the documented row "
                "limit; project-scope completeness passed for every event",
                partition.partition_id,
                len(partition_limit_events),
            )

        tables: dict[str, pd.DataFrame] = {
            "stock_bars": normalize_stock_bars(daily_frames),
            "adjustments": normalize_adjustments(adjustment_frames),
            "price_limits": normalize_limits(limit_frames),
        }
        if self.config.download.include_daily_basic:
            tables["daily_characteristics"] = normalize_daily_basic(basic_frames)
        if self.config.download.include_suspensions:
            tables["suspensions"] = normalize_suspensions(suspension_frames)
        self._validate_partition_tables(tables, partition)

        partition_root = self._partition_root(partition)
        fingerprints: dict[str, str] = {}
        rows: dict[str, int] = {}
        columns: dict[str, list[str]] = {}
        for name, frame in tables.items():
            path = partition_root / f"{name}.parquet"
            fingerprints[name] = write_parquet_atomic(frame, path)
            rows[name] = len(frame)
            columns[name] = list(frame.columns)

        manifest_path = partition_root / "partition-manifest.json"
        write_json_atomic(
            {
                "status": "success",
                "created_at": utc_now(),
                "contract_version": PARTITION_CONTRACT_VERSION,
                "contract_hash": contract_hash,
                **partition.to_dict(),
                "open_dates": len(open_dates),
                "rows": rows,
                "columns": columns,
                "fingerprints": fingerprints,
                "response_limit_events": self.response_limit_events[limit_event_start:],
            },
            manifest_path,
        )
        return PartitionSummary(
            partition_id=partition.partition_id,
            start_date=partition.start_date,
            end_date=partition.end_date,
            status="downloaded",
            rows=rows,
            manifest_path=manifest_path,
        )

    def _load_valid_partition(
        self,
        partition: DatePartition,
        contract_hash: str,
        all_open_dates: tuple[str, ...],
    ) -> PartitionSummary | None:
        partition_root = self._partition_root(partition)
        manifest_path = partition_root / "partition-manifest.json"
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                return None
            if payload.get("contract_hash") != contract_hash:
                return None
            frames: dict[str, pd.DataFrame] = {}
            expected_tables = self._partition_table_names()
            if set(payload.get("fingerprints", {})) != set(expected_tables):
                return None
            for name in expected_tables:
                path = partition_root / f"{name}.parquet"
                if not path.exists():
                    return None
                if sha256_file(path) != payload["fingerprints"][name]:
                    return None
                frame = pd.read_parquet(path)
                if len(frame) != int(payload["rows"][name]):
                    return None
                if list(frame.columns) != list(payload["columns"][name]):
                    return None
                frames[name] = frame
            open_dates = [
                date
                for date in all_open_dates
                if partition.start_date <= date <= partition.end_date
            ]
            if len(open_dates) != int(payload.get("open_dates", -1)):
                return None
            self._validate_partition_tables(frames, partition)
            self.response_limit_events.extend(payload.get("response_limit_events", []))
        except Exception:
            LOGGER.warning(
                "Ignoring invalid annual partition cache: %s",
                partition.partition_id,
            )
            return None
        LOGGER.info("Reusing validated annual partition %s", partition.partition_id)
        return PartitionSummary(
            partition_id=partition.partition_id,
            start_date=partition.start_date,
            end_date=partition.end_date,
            status="reused",
            rows={name: int(payload["rows"][name]) for name in expected_tables},
            manifest_path=manifest_path,
        )

    def _materialize_snapshot(
        self,
        reference_tables: dict[str, pd.DataFrame],
        partitions: tuple[DatePartition, ...],
    ) -> tuple[dict[str, Path], dict[str, int], dict[str, str]]:
        paths: dict[str, Path] = {}
        rows: dict[str, int] = {}
        fingerprints: dict[str, str] = {}
        for name, frame in reference_tables.items():
            path = self.config.paths.silver_root / f"{name}.parquet"
            fingerprints[name] = write_parquet_atomic(frame, path)
            paths[name] = path
            rows[name] = len(frame)

        for name in self._partition_table_names():
            frames = [
                pd.read_parquet(self._partition_root(partition) / f"{name}.parquet")
                for partition in partitions
            ]
            frame = self._merge_partition_frames(name, frames)
            path = self.config.paths.silver_root / f"{name}.parquet"
            fingerprints[name] = write_parquet_atomic(frame, path)
            paths[name] = path
            rows[name] = len(frame)
        return paths, rows, fingerprints

    @staticmethod
    def _merge_partition_frames(
        name: str,
        frames: list[pd.DataFrame],
    ) -> pd.DataFrame:
        if not frames:
            raise DataQualityError(f"No annual partition frames found for {name}")
        merged = pd.concat(frames, ignore_index=True)
        keys = {
            "stock_bars": ["trade_date", "instrument"],
            "adjustments": ["trade_date", "instrument"],
            "price_limits": ["trade_date", "instrument"],
            "daily_characteristics": ["trade_date", "instrument"],
            "suspensions": ["trade_date", "instrument", "suspend_type"],
        }[name]
        return merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)

    def _validate_partition_tables(
        self,
        tables: dict[str, pd.DataFrame],
        partition: DatePartition,
    ) -> None:
        required_columns = {
            "stock_bars": {
                "trade_date",
                "instrument",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "volume_shares",
                "amount_cny",
            },
            "adjustments": {"trade_date", "instrument", "adj_factor"},
            "price_limits": {
                "trade_date",
                "instrument",
                "up_limit",
                "down_limit",
            },
            "daily_characteristics": {
                "trade_date",
                "instrument",
                "turnover_rate",
                "turnover_rate_f",
                "pb",
                "total_mv_cny",
                "circ_mv_cny",
            },
            "suspensions": {
                "trade_date",
                "instrument",
                "suspend_timing",
                "suspend_type",
            },
        }
        keys = {
            "stock_bars": ["trade_date", "instrument"],
            "adjustments": ["trade_date", "instrument"],
            "price_limits": ["trade_date", "instrument"],
            "daily_characteristics": ["trade_date", "instrument"],
            "suspensions": ["trade_date", "instrument", "suspend_type"],
        }
        expected_names = set(self._partition_table_names())
        if set(tables) != expected_names:
            raise DataQualityError(
                f"Annual partition table mismatch: expected={sorted(expected_names)}, "
                f"actual={sorted(tables)}"
            )
        for name, frame in tables.items():
            missing = sorted(required_columns[name].difference(frame.columns))
            if missing:
                raise DataQualityError(
                    f"Annual partition schema mismatch: table={name}, missing={missing}"
                )
            if name != "suspensions" and frame.empty:
                raise DataQualityError(
                    f"Annual partition unexpectedly empty: {partition.partition_id}/{name}"
                )
            if frame.duplicated(keys[name]).any():
                raise DataQualityError(
                    f"Annual partition contains duplicate keys: {partition.partition_id}/{name}"
                )
            if not frame.empty:
                dates = frame["trade_date"].astype(str)
                if dates.min() < partition.start_date or dates.max() > partition.end_date:
                    raise DataQualityError(
                        f"Annual partition date leak: {partition.partition_id}/{name}"
                    )

        bar_keys = tables["stock_bars"][["trade_date", "instrument"]]
        for name, threshold in (("adjustments", 0.98), ("price_limits", 0.98)):
            coverage = (
                bar_keys.merge(
                    tables[name][["trade_date", "instrument"]],
                    on=["trade_date", "instrument"],
                    how="left",
                    indicator=True,
                )["_merge"]
                .eq("both")
                .mean()
            )
            if float(coverage) < threshold:
                raise DataQualityError(
                    f"Annual partition cross-table coverage failed: "
                    f"partition={partition.partition_id}, table={name}, coverage={coverage:.6f}"
                )
        if self.config.download.include_daily_basic:
            coverage = (
                bar_keys.merge(
                    tables["daily_characteristics"][["trade_date", "instrument"]],
                    on=["trade_date", "instrument"],
                    how="left",
                    indicator=True,
                )["_merge"]
                .eq("both")
                .mean()
            )
            if float(coverage) < 0.98:
                raise DataQualityError(
                    "Annual partition daily-characteristics coverage failed: "
                    f"partition={partition.partition_id}, coverage={coverage:.6f}"
                )

    def _partition_table_names(self) -> tuple[str, ...]:
        names = ["stock_bars", "adjustments", "price_limits"]
        if self.config.download.include_daily_basic:
            names.append("daily_characteristics")
        if self.config.download.include_suspensions:
            names.append("suspensions")
        return tuple(names)

    def _verify_scoped_limit_response(
        self,
        *,
        raw: pd.DataFrame,
        scoped: pd.DataFrame,
        required_codes: set[str],
        trade_date: str,
    ) -> None:
        documented_limit = 5800
        if len(raw) < documented_limit:
            return
        available_codes = set(scoped["ts_code"].astype(str))
        missing = sorted(required_codes.difference(available_codes))
        event = {
            "api_name": "stk_limit",
            "trade_date": trade_date,
            "raw_rows": len(raw),
            "documented_limit": documented_limit,
            "required_scope_rows": len(required_codes),
            "missing_required_codes": len(missing),
            "resolution": "target_scope_complete" if not missing else "target_scope_incomplete",
        }
        self.response_limit_events.append(event)
        if missing:
            raise DataQualityError(
                "Tushare stk_limit response reached the documented limit and omitted "
                f"required project-scope codes: trade_date={trade_date}, "
                f"raw_rows={len(raw)}, missing_count={len(missing)}, "
                f"missing_sample={missing[:10]}"
            )
        LOGGER.debug(
            "stk_limit returned %d rows on %s (documented limit %d), but all %d "
            "project-scope traded instruments were present",
            len(raw),
            trade_date,
            documented_limit,
            len(required_codes),
        )

    def _partition_root(self, partition: DatePartition) -> Path:
        return self.config.paths.silver_root / "_partitions" / partition.partition_id

    def _partition_contract_hash(
        self,
        partition: DatePartition,
        instrument_hash: str,
    ) -> str:
        return sha256_text(
            canonical_json(
                {
                    "contract_version": PARTITION_CONTRACT_VERSION,
                    "partition": partition.to_dict(),
                    "instrument_hash": instrument_hash,
                    "source": self._partition_source_contract(),
                    "download": self._partition_download_contract(),
                    "tables": list(self._partition_table_names()),
                }
            )
        )

    def _source_contract(self) -> dict[str, Any]:
        return {
            "vendor": "Tushare Pro",
            "exchange": self.config.source.exchange,
            "index_code": self.config.source.index_code,
            "total_return_index_code": (
                self.config.source.total_return_index_code
            ),
            "benchmark_membership_contract": (
                BENCHMARK_MEMBERSHIP_CONTRACT_VERSION
            ),
            "request_timeout_seconds": self.config.source.request_timeout_seconds,
            "calls_per_minute_limit": self.config.source.calls_per_minute_limit,
            "effective_min_request_interval_seconds": (
                self.config.source.effective_min_request_interval_seconds
            ),
        }

    def _partition_source_contract(self) -> dict[str, Any]:
        """Preserve the v3 daily-partition identity while reference data evolves."""

        source = self._source_contract().copy()
        source.pop("total_return_index_code")
        source.pop("benchmark_membership_contract")
        return source

    def _partition_download_contract(self) -> dict[str, Any]:
        """Exclude reference-only refresh settings from daily partition identity."""

        download = asdict(self.config.download)
        download.pop("reference_cache_tag")
        download.pop("eligibility_refresh_start")
        return download

    @staticmethod
    def _failure_payload(exc: Exception) -> dict[str, str]:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        }

    @staticmethod
    def _initial_progress(plan: DownloadPlan) -> dict[str, Any]:
        return {
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "plan": plan.to_dict(),
            "reference": {"status": "pending"},
            "partitions": {
                partition.partition_id: {
                    **partition.to_dict(),
                    "status": "pending",
                }
                for partition in plan.partitions
            },
        }
