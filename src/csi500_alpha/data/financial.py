from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from csi500_alpha.config import AppConfig
from csi500_alpha.data.client import TushareClient
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import ConfigurationError, DataQualityError
from csi500_alpha.utils import canonical_json, sha256_file, sha256_text, utc_now

FINANCIAL_CONTRACT_VERSION = "csi500-tushare-financial-v2"
CORE_FINANCIAL_APIS = ("fina_indicator", "income", "balancesheet", "cashflow")

FINANCIAL_API_FIELDS: dict[str, tuple[str, ...]] = {
    "fina_indicator": (
        "ts_code",
        "ann_date",
        "end_date",
        "roe",
        "roa",
        "roic",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "current_ratio",
        "quick_ratio",
        "assets_turn",
        "ocf_to_or",
        "ocf_to_profit",
        "netprofit_yoy",
        "dt_netprofit_yoy",
        "ocf_yoy",
        "assets_yoy",
        "eqt_yoy",
        "q_sales_yoy",
        "q_op_yoy",
        "q_netprofit_yoy",
        "q_netprofit_qoq",
        "rd_exp",
        "update_flag",
    ),
    "income": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "total_revenue",
        "revenue",
        "oper_cost",
        "operate_profit",
        "total_profit",
        "n_income",
        "n_income_attr_p",
        "update_flag",
    ),
    "balancesheet": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "money_cap",
        "accounts_receiv",
        "inventories",
        "total_cur_assets",
        "fix_assets",
        "intan_assets",
        "goodwill",
        "total_assets",
        "st_borr",
        "lt_borr",
        "acct_payable",
        "total_cur_liab",
        "total_liab",
        "total_hldr_eqy_exc_min_int",
        "update_flag",
    ),
    "cashflow": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "net_profit",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
        "n_cashflow_inv_act",
        "n_cash_flows_fnc_act",
        "free_cashflow",
        "update_flag",
    ),
}

DISCLOSURE_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "pre_date",
    "actual_date",
    "modify_date",
)

_COMMON_STATEMENT_COLUMNS = (
    "instrument",
    "report_period",
    "statement_type",
    "report_type",
    "company_type",
    "period_type",
    "update_flag",
    "announcement_date",
    "actual_announcement_date",
    "disclosure_actual_date",
    "source_announcement_date",
    "available_date",
    "availability_status",
    "version_sequence",
    "vendor_row_hash",
    "materialized_at_utc",
)


def _date(value: Any, key: str) -> str:
    text = str(value)
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ConfigurationError(f"{key} must use YYYYMMDD: {text}") from exc
    return text


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _sequence(value: Any, key: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{key} must be a sequence")
    return tuple(value)


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be true or false")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], key: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        names = ", ".join(f"{key}.{name}" for name in unknown)
        raise ConfigurationError(f"Unknown configuration keys: {names}")


@dataclass(frozen=True)
class FinancialDownloadSpec:
    config_path: Path
    base_config: AppConfig
    output_subdirectory: str
    announcement_start: str
    announcement_end: str
    report_period_start: str
    report_period_end: str
    api_names: tuple[str, ...]
    report_type: str
    instruments: tuple[str, ...]
    instrument_limit: int | None
    response_row_limit: int
    include_disclosure_schedule: bool
    disclosure_response_row_limit: int
    availability_lag_open_days: int
    cache_tag: str | None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> FinancialDownloadSpec:
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigurationError(f"Financial configuration does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("Financial configuration root must be a mapping")
        _reject_unknown(raw, {"base_config", "financial"}, "financial_config")
        if "base_config" not in raw:
            raise ConfigurationError("Missing configuration key: base_config")
        base_path = Path(str(raw["base_config"]))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_config = AppConfig.from_yaml(base_path)
        section = _mapping(raw.get("financial", {}), "financial")
        _reject_unknown(
            section,
            {
                "output_subdirectory",
                "announcement_start",
                "announcement_end",
                "report_period_start",
                "report_period_end",
                "apis",
                "report_type",
                "instruments",
                "instrument_limit",
                "response_row_limit",
                "include_disclosure_schedule",
                "disclosure_response_row_limit",
                "availability_lag_open_days",
                "cache_tag",
            },
            "financial",
        )
        announcement_start = _date(
            section.get("announcement_start", base_config.dates.raw_start),
            "financial.announcement_start",
        )
        announcement_end = _date(
            section.get("announcement_end", base_config.dates.end),
            "financial.announcement_end",
        )
        report_period_start = _date(
            section.get("report_period_start", announcement_start),
            "financial.report_period_start",
        )
        report_period_end = _date(
            section.get("report_period_end", announcement_end),
            "financial.report_period_end",
        )
        if announcement_start > announcement_end:
            raise ConfigurationError(
                "financial announcement_start must not exceed announcement_end"
            )
        if announcement_end > base_config.dates.end:
            raise ConfigurationError(
                "financial.announcement_end must not exceed the base calendar end date"
            )
        if report_period_start > report_period_end:
            raise ConfigurationError(
                "financial report_period_start must not exceed report_period_end"
            )
        if not _is_quarter_end(report_period_start) or not _is_quarter_end(report_period_end):
            raise ConfigurationError("Financial report-period bounds must be calendar quarter ends")

        api_names = tuple(
            str(name)
            for name in _sequence(section.get("apis", CORE_FINANCIAL_APIS), "financial.apis")
        )
        if not api_names or len(set(api_names)) != len(api_names):
            raise ConfigurationError("financial.apis must contain unique API names")
        unknown_apis = sorted(set(api_names).difference(CORE_FINANCIAL_APIS))
        if unknown_apis:
            raise ConfigurationError(f"Unsupported financial APIs: {unknown_apis}")

        output_subdirectory = str(section.get("output_subdirectory", "financial"))
        output_path = Path(output_subdirectory)
        if (
            output_path.is_absolute()
            or output_path.name != output_subdirectory
            or output_subdirectory in {"", ".", ".."}
        ):
            raise ConfigurationError("financial.output_subdirectory must be one path segment")

        instruments = tuple(
            str(value).upper()
            for value in _sequence(section.get("instruments", ()), "financial.instruments")
        )
        invalid_instruments = [
            instrument
            for instrument in instruments
            if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", instrument) is None
        ]
        if invalid_instruments:
            raise ConfigurationError(
                f"Invalid Tushare instruments: {invalid_instruments[:5]}"
            )
        if len(set(instruments)) != len(instruments):
            raise ConfigurationError("financial.instruments must be unique")

        raw_limit = section.get("instrument_limit")
        instrument_limit = int(raw_limit) if raw_limit is not None else None
        if instrument_limit is not None and instrument_limit <= 0:
            raise ConfigurationError("financial.instrument_limit must be positive")
        response_row_limit = int(section.get("response_row_limit", 100))
        disclosure_row_limit = int(section.get("disclosure_response_row_limit", 6000))
        if response_row_limit < 2 or disclosure_row_limit < 2:
            raise ConfigurationError("Financial response row limits must be at least two")
        availability_lag = int(section.get("availability_lag_open_days", 1))
        if availability_lag < 1:
            raise ConfigurationError("financial.availability_lag_open_days must be positive")
        report_type = str(section.get("report_type", "1"))
        if report_type != "1":
            raise ConfigurationError(
                "The initial financial contract only supports latest consolidated report_type=1"
            )

        return cls(
            config_path=path,
            base_config=base_config,
            output_subdirectory=output_subdirectory,
            announcement_start=announcement_start,
            announcement_end=announcement_end,
            report_period_start=report_period_start,
            report_period_end=report_period_end,
            api_names=api_names,
            report_type=report_type,
            instruments=instruments,
            instrument_limit=instrument_limit,
            response_row_limit=response_row_limit,
            include_disclosure_schedule=_boolean(
                section.get("include_disclosure_schedule", True),
                "financial.include_disclosure_schedule",
            ),
            disclosure_response_row_limit=disclosure_row_limit,
            availability_lag_open_days=availability_lag,
            cache_tag=(
                str(section["cache_tag"]) if section.get("cache_tag") is not None else None
            ),
        )

    @property
    def silver_root(self) -> Path:
        return self.base_config.paths.silver_root / self.output_subdirectory

    @property
    def quality_root(self) -> Path:
        return self.base_config.paths.quality_root

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "base_config_path": str(self.base_config.config_path),
            "output_subdirectory": self.output_subdirectory,
            "announcement_start": self.announcement_start,
            "announcement_end": self.announcement_end,
            "report_period_start": self.report_period_start,
            "report_period_end": self.report_period_end,
            "apis": list(self.api_names),
            "report_type": self.report_type,
            "configured_instruments": list(self.instruments),
            "instrument_limit": self.instrument_limit,
            "response_row_limit": self.response_row_limit,
            "include_disclosure_schedule": self.include_disclosure_schedule,
            "disclosure_response_row_limit": self.disclosure_response_row_limit,
            "availability_lag_open_days": self.availability_lag_open_days,
            "cache_tag": self.cache_tag,
        }


@dataclass(frozen=True)
class FinancialDownloadPlan:
    instruments: tuple[str, ...]
    report_periods: tuple[str, ...]
    base_requests: int
    theoretical_minimum_minutes: float
    output_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_count": len(self.instruments),
            "instrument_sample": list(self.instruments[:10]),
            "report_period_count": len(self.report_periods),
            "report_periods": list(self.report_periods),
            "base_requests": self.base_requests,
            "request_estimate_note": (
                "Base count; saturated responses are split into smaller announcement-date windows."
            ),
            "theoretical_minimum_minutes": self.theoretical_minimum_minutes,
            "output_root": str(self.output_root),
        }


@dataclass(frozen=True)
class FinancialDownloadSummary:
    paths: dict[str, Path]
    rows: dict[str, int]
    cache_hits: int
    network_requests: int
    quality_path: Path
    manifest_path: Path
    progress_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": {name: str(path) for name, path in self.paths.items()},
            "rows": self.rows,
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "quality_path": str(self.quality_path),
            "manifest_path": str(self.manifest_path),
            "progress_path": str(self.progress_path),
        }


def build_financial_download_plan(spec: FinancialDownloadSpec) -> FinancialDownloadPlan:
    instruments = resolve_financial_instruments(spec)
    periods = quarter_ends(spec.report_period_start, spec.report_period_end)
    disclosure_requests = len(periods) if spec.include_disclosure_schedule else 0
    base_requests = len(instruments) * len(spec.api_names) + disclosure_requests
    return FinancialDownloadPlan(
        instruments=instruments,
        report_periods=periods,
        base_requests=base_requests,
        theoretical_minimum_minutes=(
            base_requests
            * spec.base_config.source.effective_min_request_interval_seconds
            / 60.0
        ),
        output_root=spec.silver_root,
    )


def resolve_financial_instruments(spec: FinancialDownloadSpec) -> tuple[str, ...]:
    weights_path = spec.base_config.paths.silver_root / "benchmark_weights.parquet"
    if not weights_path.exists():
        raise DataQualityError(
            "Financial download requires an existing benchmark_weights.parquet snapshot"
        )
    weights = pd.read_parquet(weights_path, columns=["instrument"])
    universe = tuple(sorted(weights["instrument"].dropna().astype(str).unique()))
    if not universe:
        raise DataQualityError("Financial download found no benchmark instruments")
    if spec.instruments:
        missing = sorted(set(spec.instruments).difference(universe))
        if missing:
            raise DataQualityError(
                "Configured financial instruments are outside the benchmark history: "
                f"{missing[:10]}"
            )
        instruments = tuple(sorted(spec.instruments))
    else:
        instruments = universe
    if spec.instrument_limit is not None:
        instruments = instruments[: spec.instrument_limit]
    if not instruments:
        raise DataQualityError("Financial instrument selection is empty")
    return instruments


def quarter_ends(start_date: str, end_date: str) -> tuple[str, ...]:
    if not _is_quarter_end(start_date) or not _is_quarter_end(end_date):
        raise ConfigurationError("Report-period bounds must be calendar quarter ends")
    periods = pd.period_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
        freq="Q-DEC",
    )
    return tuple(period.end_time.strftime("%Y%m%d") for period in periods)


def _is_quarter_end(value: str) -> bool:
    return value[4:] in {"0331", "0630", "0930", "1231"}


class FinancialDownloader:
    """Resumable point-in-time downloader for core financial statements."""

    def __init__(self, spec: FinancialDownloadSpec, client: TushareClient) -> None:
        self.spec = spec
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

    def run(self, *, force: bool = False) -> FinancialDownloadSummary:
        self.cache_hits = 0
        self.network_requests = 0
        plan = build_financial_download_plan(self.spec)
        open_dates = _load_open_dates(self.spec.base_config)
        file_stem = self.spec.output_subdirectory
        progress_path = self.spec.quality_root / f"{file_stem}-download-progress.json"
        quality_path = self.spec.quality_root / f"{file_stem}-data-quality.json"
        manifest_path = self.spec.quality_root / f"{file_stem}-dataset-manifest.json"
        progress: dict[str, Any] = {
            "status": "running",
            "started_at": utc_now(),
            "contract_version": FINANCIAL_CONTRACT_VERSION,
            "plan": plan.to_dict(),
            "apis": {
                api_name: {
                    "completed_instruments": 0,
                    "total_instruments": len(plan.instruments),
                }
                for api_name in self.spec.api_names
            },
            "disclosure_schedule": {
                "completed_periods": 0,
                "total_periods": len(plan.report_periods),
            },
            "force": force,
        }
        write_json_atomic(progress, progress_path)

        raw_tables: dict[str, list[pd.DataFrame]] = {
            api_name: [] for api_name in self.spec.api_names
        }
        disclosure_frames: list[pd.DataFrame] = []
        try:
            for api_name in self.spec.api_names:
                for position, instrument in enumerate(plan.instruments, start=1):
                    raw_tables[api_name].append(
                        self._fetch_statement_range(
                            api_name,
                            instrument,
                            self.spec.announcement_start,
                            self.spec.announcement_end,
                            force=force,
                        )
                    )
                    if position % 25 == 0 or position == len(plan.instruments):
                        progress["apis"][api_name]["completed_instruments"] = position
                        self._update_request_counts(progress, progress_path)

            if self.spec.include_disclosure_schedule:
                for position, report_period in enumerate(plan.report_periods, start=1):
                    disclosure_frames.append(
                        self._fetch_disclosure_period(
                            report_period,
                            plan.instruments,
                            force=force,
                        )
                    )
                    progress["disclosure_schedule"]["completed_periods"] = position
                    self._update_request_counts(progress, progress_path)
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

        materialized_at = utc_now()
        disclosure = normalize_disclosure_schedule(
            disclosure_frames,
            instruments=plan.instruments,
            materialized_at=materialized_at,
        )
        statement_tables = {
            api_name: normalize_financial_statement(
                api_name,
                frames,
                disclosure_schedule=disclosure,
                open_dates=open_dates,
                availability_lag_open_days=self.spec.availability_lag_open_days,
                instruments=plan.instruments,
                materialized_at=materialized_at,
                announcement_start=self.spec.announcement_start,
                announcement_end=self.spec.announcement_end,
                report_period_start=self.spec.report_period_start,
                report_period_end=self.spec.report_period_end,
            )
            for api_name, frames in raw_tables.items()
        }
        availability_index = build_financial_availability_index(statement_tables)
        tables: dict[str, pd.DataFrame] = {
            **statement_tables,
            "disclosure_schedule": disclosure,
            "availability_index": availability_index,
        }
        validation = validate_financial_tables(
            statement_tables,
            disclosure,
            instruments=plan.instruments,
            open_dates=open_dates,
            report_period_start=self.spec.report_period_start,
            report_period_end=self.spec.report_period_end,
        )
        paths = {
            name: self.spec.silver_root / f"{name}.parquet" for name in tables
        }
        fingerprints = {
            name: write_parquet_atomic(table, paths[name]) for name, table in tables.items()
        }
        rows = {name: len(table) for name, table in tables.items()}
        quality = {
            "status": "success" if validation["passed"] else "failed",
            "created_at": materialized_at,
            "contract_version": FINANCIAL_CONTRACT_VERSION,
            "spec": self.spec.to_dict(),
            "plan": plan.to_dict(),
            "rows": rows,
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "validation": validation,
            "paths": {name: str(path) for name, path in paths.items()},
            "fingerprints": fingerprints,
        }
        write_json_atomic(quality, quality_path)
        dataset_manifest = {
            "status": quality["status"],
            "created_at": materialized_at,
            "contract_version": FINANCIAL_CONTRACT_VERSION,
            "financial_config": {
                "path": str(self.spec.config_path),
                "sha256": sha256_file(self.spec.config_path),
            },
            "base_config": {
                "path": str(self.spec.base_config.config_path),
                "sha256": sha256_file(self.spec.base_config.config_path),
            },
            "request_counts": {
                "cache_hits": self.cache_hits,
                "network_requests": self.network_requests,
            },
            "tables": {
                name: {
                    "path": str(paths[name]),
                    "rows": rows[name],
                    "sha256": fingerprints[name],
                }
                for name in tables
            },
            "quality_path": str(quality_path),
        }
        write_json_atomic(dataset_manifest, manifest_path)
        progress.update(
            {
                "status": quality["status"],
                "finished_at": utc_now(),
                "quality_path": str(quality_path),
                "manifest_path": str(manifest_path),
                "cache_hits": self.cache_hits,
                "network_requests": self.network_requests,
            }
        )
        write_json_atomic(progress, progress_path)
        if not validation["passed"]:
            raise DataQualityError(
                f"Financial data failed quality checks; see {quality_path}"
            )
        return FinancialDownloadSummary(
            paths=paths,
            rows=rows,
            cache_hits=self.cache_hits,
            network_requests=self.network_requests,
            quality_path=quality_path,
            manifest_path=manifest_path,
            progress_path=progress_path,
        )

    def _update_request_counts(self, progress: dict[str, Any], path: Path) -> None:
        progress["cache_hits"] = self.cache_hits
        progress["network_requests"] = self.network_requests
        write_json_atomic(progress, path)

    def _fetch_statement_range(
        self,
        api_name: str,
        instrument: str,
        announcement_start: str,
        announcement_end: str,
        *,
        force: bool,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "ts_code": instrument,
            "start_date": announcement_start,
            "end_date": announcement_end,
        }
        if api_name != "fina_indicator":
            params["report_type"] = self.spec.report_type
        frame = self._fetch(
            api_name,
            params=params,
            fields=FINANCIAL_API_FIELDS[api_name],
            force=force,
            cache_tag=self.spec.cache_tag,
        )
        if len(frame) < self.spec.response_row_limit:
            return frame
        if announcement_start == announcement_end:
            raise DataQualityError(
                f"{api_name} remained saturated for {instrument} at "
                f"{announcement_start}; "
                "the response cannot be proven complete"
            )
        left_end, right_start = _split_date_window(
            announcement_start,
            announcement_end,
        )
        left = self._fetch_statement_range(
            api_name,
            instrument,
            announcement_start,
            left_end,
            force=force,
        )
        right = self._fetch_statement_range(
            api_name,
            instrument,
            right_start,
            announcement_end,
            force=force,
        )
        return pd.concat([left, right], ignore_index=True).drop_duplicates().reset_index(drop=True)

    def _fetch_disclosure_period(
        self,
        report_period: str,
        instruments: tuple[str, ...],
        *,
        force: bool,
    ) -> pd.DataFrame:
        frame = self._fetch(
            "disclosure_date",
            params={"end_date": report_period},
            fields=DISCLOSURE_FIELDS,
            force=force,
            cache_tag=self.spec.cache_tag,
        )
        if len(frame) < self.spec.disclosure_response_row_limit:
            return frame
        frames = [
            self._fetch(
                "disclosure_date",
                params={"ts_code": instrument, "end_date": report_period},
                fields=DISCLOSURE_FIELDS,
                force=force,
                cache_tag=self.spec.cache_tag,
            )
            for instrument in instruments
        ]
        return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def normalize_disclosure_schedule(
    frames: Iterable[pd.DataFrame],
    *,
    instruments: tuple[str, ...],
    materialized_at: str,
) -> pd.DataFrame:
    columns = (
        "instrument",
        "report_period",
        "schedule_announcement_date",
        "scheduled_announcement_date",
        "disclosure_actual_date",
        "schedule_modified_date_history",
        "schedule_modified_date",
        "vendor_row_hash",
        "materialized_at_utc",
    )
    material = [frame for frame in frames if not frame.empty]
    if not material:
        return pd.DataFrame(columns=list(columns))
    result = pd.concat(material, ignore_index=True).rename(
        columns={
            "ts_code": "instrument",
            "end_date": "report_period",
            "ann_date": "schedule_announcement_date",
            "pre_date": "scheduled_announcement_date",
            "actual_date": "disclosure_actual_date",
            "modify_date": "schedule_modified_date_history",
        }
    )
    result = result[result["instrument"].astype(str).isin(set(instruments))].copy()
    for column in (
        "report_period",
        "schedule_announcement_date",
        "scheduled_announcement_date",
        "disclosure_actual_date",
    ):
        result[column] = _normalize_date_series(result[column], f"disclosure_date.{column}")
    (
        result["schedule_modified_date_history"],
        result["schedule_modified_date"],
    ) = _normalize_date_history_series(
        result["schedule_modified_date_history"],
        "disclosure_date.schedule_modified_date_history",
    )
    result["instrument"] = result["instrument"].astype("string")
    hash_columns = [column for column in result.columns if column != "vendor_row_hash"]
    result["vendor_row_hash"] = _row_hashes(result, hash_columns)
    result = result.drop_duplicates("vendor_row_hash").sort_values(
        ["instrument", "report_period", "schedule_modified_date", "vendor_row_hash"],
        na_position="first",
    )
    result["materialized_at_utc"] = materialized_at
    return result.loc[:, list(columns)].reset_index(drop=True)


def normalize_financial_statement(
    api_name: str,
    frames: Iterable[pd.DataFrame],
    *,
    disclosure_schedule: pd.DataFrame,
    open_dates: tuple[str, ...],
    availability_lag_open_days: int,
    instruments: tuple[str, ...],
    materialized_at: str,
    announcement_start: str | None = None,
    announcement_end: str | None = None,
    report_period_start: str | None = None,
    report_period_end: str | None = None,
) -> pd.DataFrame:
    if api_name not in FINANCIAL_API_FIELDS:
        raise DataQualityError(f"Unsupported financial statement API: {api_name}")
    material = [frame for frame in frames if not frame.empty]
    metric_columns = [
        column
        for column in FINANCIAL_API_FIELDS[api_name]
        if column
        not in {
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "end_type",
            "update_flag",
        }
    ]
    if not material:
        return pd.DataFrame(columns=[*_COMMON_STATEMENT_COLUMNS, *metric_columns])
    result = pd.concat(material, ignore_index=True).rename(
        columns={
            "ts_code": "instrument",
            "end_date": "report_period",
            "ann_date": "announcement_date",
            "f_ann_date": "actual_announcement_date",
            "comp_type": "company_type",
            "end_type": "period_type",
        }
    )
    result = result[result["instrument"].astype(str).isin(set(instruments))].copy()
    if "actual_announcement_date" not in result:
        result["actual_announcement_date"] = pd.NA
    for column in ("report_period", "announcement_date", "actual_announcement_date"):
        result[column] = _normalize_date_series(result[column], f"{api_name}.{column}")
    for column, default in (
        ("report_type", "NA"),
        ("company_type", ""),
        ("period_type", ""),
        ("update_flag", ""),
    ):
        if column not in result:
            result[column] = default
        result[column] = result[column].fillna(default).astype(str)
    result["instrument"] = result["instrument"].astype("string")
    result["statement_type"] = api_name
    for column in metric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    disclosure_dates = _latest_disclosure_dates(disclosure_schedule)
    result = result.merge(
        disclosure_dates,
        how="left",
        on=["instrument", "report_period"],
        validate="many_to_one",
    )
    result["disclosure_actual_date"] = _normalize_date_series(
        result["disclosure_actual_date"],
        f"{api_name}.disclosure_actual_date",
    )
    result["source_announcement_date"] = _row_max_date(
        result,
        (
            "announcement_date",
            "actual_announcement_date",
            "disclosure_actual_date",
        ),
    )
    source_rows = len(result)
    before_window = (
        result["source_announcement_date"].notna()
        & result["source_announcement_date"].astype(str).lt(announcement_start)
        if announcement_start is not None
        else pd.Series(False, index=result.index)
    )
    after_window = (
        result["source_announcement_date"].notna()
        & result["source_announcement_date"].astype(str).gt(announcement_end)
        if announcement_end is not None
        else pd.Series(False, index=result.index)
    )
    before_report_window = (
        result["report_period"].notna()
        & result["report_period"].astype(str).lt(report_period_start)
        if report_period_start is not None
        else pd.Series(False, index=result.index)
    )
    after_report_window = (
        result["report_period"].notna()
        & result["report_period"].astype(str).gt(report_period_end)
        if report_period_end is not None
        else pd.Series(False, index=result.index)
    )
    result = result.loc[
        ~before_window
        & ~after_window
        & ~before_report_window
        & ~after_report_window
    ].copy()
    available, statuses = _map_to_open_dates(
        result["source_announcement_date"],
        open_dates,
        lag_open_days=availability_lag_open_days,
    )
    result["available_date"] = available
    result["availability_status"] = statuses

    vendor_columns = [
        column
        for column in result.columns
        if column
        not in {
            "source_announcement_date",
            "available_date",
            "availability_status",
            "version_sequence",
            "vendor_row_hash",
            "materialized_at_utc",
        }
    ]
    result["vendor_row_hash"] = _row_hashes(result, vendor_columns)
    result = result.drop_duplicates("vendor_row_hash")
    result = result.sort_values(
        [
            "instrument",
            "report_period",
            "report_type",
            "source_announcement_date",
            "announcement_date",
            "actual_announcement_date",
            "update_flag",
            "vendor_row_hash",
        ],
        na_position="first",
    )
    result["version_sequence"] = (
        result.groupby(
            ["instrument", "report_period", "report_type"],
            sort=False,
            dropna=False,
        ).cumcount()
        + 1
    )
    result["materialized_at_utc"] = materialized_at
    result = result.loc[:, [*_COMMON_STATEMENT_COLUMNS, *metric_columns]].reset_index(
        drop=True
    )
    result.attrs["normalization"] = {
        "source_rows": source_rows,
        "rows_before_announcement_window": int(before_window.sum()),
        "rows_after_announcement_window": int(after_window.sum()),
        "rows_before_report_period_window": int(before_report_window.sum()),
        "rows_after_report_period_window": int(after_report_window.sum()),
        "materialized_rows": len(result),
        "announcement_start": announcement_start,
        "announcement_end": announcement_end,
        "report_period_start": report_period_start,
        "report_period_end": report_period_end,
    }
    return result


def build_financial_availability_index(
    statement_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = list(_COMMON_STATEMENT_COLUMNS)
    material = [
        table.loc[:, columns]
        for table in statement_tables.values()
        if not table.empty
    ]
    if not material:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(material, ignore_index=True)
        .sort_values(
            ["available_date", "instrument", "report_period", "statement_type"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_financial_versions_asof(
    table: pd.DataFrame,
    decision_date: str,
) -> pd.DataFrame:
    """Select the latest visible revision of every report period at a decision date."""

    try:
        datetime.strptime(decision_date, "%Y%m%d")
    except ValueError as exc:
        raise DataQualityError(
            f"decision_date must use YYYYMMDD: {decision_date}"
        ) from exc
    required = {
        "instrument",
        "report_period",
        "report_type",
        "available_date",
        "version_sequence",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise DataQualityError(f"Financial as-of table is missing columns: {missing}")
    eligible = table.loc[
        table["available_date"].notna()
        & table["available_date"].astype(str).le(decision_date)
    ].copy()
    if eligible.empty:
        eligible["decision_date"] = pd.Series(dtype="string")
        return eligible
    eligible = eligible.sort_values(
        [
            "instrument",
            "report_period",
            "report_type",
            "available_date",
            "version_sequence",
        ]
    )
    selected = eligible.groupby(
        ["instrument", "report_period", "report_type"],
        as_index=False,
        sort=False,
        dropna=False,
    ).tail(1)
    selected["decision_date"] = decision_date
    return selected.reset_index(drop=True)


def validate_financial_tables(
    statement_tables: dict[str, pd.DataFrame],
    disclosure_schedule: pd.DataFrame,
    *,
    instruments: tuple[str, ...],
    open_dates: tuple[str, ...],
    report_period_start: str,
    report_period_end: str,
) -> dict[str, Any]:
    expected_instruments = set(instruments)
    open_date_set = set(open_dates)
    table_results: dict[str, Any] = {}
    all_passed = True
    for name, table in statement_tables.items():
        covered = set(table["instrument"].dropna().astype(str))
        coverage = len(covered.intersection(expected_instruments)) / len(expected_instruments)
        duplicate_versions = int(
            table.duplicated(
                ["instrument", "report_period", "report_type", "version_sequence"]
            ).sum()
        )
        missing_source = int(table["source_announcement_date"].isna().sum())
        announcement_before_period = int(
            (
                table["source_announcement_date"].notna()
                & (
                    table["source_announcement_date"].astype("string")
                    < table["report_period"].astype("string")
                )
            ).sum()
        )
        mapped = table["available_date"].notna()
        non_strict_availability = int(
            (
                mapped
                & (
                    table["available_date"].astype("string")
                    <= table["source_announcement_date"].astype("string")
                )
            ).sum()
        )
        unavailable_without_reason = int(
            (
                table["available_date"].isna()
                & ~table["availability_status"].isin(
                    ["after_calendar_end", "missing_source_announcement"]
                )
            ).sum()
        )
        available_not_open = int(
            (~table.loc[mapped, "available_date"].astype(str).isin(open_date_set)).sum()
        )
        outside_universe = int(
            (~table["instrument"].astype(str).isin(expected_instruments)).sum()
        )
        missing_report_period = int(table["report_period"].isna().sum())
        outside_report_window = int(
            (
                table["report_period"].astype("string").lt(report_period_start)
                | table["report_period"].astype("string").gt(report_period_end)
            ).sum()
        )
        latest_period_rows = table.loc[
            table["report_period"].astype("string").eq(report_period_end)
        ]
        latest_period_coverage = (
            latest_period_rows["instrument"].astype(str).nunique()
            / len(expected_instruments)
        )
        statuses = {
            str(key): int(value)
            for key, value in table["availability_status"].value_counts(
                dropna=False
            ).items()
        }
        passed = (
            not table.empty
            and coverage >= 0.95
            and duplicate_versions == 0
            and missing_source == 0
            and announcement_before_period == 0
            and non_strict_availability == 0
            and unavailable_without_reason == 0
            and available_not_open == 0
            and outside_universe == 0
            and missing_report_period == 0
            and outside_report_window == 0
            and latest_period_coverage >= 0.90
        )
        all_passed &= passed
        table_results[name] = {
            "passed": passed,
            "rows": len(table),
            "instrument_coverage": coverage,
            "covered_instruments": len(covered.intersection(expected_instruments)),
            "duplicate_versions": duplicate_versions,
            "missing_source_announcement_dates": missing_source,
            "announcement_before_report_period": announcement_before_period,
            "non_strict_availability_dates": non_strict_availability,
            "unavailable_without_reason": unavailable_without_reason,
            "available_dates_not_in_calendar": available_not_open,
            "outside_universe_rows": outside_universe,
            "missing_report_periods": missing_report_period,
            "outside_report_window_rows": outside_report_window,
            "latest_report_period": report_period_end,
            "latest_report_period_instrument_coverage": latest_period_coverage,
            "availability_status": statuses,
            "revised_report_periods": int(
                (
                    table.groupby(
                        ["instrument", "report_period", "report_type"],
                        dropna=False,
                    ).size()
                    > 1
                ).sum()
            ),
            "normalization": dict(table.attrs.get("normalization", {})),
        }

    disclosure_duplicates = int(
        disclosure_schedule.duplicated("vendor_row_hash").sum()
        if not disclosure_schedule.empty
        else 0
    )
    disclosure_outside = int(
        (~disclosure_schedule["instrument"].astype(str).isin(expected_instruments)).sum()
        if not disclosure_schedule.empty
        else 0
    )
    all_passed &= disclosure_duplicates == 0 and disclosure_outside == 0
    return {
        "passed": all_passed,
        "tables": table_results,
        "disclosure_schedule": {
            "rows": len(disclosure_schedule),
            "exact_duplicates": disclosure_duplicates,
            "outside_universe_rows": disclosure_outside,
            "actual_date_coverage": (
                float(disclosure_schedule["disclosure_actual_date"].notna().mean())
                if not disclosure_schedule.empty
                else None
            ),
        },
        "policy": {
            "minimum_api_instrument_coverage": 0.95,
            "minimum_latest_report_period_instrument_coverage": 0.90,
            "report_period_window": [report_period_start, report_period_end],
            "availability": "first configured open date strictly after source disclosure",
            "missing_announcement_fallback": "forbidden",
            "revision_policy": "retain and sequence all distinct vendor versions",
        },
    }


def _load_open_dates(config: AppConfig) -> tuple[str, ...]:
    calendar_path = config.paths.silver_root / "calendar.parquet"
    if not calendar_path.exists():
        raise DataQualityError(
            "Financial download requires an existing calendar.parquet snapshot"
        )
    calendar = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"])
    open_dates = tuple(
        sorted(
            calendar.loc[calendar["is_open"].astype(int).eq(1), "trade_date"]
            .dropna()
            .astype(str)
            .unique()
        )
    )
    if not open_dates:
        raise DataQualityError("Financial availability mapping found no open dates")
    return open_dates


def _latest_disclosure_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    columns = ["instrument", "report_period", "disclosure_actual_date"]
    if schedule.empty:
        return pd.DataFrame(columns=columns)
    actual = schedule.dropna(subset=["disclosure_actual_date"])
    if actual.empty:
        return pd.DataFrame(columns=columns)
    return (
        actual.groupby(["instrument", "report_period"], as_index=False)[
            "disclosure_actual_date"
        ]
        .max()
        .loc[:, columns]
    )


def _normalize_date_series(series: pd.Series, key: str) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.mask(values.isin(["", "None", "nan", "NaT"]))
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        sample = values.loc[invalid].astype(str).head(5).tolist()
        raise DataQualityError(f"Invalid YYYYMMDD values in {key}: {sample}")
    return parsed.dt.strftime("%Y%m%d").astype("string")


def _normalize_date_history_series(
    series: pd.Series,
    key: str,
) -> tuple[pd.Series, pd.Series]:
    """Normalize a comma-delimited date history and derive its latest date."""
    values = series.astype("string").str.strip()
    values = values.mask(values.isin(["", "None", "nan", "NaT"]))
    histories: dict[str, str] = {}
    latest_dates: dict[str, str] = {}
    invalid: list[str] = []

    for value in values.dropna().drop_duplicates().astype(str):
        tokens = [token.strip() for token in value.split(",")]
        if any(not token for token in tokens):
            invalid.append(value)
            continue
        try:
            normalized = _normalize_date_series(pd.Series(tokens), key)
        except DataQualityError:
            invalid.append(value)
            continue
        if normalized.isna().any():
            invalid.append(value)
            continue
        canonical = normalized.astype(str).tolist()
        histories[value] = ",".join(canonical)
        latest_dates[value] = max(canonical)

    if invalid:
        raise DataQualityError(
            f"Invalid comma-delimited YYYYMMDD values in {key}: {invalid[:5]}"
        )

    return (
        values.map(histories).astype("string"),
        values.map(latest_dates).astype("string"),
    )


def _row_max_date(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    def maximum(row: pd.Series) -> Any:
        values = [str(value) for value in row if pd.notna(value) and str(value)]
        return max(values) if values else pd.NA

    return frame.loc[:, list(columns)].apply(maximum, axis=1).astype("string")


def _map_to_open_dates(
    source_dates: pd.Series,
    open_dates: tuple[str, ...],
    *,
    lag_open_days: int,
) -> tuple[pd.Series, pd.Series]:
    available: list[Any] = []
    statuses: list[str] = []
    first_open = open_dates[0]
    for value in source_dates:
        if pd.isna(value) or not str(value):
            available.append(pd.NA)
            statuses.append("missing_source_announcement")
            continue
        source_date = str(value)
        index = bisect_right(open_dates, source_date) + lag_open_days - 1
        if index >= len(open_dates):
            available.append(pd.NA)
            statuses.append("after_calendar_end")
            continue
        available.append(open_dates[index])
        statuses.append(
            "clipped_to_calendar_start" if source_date < first_open else "mapped"
        )
    return (
        pd.Series(available, index=source_dates.index, dtype="string"),
        pd.Series(statuses, index=source_dates.index, dtype="string"),
    )


def _row_hashes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    def encode(row: pd.Series) -> str:
        payload = {
            column: _json_scalar(value) for column, value in row.items()
        }
        return sha256_text(canonical_json(payload))

    return frame.loc[:, columns].apply(encode, axis=1).astype("string")


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _split_date_window(start_date: str, end_date: str) -> tuple[str, str]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    midpoint = start + (end - start) / 2
    left_end = midpoint.date()
    right_start = left_end + timedelta(days=1)
    return left_end.strftime("%Y%m%d"), right_start.strftime("%Y%m%d")
