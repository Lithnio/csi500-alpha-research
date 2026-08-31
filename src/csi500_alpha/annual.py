from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from csi500_alpha.config import AppConfig, ExperimentSettings
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.research.universe import select_rebalance_dates
from csi500_alpha.study import (
    GateRule,
    StudyContext,
    StudySpec,
    StudyTrial,
    select_study_candidates,
)
from csi500_alpha.utils import canonical_json, sha256_file, sha256_text, utc_now

_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalExperiment:
    experiment_id: str
    role: str
    status: str
    included_in_selection: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.experiment_id,
            "role": self.role,
            "status": self.status,
            "included_in_selection": self.included_in_selection,
            "note": self.note,
        }


@dataclass(frozen=True)
class AnnualStudySpec:
    config_path: Path
    annual_id: str
    method_study_path: Path
    method_study_reference: str
    method_study: StudySpec
    years: tuple[int, ...]
    train_start: str
    embargo_days: int
    max_workers: int
    publication_gates: tuple[GateRule, ...]
    history: tuple[HistoricalExperiment, ...]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> AnnualStudySpec:
        path = Path(config_path).resolve()
        if not path.is_file():
            raise ConfigurationError(
                f"Annual study configuration does not exist: {path}"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        root = _mapping(raw, "annual study config")
        _reject_unknown(
            root,
            {"annual_study", "publication_gates", "history"},
            "annual study config",
        )
        annual = _mapping(
            _required(root, "annual_study", "annual study config"),
            "annual_study",
        )
        _reject_unknown(
            annual,
            {
                "id",
                "method_study",
                "years",
                "train_start",
                "embargo_days",
                "max_workers",
            },
            "annual_study",
        )
        annual_id = _simple_name(
            _required(annual, "id", "annual_study"),
            "annual_study.id",
        )
        method_reference = str(
            _required(annual, "method_study", "annual_study")
        )
        method_path = (path.parent / method_reference).resolve()
        method_study = StudySpec.from_yaml(method_path)
        raw_years = _sequence(
            _required(annual, "years", "annual_study"),
            "annual_study.years",
        )
        years = tuple(int(value) for value in raw_years)
        if not years or years != tuple(sorted(set(years))):
            raise ConfigurationError(
                "annual_study.years must be nonempty, sorted and unique"
            )
        if any(year < 1990 or year > 2100 for year in years):
            raise ConfigurationError("annual_study.years contain an invalid year")
        if any(
            right != left + 1
            for left, right in zip(years, years[1:], strict=False)
        ):
            raise ConfigurationError("annual_study.years must be consecutive")
        train_start = _date(
            _required(annual, "train_start", "annual_study"),
            "annual_study.train_start",
        )
        embargo_days = int(annual.get("embargo_days", 5))
        if embargo_days < 0:
            raise ConfigurationError("annual_study.embargo_days cannot be negative")
        max_workers = int(annual.get("max_workers", 1))
        if not 1 <= max_workers <= 8:
            raise ConfigurationError("annual_study.max_workers must be between 1 and 8")

        raw_gates = _sequence(root.get("publication_gates", ()), "publication_gates")
        publication_gates = tuple(
            _gate_rule(value, position)
            for position, value in enumerate(raw_gates)
        )
        raw_history = _sequence(root.get("history", ()), "history")
        history = tuple(
            _historical_experiment(value, position)
            for position, value in enumerate(raw_history)
        )
        history_ids = [item.experiment_id for item in history]
        if len(history_ids) != len(set(history_ids)):
            raise ConfigurationError("Historical experiment ids must be unique")
        current_ids = {trial.trial_id for trial in method_study.trials}
        collisions = sorted(current_ids.intersection(history_ids))
        if collisions:
            raise ConfigurationError(
                "Historical experiment ids collide with current candidates: "
                f"{collisions}"
            )
        if any(item.included_in_selection for item in history):
            raise ConfigurationError(
                "Historical experiments are disclosure records and cannot enter "
                "the current candidate selection"
            )
        return cls(
            config_path=path,
            annual_id=annual_id,
            method_study_path=method_path,
            method_study_reference=method_reference,
            method_study=method_study,
            years=years,
            train_start=train_start,
            embargo_days=embargo_days,
            max_workers=max_workers,
            publication_gates=publication_gates,
            history=history,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annual_study": {
                "id": self.annual_id,
                "method_study": self.method_study_reference,
                "method_study_hash": self.method_study.spec_hash,
                "years": list(self.years),
                "train_start": self.train_start,
                "embargo_days": self.embargo_days,
                "max_workers": self.max_workers,
            },
            "publication_gates": [gate.to_dict() for gate in self.publication_gates],
            "history": [item.to_dict() for item in self.history],
        }

    @property
    def spec_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def registry(self) -> dict[str, Any]:
        candidates = [
            {
                "id": trial.trial_id,
                "purpose": trial.purpose,
                "role": "current_candidate",
                "included_in_selection": True,
            }
            for trial in self.method_study.trials
        ]
        history = [item.to_dict() for item in self.history]
        return {
            "current_candidates": candidates,
            "historical_experiments": history,
            "declared_current_candidate_count": len(candidates),
            "declared_historical_experiment_count": len(history),
            "declared_total_experiment_count": len(candidates) + len(history),
        }


@dataclass(frozen=True)
class AnnualFold:
    year: int
    train_start: str
    train_end: str
    embargo_start: str
    last_mature_label_date: str
    evaluation_start: str
    evaluation_end: str
    first_decision_date: str
    last_decision_date: str
    open_session_count: int
    decision_count: int

    @property
    def fold_id(self) -> str:
        return str(self.year)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "year": self.year,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "embargo_start": self.embargo_start,
            "last_mature_label_date": self.last_mature_label_date,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "first_decision_date": self.first_decision_date,
            "last_decision_date": self.last_decision_date,
            "open_session_count": self.open_session_count,
            "decision_count": self.decision_count,
        }


@dataclass(frozen=True)
class AnnualFoldExecution:
    run_id: str
    config_hash: str
    method_hash: str
    cost_hash: str
    summary: dict[str, Any]
    artifact_fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class AnnualTrialAggregate:
    summary: dict[str, Any]
    artifact_fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class AnnualStudyResult:
    annual_id: str
    status: str
    annual_root: Path
    task_count: int
    completed_task_count: int
    failed_task_count: int
    pending_task_count: int
    completed_trial_count: int
    skipped_task_count: int
    selected_trial_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annual_id": self.annual_id,
            "status": self.status,
            "annual_root": str(self.annual_root),
            "task_count": self.task_count,
            "completed_task_count": self.completed_task_count,
            "failed_task_count": self.failed_task_count,
            "pending_task_count": self.pending_task_count,
            "completed_trial_count": self.completed_trial_count,
            "skipped_task_count": self.skipped_task_count,
            "selected_trial_id": self.selected_trial_id,
        }


AnnualFoldExecutor = Callable[[StudyTrial, AnnualFold, Path], AnnualFoldExecution]
AnnualTrialAggregator = Callable[
    [StudyTrial, Sequence[Mapping[str, Any]], Path],
    AnnualTrialAggregate,
]


def build_annual_folds(
    spec: AnnualStudySpec,
    base_config: AppConfig,
    open_dates: Sequence[str],
) -> tuple[AnnualFold, ...]:
    dates = tuple(str(value) for value in open_dates)
    if dates != tuple(sorted(set(dates))):
        raise ConfigurationError("Annual fold open dates must be sorted and unique")
    if spec.train_start not in dates:
        raise ConfigurationError(
            f"annual_study.train_start is not an open date: {spec.train_start}"
        )
    if spec.train_start < base_config.workflow.feature_start:
        raise ConfigurationError(
            "annual_study.train_start cannot predate workflow.feature_start"
        )
    if spec.embargo_days < base_config.features.label_horizon:
        raise ConfigurationError(
            "Annual embargo must be at least the forward-label horizon"
        )
    decision_dates = select_rebalance_dates(
        list(dates),
        start_date=base_config.workflow.feature_start,
        end_date=base_config.dates.end,
        every=base_config.research.rebalance_every,
    )
    folds: list[AnnualFold] = []
    for year in spec.years:
        prefix = str(year)
        year_open = [date for date in dates if date.startswith(prefix)]
        year_decisions = [date for date in decision_dates if date.startswith(prefix)]
        if not year_open or not year_decisions:
            raise ConfigurationError(f"Annual fold {year} has no open or decision dates")
        evaluation_start = year_decisions[0]
        evaluation_end = year_open[-1]
        earlier_decisions = [date for date in decision_dates if date < evaluation_start]
        if not earlier_decisions:
            raise ConfigurationError(f"Annual fold {year} has no prior training decision")
        train_end = earlier_decisions[-1]
        start_position = dates.index(evaluation_start)
        mature_position = start_position - spec.embargo_days - 1
        embargo_position = start_position - spec.embargo_days
        if mature_position < 0 or embargo_position < 0:
            raise ConfigurationError(
                f"Annual fold {year} lacks history for its embargo"
            )
        folds.append(
            AnnualFold(
                year=year,
                train_start=spec.train_start,
                train_end=train_end,
                embargo_start=dates[embargo_position],
                last_mature_label_date=dates[mature_position],
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                first_decision_date=year_decisions[0],
                last_decision_date=year_decisions[-1],
                open_session_count=len(year_open),
                decision_count=len(year_decisions),
            )
        )
    for previous, current in zip(folds, folds[1:], strict=False):
        if previous.evaluation_end >= current.evaluation_start:
            raise ConfigurationError("Annual fold evaluation windows overlap")
    return tuple(folds)


def resolve_annual_fold_config(
    method_config: AppConfig,
    *,
    annual_id: str,
    trial_id: str,
    fold: AnnualFold,
    embargo_days: int,
) -> AppConfig:
    protocol_id = _simple_name(
        f"{method_config.experiment.protocol_id}-{annual_id}-{trial_id}-{fold.fold_id}",
        "annual fold protocol_id",
    )
    experiment = ExperimentSettings(
        stage="validation",
        protocol_id=protocol_id,
        train_start=fold.train_start,
        train_end=fold.train_end,
        validation_start=fold.evaluation_start,
        validation_end=fold.evaluation_end,
        test_start=f"{fold.year + 1}0101",
        test_end=f"{fold.year + 1}0101",
        embargo_days=embargo_days,
        allow_frozen_test=False,
    )
    result = replace(method_config, experiment=experiment)
    result.validate()
    return result


def annual_feature_contract(config: AppConfig) -> dict[str, Any]:
    """Return inputs that must match before a prepared factor layer is shared."""

    return {
        "dataset": config.paths.dataset,
        "dates": {
            "raw_start": config.dates.raw_start,
            "backtest_start": config.dates.backtest_start,
            "end": config.dates.end,
        },
        "feature_start": config.workflow.feature_start,
        "rebalance_every": config.research.rebalance_every,
        "factor_names": list(config.workflow.factor_names),
        "feature_provider": {
            "name": config.workflow.feature_provider.name,
            "params": config.workflow.feature_provider.params,
        },
        "feature_processing": {
            "label_horizon": config.features.label_horizon,
            "min_factor_coverage": config.features.min_factor_coverage,
            "mad_clip": config.features.mad_clip,
            "industry_coverage_threshold": (
                config.features.industry_coverage_threshold
            ),
            "industry_transition_date": config.features.industry_transition_date,
        },
    }


def annual_cost_contract(config: AppConfig) -> dict[str, Any]:
    return {
        "linear_cost_bps": config.research.linear_cost_bps,
        "stamp_duty_change_date": config.research.stamp_duty_change_date,
        "stamp_duty_before": config.research.stamp_duty_before,
        "stamp_duty_after": config.research.stamp_duty_after,
        "portfolio_aum_cny": config.optimizer.portfolio_aum_cny,
        "liquidity_enabled": config.optimizer.liquidity_enabled,
        "adv_lookback": config.optimizer.adv_lookback,
        "min_adv_observations": config.optimizer.min_adv_observations,
        "max_adv_participation": config.optimizer.max_adv_participation,
        "impact_bps_at_max_participation": (
            config.optimizer.impact_bps_at_max_participation
        ),
    }


def annual_study_plan(
    config_path: str | Path,
    *,
    open_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = AnnualStudySpec.from_yaml(config_path)
    base = AppConfig.from_yaml(spec.method_study.base_config_path)
    payload: dict[str, Any] = {
        "annual_id": spec.annual_id,
        "annual_spec_hash": spec.spec_hash,
        "method_study_id": spec.method_study.study_id,
        "method_study_hash": spec.method_study.spec_hash,
        "base_config_path": str(spec.method_study.base_config_path),
        "years": list(spec.years),
        "trial_ids": [trial.trial_id for trial in spec.method_study.trials],
        "task_count": len(spec.years) * len(spec.method_study.trials),
        "max_workers": spec.max_workers,
        "publication_gates": [gate.to_dict() for gate in spec.publication_gates],
        "registry": spec.registry(),
    }
    if open_dates is not None:
        folds = build_annual_folds(spec, base, open_dates)
        payload["folds"] = [fold.to_dict() for fold in folds]
        payload["fold_plan_hash"] = _fold_plan_hash(folds)
    return payload


class AnnualStudyRunner:
    """Run and resume a deterministic candidate-by-year research matrix."""

    def __init__(
        self,
        spec: AnnualStudySpec,
        *,
        folds: Sequence[AnnualFold],
        annual_root: Path,
        context: StudyContext,
        executor: AnnualFoldExecutor,
        aggregator: AnnualTrialAggregator,
        max_workers: int | None = None,
        selected_years: Sequence[int] | None = None,
        selected_trials: Sequence[str] | None = None,
    ) -> None:
        self.spec = spec
        self.folds = tuple(folds)
        self.annual_root = annual_root.resolve()
        self.context = context
        self.executor = executor
        self.aggregator = aggregator
        self.max_workers = spec.max_workers if max_workers is None else max_workers
        if not 1 <= self.max_workers <= 8:
            raise ConfigurationError("Annual runner max_workers must be between 1 and 8")
        known_years = {fold.year for fold in self.folds}
        known_trials = {trial.trial_id for trial in spec.method_study.trials}
        self.selected_years = (
            set(int(value) for value in selected_years)
            if selected_years is not None
            else known_years
        )
        self.selected_trials = (
            set(str(value) for value in selected_trials)
            if selected_trials is not None
            else known_trials
        )
        unknown_years = sorted(self.selected_years.difference(known_years))
        unknown_trials = sorted(self.selected_trials.difference(known_trials))
        if unknown_years or unknown_trials:
            raise ConfigurationError(
                "Annual task filter contains unknown values: "
                f"years={unknown_years}, trials={unknown_trials}"
            )

    def run(self) -> AnnualStudyResult:
        parent = self._initialize_manifest()
        runnable: list[tuple[StudyTrial, AnnualFold, int, Path]] = []
        skipped = 0
        for trial in self.spec.method_study.trials:
            for fold in self.folds:
                path = self._task_manifest_path(trial, fold)
                existing = self._read_json(path)
                task_hash = self._task_hash(trial, fold)
                if existing:
                    self._assert_task_identity(existing, trial, fold, task_hash)
                    if existing.get("status") == "completed":
                        self._verify_completed_manifest_artifacts(existing)
                        skipped += int(
                            trial.trial_id in self.selected_trials
                            and fold.year in self.selected_years
                        )
                        continue
                if (
                    trial.trial_id not in self.selected_trials
                    or fold.year not in self.selected_years
                ):
                    continue
                attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
                attempt_root = path.parent / f"attempt-{attempts:03d}"
                runnable.append((trial, fold, attempts, attempt_root))

        LOGGER.info(
            "annual=%s | runnable_tasks=%d | skipped_tasks=%d | workers=%d",
            self.spec.annual_id,
            len(runnable),
            skipped,
            self.max_workers,
        )

        if self.max_workers == 1:
            for trial, fold, attempts, attempt_root in runnable:
                self._execute_task(trial, fold, attempts, attempt_root)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self._execute_task,
                        trial,
                        fold,
                        attempts,
                        attempt_root,
                    ): (trial.trial_id, fold.year)
                    for trial, fold, attempts, attempt_root in runnable
                }
                for future in as_completed(futures):
                    future.result()

        manifests = self._all_task_manifests()
        self._assert_completed_contracts(manifests)
        write_parquet_atomic(
            self._task_table(manifests),
            self.annual_root / "fold-tasks.parquet",
        )
        aggregate_manifests = self._aggregate_trials(manifests)
        write_parquet_atomic(
            self._aggregate_table(aggregate_manifests),
            self.annual_root / "trial-aggregates.parquet",
        )
        gates = self._publication_gate_results(aggregate_manifests)
        write_json_atomic(gates, self.annual_root / "publication-gates.json")
        selection = select_study_candidates(
            self.spec.method_study,
            aggregate_manifests,
        )
        write_json_atomic(selection, self.annual_root / "selection.json")
        selected_trial_id = selection["selected_trial_id"]

        completed_tasks = sum(
            manifest.get("status") == "completed" for manifest in manifests
        )
        failed_tasks = sum(
            manifest.get("status") == "failed" for manifest in manifests
        )
        total_tasks = len(self.folds) * len(self.spec.method_study.trials)
        pending_tasks = total_tasks - completed_tasks - failed_tasks
        completed_trials = sum(
            manifest.get("status") == "completed"
            for manifest in aggregate_manifests
        )
        if completed_tasks == total_tasks and completed_trials == len(
            self.spec.method_study.trials
        ):
            if selected_trial_id is None:
                status = "completed_without_selection"
            elif selected_trial_id not in gates["passed"]:
                status = "completed_without_publishable_candidate"
            else:
                status = "completed"
        elif failed_tasks:
            status = "incomplete_with_failures"
        else:
            status = "incomplete"
        final = {
            **parent,
            "status": status,
            "updated_at": utc_now(),
            "task_count": total_tasks,
            "completed_task_count": completed_tasks,
            "failed_task_count": failed_tasks,
            "pending_task_count": pending_tasks,
            "completed_trial_count": completed_trials,
            "selected_trial_id": selected_trial_id,
            "artifacts": {
                "fold_tasks": "fold-tasks.parquet",
                "trial_aggregates": "trial-aggregates.parquet",
                "publication_gates": "publication-gates.json",
                "selection": "selection.json",
                "fold_root": "folds",
                "aggregate_root": "aggregates",
            },
        }
        write_json_atomic(final, self.annual_root / "annual-study-manifest.json")
        return AnnualStudyResult(
            annual_id=self.spec.annual_id,
            status=status,
            annual_root=self.annual_root,
            task_count=total_tasks,
            completed_task_count=completed_tasks,
            failed_task_count=failed_tasks,
            pending_task_count=pending_tasks,
            completed_trial_count=completed_trials,
            skipped_task_count=skipped,
            selected_trial_id=selected_trial_id,
        )

    def _execute_task(
        self,
        trial: StudyTrial,
        fold: AnnualFold,
        attempts: int,
        attempt_root: Path,
    ) -> None:
        started = time.perf_counter()
        path = self._task_manifest_path(trial, fold)
        running = {
            "schema_version": 1,
            "annual_id": self.spec.annual_id,
            "method_study_id": self.spec.method_study.study_id,
            "trial": trial.to_dict(),
            "fold": fold.to_dict(),
            "task_hash": self._task_hash(trial, fold),
            **self.context.identity(),
            "status": "running",
            "attempts": attempts,
            "started_at": utc_now(),
            "completed_at": None,
            "artifact_root": attempt_root.relative_to(self.annual_root).as_posix(),
            "run_id": None,
            "resolved_config_hash": None,
            "method_hash": None,
            "cost_hash": None,
            "summary": None,
            "artifact_fingerprints": None,
            "error": None,
        }
        write_json_atomic(running, path)
        LOGGER.info(
            "annual=%s | trial=%s | year=%d | status=running | attempt=%d",
            self.spec.annual_id,
            trial.trial_id,
            fold.year,
            attempts,
        )
        try:
            execution = self.executor(trial, fold, attempt_root)
            self._verify_artifacts(attempt_root, execution.artifact_fingerprints)
        except Exception as exc:  # noqa: BLE001 - task failures are research data
            LOGGER.exception(
                "annual=%s | trial=%s | year=%d | status=failed | elapsed=%.1fs",
                self.spec.annual_id,
                trial.trial_id,
                fold.year,
                time.perf_counter() - started,
            )
            write_json_atomic(
                {
                    **running,
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
                path,
            )
            return
        write_json_atomic(
            {
                **running,
                "status": "completed",
                "completed_at": utc_now(),
                "run_id": execution.run_id,
                "resolved_config_hash": execution.config_hash,
                "method_hash": execution.method_hash,
                "cost_hash": execution.cost_hash,
                "summary": execution.summary,
                "artifact_fingerprints": dict(
                    sorted(execution.artifact_fingerprints.items())
                ),
            },
            path,
        )
        LOGGER.info(
            "annual=%s | trial=%s | year=%d | status=completed | elapsed=%.1fs",
            self.spec.annual_id,
            trial.trial_id,
            fold.year,
            time.perf_counter() - started,
        )

    def _aggregate_trials(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for trial in self.spec.method_study.trials:
            selected = [
                manifest
                for manifest in manifests
                if _trial_id(manifest) == trial.trial_id
            ]
            path = self._aggregate_manifest_path(trial)
            if len(selected) != len(self.folds) or any(
                manifest.get("status") != "completed" for manifest in selected
            ):
                pending = {
                    "schema_version": 1,
                    "annual_id": self.spec.annual_id,
                    "trial": trial.to_dict(),
                    "status": "pending",
                    "fold_count": len(selected),
                    "completed_fold_count": sum(
                        manifest.get("status") == "completed"
                        for manifest in selected
                    ),
                    "task_set_hash": self._task_set_hash(selected),
                    "summary": None,
                    "artifact_fingerprints": None,
                    "error": None,
                }
                write_json_atomic(pending, path)
                results.append(pending)
                continue
            selected = sorted(selected, key=_fold_year)
            task_set_hash = self._task_set_hash(selected)
            existing = self._read_json(path)
            if (
                existing.get("status") == "completed"
                and existing.get("task_set_hash") == task_set_hash
            ):
                fingerprints = existing.get("artifact_fingerprints")
                if not isinstance(fingerprints, Mapping):
                    raise ConfigurationError(
                        f"Aggregate {trial.trial_id!r} lacks artifact fingerprints"
                    )
                self._verify_artifacts(path.parent, fingerprints)
                results.append(existing)
                continue
            aggregate_root = path.parent
            running = {
                "schema_version": 1,
                "annual_id": self.spec.annual_id,
                "trial": trial.to_dict(),
                "status": "running",
                "fold_count": len(selected),
                "completed_fold_count": len(selected),
                "task_set_hash": task_set_hash,
                "started_at": utc_now(),
                "completed_at": None,
                "summary": None,
                "artifact_fingerprints": None,
                "error": None,
            }
            write_json_atomic(running, path)
            try:
                aggregate = self.aggregator(trial, selected, aggregate_root)
                self._verify_artifacts(
                    aggregate_root,
                    aggregate.artifact_fingerprints,
                )
            except Exception as exc:  # noqa: BLE001 - aggregate failures are auditable
                failed = {
                    **running,
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                write_json_atomic(failed, path)
                results.append(failed)
                continue
            completed = {
                **running,
                "status": "completed",
                "completed_at": utc_now(),
                "summary": aggregate.summary,
                "artifact_fingerprints": dict(
                    sorted(aggregate.artifact_fingerprints.items())
                ),
            }
            write_json_atomic(completed, path)
            results.append(completed)
        return results

    def _initialize_manifest(self) -> dict[str, Any]:
        path = self.annual_root / "annual-study-manifest.json"
        existing = self._read_json(path)
        identity = {
            "annual_spec_hash": self.spec.spec_hash,
            "method_study_hash": self.spec.method_study.spec_hash,
            "fold_plan_hash": _fold_plan_hash(self.folds),
            **self.context.identity(),
        }
        if existing:
            changed = sorted(
                key for key, value in identity.items() if existing.get(key) != value
            )
            if changed:
                raise ConfigurationError(
                    "Existing annual study identity differs; create a new annual id. "
                    f"Changed fields: {changed}"
                )
            return existing
        manifest = {
            "schema_version": 1,
            "annual_id": self.spec.annual_id,
            "annual_config_path": str(self.spec.config_path),
            "method_study_id": self.spec.method_study.study_id,
            "method_study_path": str(self.spec.method_study_path),
            **identity,
            "data_fingerprints": dict(sorted(self.context.data_fingerprints.items())),
            "git": dict(self.context.git),
            "folds": [fold.to_dict() for fold in self.folds],
            "publication_gates": [
                gate.to_dict() for gate in self.spec.publication_gates
            ],
            "registry": self.spec.registry(),
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        write_json_atomic(manifest, path)
        return manifest

    def _all_task_manifests(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for trial in self.spec.method_study.trials:
            for fold in self.folds:
                manifest = self._read_json(self._task_manifest_path(trial, fold))
                if manifest:
                    results.append(manifest)
        return results

    def _assert_completed_contracts(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> None:
        completed = [item for item in manifests if item.get("status") == "completed"]
        cost_hashes = {str(item.get("cost_hash", "")) for item in completed}
        if "" in cost_hashes or len(cost_hashes) > 1:
            raise ConfigurationError(
                "Completed annual folds do not share one frozen cost contract"
            )
        for trial in self.spec.method_study.trials:
            method_hashes = {
                str(item.get("method_hash", ""))
                for item in completed
                if _trial_id(item) == trial.trial_id
            }
            if "" in method_hashes or len(method_hashes) > 1:
                raise ConfigurationError(
                    f"Trial {trial.trial_id!r} changed method settings across folds"
                )

    def _task_table(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        indexed = {
            (_trial_id(item), _fold_year(item)): item for item in manifests
        }
        rows: list[dict[str, Any]] = []
        for trial in self.spec.method_study.trials:
            for fold in self.folds:
                item = indexed.get((trial.trial_id, fold.year), {})
                error = item.get("error")
                rows.append(
                    {
                        "annual_id": self.spec.annual_id,
                        "trial_id": trial.trial_id,
                        "fold_year": fold.year,
                        "status": item.get("status", "pending"),
                        "attempts": int(item.get("attempts", 0)),
                        "run_id": item.get("run_id"),
                        "task_hash": item.get("task_hash"),
                        "resolved_config_hash": item.get("resolved_config_hash"),
                        "method_hash": item.get("method_hash"),
                        "cost_hash": item.get("cost_hash"),
                        "data_snapshot_hash": item.get("data_snapshot_hash"),
                        "artifact_root": item.get("artifact_root"),
                        "train_end": fold.train_end,
                        "last_mature_label_date": fold.last_mature_label_date,
                        "embargo_start": fold.embargo_start,
                        "evaluation_start": fold.evaluation_start,
                        "evaluation_end": fold.evaluation_end,
                        "summary_json": (
                            canonical_json(item.get("summary"))
                            if item.get("summary") is not None
                            else None
                        ),
                        "error_type": (
                            error.get("type") if isinstance(error, Mapping) else None
                        ),
                        "error_message": (
                            error.get("message")
                            if isinstance(error, Mapping)
                            else None
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _aggregate_table(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in manifests:
            trial = _mapping(item.get("trial", {}), "aggregate trial")
            error = item.get("error")
            rows.append(
                {
                    "annual_id": self.spec.annual_id,
                    "trial_id": str(trial.get("id", "")),
                    "status": item.get("status"),
                    "fold_count": int(item.get("fold_count", 0)),
                    "completed_fold_count": int(
                        item.get("completed_fold_count", 0)
                    ),
                    "task_set_hash": item.get("task_set_hash"),
                    "summary_json": (
                        canonical_json(item.get("summary"))
                        if item.get("summary") is not None
                        else None
                    ),
                    "error_type": (
                        error.get("type") if isinstance(error, Mapping) else None
                    ),
                    "error_message": (
                        error.get("message") if isinstance(error, Mapping) else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _publication_gate_results(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for item in manifests:
            trial = _mapping(item.get("trial", {}), "aggregate trial")
            trial_id = str(trial.get("id", ""))
            failures = (
                _gate_failures(item.get("summary"), self.spec.publication_gates)
                if item.get("status") == "completed"
                else [f"aggregate_status={item.get('status')}" ]
            )
            rows.append(
                {
                    "trial_id": trial_id,
                    "passed": not failures,
                    "failures": failures,
                }
            )
        return {
            "schema_version": 1,
            "annual_id": self.spec.annual_id,
            "created_at": utc_now(),
            "rules": [gate.to_dict() for gate in self.spec.publication_gates],
            "trials": rows,
            "passed": [item["trial_id"] for item in rows if item["passed"]],
            "failed": [item["trial_id"] for item in rows if not item["passed"]],
        }

    def _task_manifest_path(self, trial: StudyTrial, fold: AnnualFold) -> Path:
        return (
            self.annual_root
            / "folds"
            / fold.fold_id
            / trial.trial_id
            / "fold-manifest.json"
        )

    def _aggregate_manifest_path(self, trial: StudyTrial) -> Path:
        return self.annual_root / "aggregates" / trial.trial_id / "aggregate-manifest.json"

    def _task_hash(self, trial: StudyTrial, fold: AnnualFold) -> str:
        return sha256_text(
            canonical_json(
                {
                    "annual_spec_hash": self.spec.spec_hash,
                    "method_study_hash": self.spec.method_study.spec_hash,
                    "base_config_hash": self.context.base_config_hash,
                    "trial": trial.to_dict(),
                    "fold": fold.to_dict(),
                }
            )
        )

    def _task_set_hash(self, manifests: Sequence[Mapping[str, Any]]) -> str:
        payload = [
            {
                "task_hash": item.get("task_hash"),
                "run_id": item.get("run_id"),
                "resolved_config_hash": item.get("resolved_config_hash"),
                "artifact_fingerprints": item.get("artifact_fingerprints"),
            }
            for item in sorted(manifests, key=_fold_year)
        ]
        return sha256_text(canonical_json(payload))

    def _assert_task_identity(
        self,
        existing: Mapping[str, Any],
        trial: StudyTrial,
        fold: AnnualFold,
        task_hash: str,
    ) -> None:
        expected = {
            "task_hash": task_hash,
            **self.context.identity(),
        }
        changed = sorted(
            key for key, value in expected.items() if existing.get(key) != value
        )
        if changed:
            raise ConfigurationError(
                f"Annual task {trial.trial_id}/{fold.fold_id} cannot resume because "
                f"identity fields changed: {changed}"
            )

    def _verify_completed_manifest_artifacts(
        self,
        manifest: Mapping[str, Any],
    ) -> None:
        reference = manifest.get("artifact_root")
        fingerprints = manifest.get("artifact_fingerprints")
        if not isinstance(reference, str) or not reference:
            raise ConfigurationError("Completed annual task lacks artifact_root")
        if not isinstance(fingerprints, Mapping):
            raise ConfigurationError(
                "Completed annual task lacks artifact fingerprints"
            )
        root = (self.annual_root / reference).resolve()
        if not root.is_relative_to(self.annual_root):
            raise ConfigurationError("Completed annual task artifact_root escapes parent")
        self._verify_artifacts(root, fingerprints)

    @staticmethod
    def _verify_artifacts(root: Path, fingerprints: Mapping[str, str]) -> None:
        if not fingerprints:
            raise ConfigurationError("Annual execution returned no artifacts")
        resolved_root = root.resolve()
        for reference, expected in fingerprints.items():
            path = (root / reference).resolve()
            if not path.is_relative_to(resolved_root):
                raise ConfigurationError(
                    f"Annual artifact escapes its task root: {reference}"
                )
            if not path.is_file():
                raise ConfigurationError(f"Annual artifact does not exist: {path}")
            if sha256_file(path) != expected:
                raise ConfigurationError(f"Annual artifact hash differs: {path}")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ConfigurationError(f"Annual JSON artifact must be an object: {path}")
        return value


def _fold_plan_hash(folds: Sequence[AnnualFold]) -> str:
    return sha256_text(canonical_json([fold.to_dict() for fold in folds]))


def _trial_id(manifest: Mapping[str, Any]) -> str:
    trial = manifest.get("trial")
    return str(trial.get("id", "")) if isinstance(trial, Mapping) else ""


def _fold_year(manifest: Mapping[str, Any]) -> int:
    fold = manifest.get("fold")
    return int(fold.get("year", 0)) if isinstance(fold, Mapping) else 0


def _historical_experiment(value: Any, position: int) -> HistoricalExperiment:
    section = f"history[{position}]"
    item = _mapping(value, section)
    _reject_unknown(
        item,
        {"id", "role", "status", "included_in_selection", "note"},
        section,
    )
    experiment_id = _simple_name(_required(item, "id", section), f"{section}.id")
    role = str(_required(item, "role", section)).strip()
    status = str(_required(item, "status", section)).strip()
    note = str(item.get("note", "")).strip()
    included = item.get("included_in_selection", False)
    if not role or not status:
        raise ConfigurationError(f"{section}.role and status cannot be empty")
    if not isinstance(included, bool):
        raise ConfigurationError(f"{section}.included_in_selection must be boolean")
    return HistoricalExperiment(
        experiment_id=experiment_id,
        role=role,
        status=status,
        included_in_selection=included,
        note=note,
    )


def _gate_rule(value: Any, position: int) -> GateRule:
    section = f"publication_gates[{position}]"
    item = _mapping(value, section)
    _reject_unknown(item, {"path", "operator", "value"}, section)
    path = str(_required(item, "path", section)).strip()
    if not path or any(not part for part in path.split(".")):
        raise ConfigurationError(f"{section}.path must be a non-empty dotted path")
    operator = str(_required(item, "operator", section))
    if operator not in _OPERATORS:
        raise ConfigurationError(
            f"{section}.operator must be one of {sorted(_OPERATORS)}"
        )
    return GateRule(
        path=path,
        operator=operator,
        value=_required(item, "value", section),
    )


def _gate_failures(summary: Any, gates: Sequence[GateRule]) -> list[str]:
    failures: list[str] = []
    for gate in gates:
        observed = _resolve_path(summary, gate.path)
        if observed is None:
            failures.append(f"missing_gate_metric={gate.path}")
            continue
        try:
            passed = _compare(observed, gate.operator, gate.value)
        except (TypeError, ValueError):
            failures.append(f"invalid_gate_metric={gate.path}")
            continue
        if not passed:
            failures.append(
                f"gate_failed={gate.path}{gate.operator}{gate.value!r};"
                f"observed={observed!r}"
            )
    return failures


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "==":
        return bool(left == right)
    if operator == "!=":
        return bool(left != right)
    if operator == ">":
        return bool(left > right)
    if operator == ">=":
        return bool(left >= right)
    if operator == "<":
        return bool(left < right)
    if operator == "<=":
        return bool(left <= right)
    raise ValueError(f"Unsupported operator: {operator}")


def _date(value: Any, section: str) -> str:
    date = str(value).strip()
    if len(date) != 8 or not date.isdigit():
        raise ConfigurationError(f"{section} must use YYYYMMDD")
    return date


def _simple_name(value: Any, section: str) -> str:
    name = str(value).strip()
    if not name or not name[0].isalnum() or any(
        not (character.isalnum() or character in "-_.") for character in name
    ):
        raise ConfigurationError(
            f"{section} must contain only letters, digits, '-', '_' or '.'"
        )
    return name


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{section} must be a mapping")
    return dict(value)


def _sequence(value: Any, section: str) -> list[Any]:
    if not isinstance(value, list | tuple):
        raise ConfigurationError(f"{section} must be a sequence")
    return list(value)


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required key {section}.{key}")
    return mapping[key]


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ConfigurationError(f"Unknown {section} keys: {unknown}")
