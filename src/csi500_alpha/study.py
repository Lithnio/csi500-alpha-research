from __future__ import annotations

import json
import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from csi500_alpha.config import AppConfig, ComponentSettings
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.utils import canonical_json, sha256_text, utc_now

_DIRECTIONS = {"maximize", "minimize"}
_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}


@dataclass(frozen=True)
class MetricRule:
    path: str
    direction: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "direction": self.direction}


@dataclass(frozen=True)
class GateRule:
    path: str
    operator: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass(frozen=True)
class SelectionRule:
    primary: MetricRule
    tie_breakers: tuple[MetricRule, ...]
    gates: tuple[GateRule, ...]
    primary_tolerance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "tie_breakers": [rule.to_dict() for rule in self.tie_breakers],
            "gates": [gate.to_dict() for gate in self.gates],
            "primary_tolerance": self.primary_tolerance,
            "deterministic_final_tie_breaker": "trial_id_ascending",
        }


@dataclass(frozen=True)
class StudyTrial:
    trial_id: str
    purpose: str
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.trial_id,
            "purpose": self.purpose,
            "overrides": self.overrides,
        }


@dataclass(frozen=True)
class StudySpec:
    config_path: Path
    study_id: str
    base_config_path: Path
    base_config_reference: str
    max_trials: int
    selection: SelectionRule
    trials: tuple[StudyTrial, ...]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> StudySpec:
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigurationError(f"Study configuration does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        root = _mapping(raw, "study config")
        _reject_unknown(root, {"study", "selection", "trials"}, "study config")

        study = _mapping(_required(root, "study", "study config"), "study")
        _reject_unknown(study, {"id", "base_config", "max_trials"}, "study")
        study_id = _simple_name(_required(study, "id", "study"), "study.id")
        base_reference = str(_required(study, "base_config", "study"))
        base_path = (path.parent / base_reference).resolve()
        if not base_path.is_file():
            raise ConfigurationError(
                f"study.base_config does not exist: {base_path}"
            )
        max_trials = int(study.get("max_trials", 12))
        if not 1 <= max_trials <= 50:
            raise ConfigurationError("study.max_trials must be between 1 and 50")

        selection = _selection_rule(
            _mapping(_required(root, "selection", "study config"), "selection")
        )
        raw_trials = _sequence(_required(root, "trials", "study config"), "trials")
        if not raw_trials:
            raise ConfigurationError("Study requires at least one trial")
        if len(raw_trials) > max_trials:
            raise ConfigurationError(
                f"Study declares {len(raw_trials)} trials but max_trials={max_trials}"
            )
        trials = tuple(_trial(value, position) for position, value in enumerate(raw_trials))
        trial_ids = [trial.trial_id for trial in trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ConfigurationError("Study trial ids must be unique")
        return cls(
            config_path=path,
            study_id=study_id,
            base_config_path=base_path,
            base_config_reference=base_reference,
            max_trials=max_trials,
            selection=selection,
            trials=trials,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": {
                "id": self.study_id,
                "base_config": self.base_config_reference,
                "max_trials": self.max_trials,
            },
            "selection": self.selection.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @property
    def spec_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass(frozen=True)
class StudyContext:
    base_config_hash: str
    source_hash: str
    data_snapshot_hash: str
    data_fingerprints: Mapping[str, str]
    git: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "base_config_hash": self.base_config_hash,
            "source_hash": self.source_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "data_fingerprints": dict(sorted(self.data_fingerprints.items())),
            "git": dict(self.git),
        }


@dataclass(frozen=True)
class TrialExecution:
    run_id: str
    config_hash: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class StudyResult:
    study_id: str
    status: str
    study_root: Path
    selected_trial_id: str | None
    trial_count: int
    completed_count: int
    failed_count: int
    skipped_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "status": self.status,
            "study_root": str(self.study_root),
            "selected_trial_id": self.selected_trial_id,
            "trial_count": self.trial_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
        }


TrialExecutor = Callable[[StudyTrial, Path], TrialExecution]


def resolve_trial_config(
    base: AppConfig,
    *,
    study_id: str,
    trial: StudyTrial,
) -> AppConfig:
    """Apply bounded method overrides while keeping data and sample dates fixed."""

    allowed_sections = {"workflow", "research", "risk", "optimizer", "features"}
    _reject_unknown(trial.overrides, allowed_sections, f"trial {trial.trial_id} overrides")
    config = base
    for section in ("research", "risk", "optimizer", "features"):
        if section not in trial.overrides:
            continue
        current = getattr(config, section)
        values = _mapping(
            trial.overrides[section],
            f"trial {trial.trial_id} overrides.{section}",
        )
        allowed = {field.name for field in fields(current)}
        _reject_unknown(values, allowed, f"trial {trial.trial_id} overrides.{section}")
        if section == "optimizer" and "solvers" in values:
            values["solvers"] = tuple(
                str(item)
                for item in _sequence(
                    values["solvers"],
                    f"trial {trial.trial_id} overrides.optimizer.solvers",
                )
            )
        config = replace(config, **{section: replace(current, **values)})

    if "workflow" in trial.overrides:
        values = _mapping(
            trial.overrides["workflow"],
            f"trial {trial.trial_id} overrides.workflow",
        )
        allowed = {
            "refit_every",
            "factors",
            "feature_provider",
            "selector",
            "model",
            "calibrator",
        }
        _reject_unknown(values, allowed, f"trial {trial.trial_id} overrides.workflow")
        workflow_values: dict[str, Any] = {}
        if "refit_every" in values:
            workflow_values["refit_every"] = int(values["refit_every"])
        if "factors" in values:
            workflow_values["factor_names"] = tuple(
                str(item)
                for item in _sequence(
                    values["factors"],
                    f"trial {trial.trial_id} overrides.workflow.factors",
                )
            )
        for component_name in (
            "feature_provider",
            "selector",
            "model",
            "calibrator",
        ):
            if component_name not in values:
                continue
            current_component = getattr(config.workflow, component_name)
            workflow_values[component_name] = _component_override(
                values[component_name],
                current_component,
                f"trial {trial.trial_id} overrides.workflow.{component_name}",
            )
        config = replace(
            config,
            workflow=replace(config.workflow, **workflow_values),
        )

    protocol_id = _simple_name(
        f"{base.experiment.protocol_id}-{study_id}-{trial.trial_id}",
        "resolved trial protocol_id",
    )
    config = replace(
        config,
        experiment=replace(config.experiment, protocol_id=protocol_id),
    )
    config.validate()
    return config


class StudyRunner:
    """Run, resume and select a bounded set of auditable research trials."""

    def __init__(
        self,
        spec: StudySpec,
        *,
        study_root: Path,
        context: StudyContext,
        executor: TrialExecutor,
    ) -> None:
        self.spec = spec
        self.study_root = study_root.resolve()
        self.context = context
        self.executor = executor

    def run(self) -> StudyResult:
        manifest = self._initialize_study_manifest()
        skipped = 0
        for trial in self.spec.trials:
            trial_path = self._trial_manifest_path(trial)
            existing = self._read_json(trial_path)
            trial_hash = self._trial_hash(trial)
            if existing:
                self._assert_trial_identity(existing, trial, trial_hash)
                if existing.get("status") == "completed":
                    skipped += 1
                    continue
            attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
            attempt_root = trial_path.parent / f"attempt-{attempts:03d}"
            started_at = utc_now()
            running = {
                "schema_version": 1,
                "study_id": self.spec.study_id,
                "trial": trial.to_dict(),
                "trial_hash": trial_hash,
                **self.context.identity(),
                "status": "running",
                "attempts": attempts,
                "started_at": started_at,
                "completed_at": None,
                "artifact_root": attempt_root.relative_to(self.study_root).as_posix(),
                "run_id": None,
                "resolved_config_hash": None,
                "summary": None,
                "error": None,
            }
            write_json_atomic(running, trial_path)
            try:
                execution = self.executor(trial, attempt_root)
            except Exception as exc:  # noqa: BLE001 - trial failures are study data
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
                write_json_atomic(failed, trial_path)
                continue
            completed = {
                **running,
                "status": "completed",
                "completed_at": utc_now(),
                "run_id": execution.run_id,
                "resolved_config_hash": execution.config_hash,
                "summary": execution.summary,
            }
            write_json_atomic(completed, trial_path)

        trial_manifests = [
            self._read_json(self._trial_manifest_path(trial))
            for trial in self.spec.trials
        ]
        write_parquet_atomic(
            self._trial_table(trial_manifests),
            self.study_root / "trials.parquet",
        )
        selection = self._selection(trial_manifests)
        write_json_atomic(selection, self.study_root / "selection.json")
        completed_count = sum(
            item.get("status") == "completed" for item in trial_manifests
        )
        failed_count = sum(item.get("status") == "failed" for item in trial_manifests)
        if selection["selected_trial_id"] is None:
            status = "completed_without_selection"
        elif failed_count:
            status = "completed_with_failures"
        else:
            status = "completed"
        final_manifest = {
            **manifest,
            "status": status,
            "updated_at": utc_now(),
            "trial_count": len(self.spec.trials),
            "completed_count": completed_count,
            "failed_count": failed_count,
            "selected_trial_id": selection["selected_trial_id"],
            "artifacts": {
                "trials": "trials.parquet",
                "selection": "selection.json",
                "trial_root": "trials",
            },
        }
        write_json_atomic(final_manifest, self.study_root / "study-manifest.json")
        return StudyResult(
            study_id=self.spec.study_id,
            status=status,
            study_root=self.study_root,
            selected_trial_id=selection["selected_trial_id"],
            trial_count=len(self.spec.trials),
            completed_count=completed_count,
            failed_count=failed_count,
            skipped_count=skipped,
        )

    def _initialize_study_manifest(self) -> dict[str, Any]:
        path = self.study_root / "study-manifest.json"
        existing = self._read_json(path)
        identity = {
            "spec_hash": self.spec.spec_hash,
            **self.context.identity(),
        }
        if existing:
            changed = sorted(
                key for key, value in identity.items() if existing.get(key) != value
            )
            if changed:
                raise ConfigurationError(
                    "Existing study identity differs; create a new study id. "
                    f"Changed fields: {changed}"
                )
            return existing
        manifest = {
            "schema_version": 1,
            "study_id": self.spec.study_id,
            "study_config_path": str(self.spec.config_path),
            "base_config_path": str(self.spec.base_config_path),
            **identity,
            "data_fingerprints": dict(sorted(self.context.data_fingerprints.items())),
            "git": dict(self.context.git),
            "selection_rule": self.spec.selection.to_dict(),
            "trial_count": len(self.spec.trials),
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        write_json_atomic(manifest, path)
        return manifest

    def _trial_manifest_path(self, trial: StudyTrial) -> Path:
        return self.study_root / "trials" / trial.trial_id / "trial-manifest.json"

    def _trial_hash(self, trial: StudyTrial) -> str:
        payload = {
            "base_config_hash": self.context.base_config_hash,
            "trial": trial.to_dict(),
        }
        return sha256_text(canonical_json(payload))

    def _assert_trial_identity(
        self,
        existing: Mapping[str, Any],
        trial: StudyTrial,
        trial_hash: str,
    ) -> None:
        expected = {"trial_hash": trial_hash, **self.context.identity()}
        changed = sorted(
            key for key, value in expected.items() if existing.get(key) != value
        )
        if changed:
            raise ConfigurationError(
                f"Trial {trial.trial_id!r} cannot resume because identity fields "
                f"changed: {changed}"
            )

    def _trial_table(self, manifests: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        metric_paths = _metric_paths(self.spec.selection)
        rows: list[dict[str, Any]] = []
        for manifest in manifests:
            trial = _mapping(manifest.get("trial", {}), "trial manifest trial")
            summary = manifest.get("summary")
            error = manifest.get("error")
            row: dict[str, Any] = {
                "study_id": self.spec.study_id,
                "trial_id": str(trial.get("id", "")),
                "purpose": str(trial.get("purpose", "")),
                "status": manifest.get("status"),
                "attempts": int(manifest.get("attempts", 0)),
                "run_id": manifest.get("run_id"),
                "trial_hash": manifest.get("trial_hash"),
                "resolved_config_hash": manifest.get("resolved_config_hash"),
                "base_config_hash": manifest.get("base_config_hash"),
                "source_hash": manifest.get("source_hash"),
                "data_snapshot_hash": manifest.get("data_snapshot_hash"),
                "artifact_root": manifest.get("artifact_root"),
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
                "overrides_json": canonical_json(trial.get("overrides", {})),
                "summary_json": canonical_json(summary) if summary is not None else None,
                "error_type": error.get("type") if isinstance(error, dict) else None,
                "error_message": (
                    error.get("message") if isinstance(error, dict) else None
                ),
            }
            for path in metric_paths:
                row[_metric_column(path)] = _resolve_path(summary, path)
            rows.append(row)
        return pd.DataFrame(rows)

    def _selection(self, manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        eligible: list[tuple[str, float, tuple[float, ...]]] = []
        ranking_metrics: dict[str, dict[str, float]] = {}
        rejection_reasons: dict[str, list[str]] = {}
        for manifest in manifests:
            trial = _mapping(manifest.get("trial", {}), "trial manifest trial")
            trial_id = str(trial.get("id", ""))
            if manifest.get("status") != "completed":
                error = manifest.get("error")
                detail = error.get("message") if isinstance(error, dict) else None
                rejection_reasons[trial_id] = [
                    f"trial_status={manifest.get('status')}",
                    *([f"error={detail}"] if detail else []),
                ]
                continue
            summary = manifest.get("summary")
            gate_failures = _gate_failures(summary, self.spec.selection.gates)
            if gate_failures:
                rejection_reasons[trial_id] = gate_failures
                continue
            primary_rule = self.spec.selection.primary
            primary_value = _finite_number(
                _resolve_path(summary, primary_rule.path)
            )
            observed_metrics: dict[str, float] = {}
            invalid: list[str] = []
            if primary_value is None:
                invalid.append(f"non_finite_metric={primary_rule.path}")
            else:
                observed_metrics[primary_rule.path] = primary_value
            tie_values: list[float] = []
            for rule in self.spec.selection.tie_breakers:
                value = _finite_number(_resolve_path(summary, rule.path))
                if value is None:
                    invalid.append(f"non_finite_metric={rule.path}")
                    continue
                observed_metrics[rule.path] = value
                tie_values.append(
                    -value if rule.direction == "maximize" else value
                )
            if invalid:
                rejection_reasons[trial_id] = invalid
                continue
            if primary_value is None:
                raise AssertionError("Validated primary metric unexpectedly missing")
            oriented_primary = (
                -primary_value
                if primary_rule.direction == "maximize"
                else primary_value
            )
            eligible.append((trial_id, oriented_primary, tuple(tie_values)))
            ranking_metrics[trial_id] = observed_metrics

        best_primary = min((item[1] for item in eligible), default=None)
        shortlisted = [
            item
            for item in eligible
            if best_primary is not None
            and item[1] <= best_primary + self.spec.selection.primary_tolerance + 1e-12
        ]
        ranked_shortlist = sorted(
            shortlisted,
            key=lambda item: (item[2], item[1], item[0]),
        )
        outside = [item for item in eligible if item not in shortlisted]
        ranked_outside = sorted(outside, key=lambda item: (item[1], item[2], item[0]))
        ranked = [*ranked_shortlist, *ranked_outside]
        selected_trial_id = ranked_shortlist[0][0] if ranked_shortlist else None
        eligible_ids = [trial_id for trial_id, _, _ in ranked]
        shortlisted_ids = [trial_id for trial_id, _, _ in ranked_shortlist]
        for trial_id in eligible_ids:
            if trial_id != selected_trial_id:
                rejection_reasons[trial_id] = [
                    (
                        "ranked_below_selected_trial_within_primary_band"
                        if trial_id in shortlisted_ids
                        else "outside_primary_equivalence_band"
                    )
                ]
        return {
            "schema_version": 1,
            "study_id": self.spec.study_id,
            "created_at": utc_now(),
            "rule": self.spec.selection.to_dict(),
            "selected_trial_id": selected_trial_id,
            "eligible_trial_ids": eligible_ids,
            "shortlisted_trial_ids": shortlisted_ids,
            "ranking": [
                {
                    "rank": rank,
                    "trial_id": trial_id,
                    "metrics": ranking_metrics[trial_id],
                    "within_primary_equivalence_band": trial_id in shortlisted_ids,
                    "primary_gap_from_best": (
                        oriented_primary - best_primary
                        if best_primary is not None
                        else None
                    ),
                }
                for rank, (trial_id, oriented_primary, _) in enumerate(
                    ranked,
                    start=1,
                )
            ],
            "rejected_trials": [
                {"trial_id": trial_id, "reasons": rejection_reasons[trial_id]}
                for trial_id in sorted(rejection_reasons)
            ],
            "multiplicity": {
                "declared_trial_count": len(self.spec.trials),
                "completed_trial_count": sum(
                    manifest.get("status") == "completed" for manifest in manifests
                ),
                "eligible_candidate_count": len(eligible),
            },
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def study_plan(config_path: str | Path) -> dict[str, Any]:
    spec = StudySpec.from_yaml(config_path)
    return {
        "study_id": spec.study_id,
        "base_config_path": str(spec.base_config_path),
        "trial_count": len(spec.trials),
        "max_trials": spec.max_trials,
        "selection": spec.selection.to_dict(),
        "trials": [trial.to_dict() for trial in spec.trials],
        "spec_hash": spec.spec_hash,
    }


def _selection_rule(value: Mapping[str, Any]) -> SelectionRule:
    _reject_unknown(
        value,
        {"primary", "tie_breakers", "gates", "primary_tolerance"},
        "selection",
    )
    primary = _metric_rule(
        _mapping(_required(value, "primary", "selection"), "selection.primary"),
        "selection.primary",
    )
    raw_ties = _sequence(value.get("tie_breakers", ()), "selection.tie_breakers")
    tie_breakers = tuple(
        _metric_rule(_mapping(item, f"selection.tie_breakers[{position}]"), "tie breaker")
        for position, item in enumerate(raw_ties)
    )
    raw_gates = _sequence(value.get("gates", ()), "selection.gates")
    gates = tuple(
        _gate_rule(_mapping(item, f"selection.gates[{position}]"), position)
        for position, item in enumerate(raw_gates)
    )
    primary_tolerance = float(value.get("primary_tolerance", 0.0))
    if not math.isfinite(primary_tolerance) or primary_tolerance < 0:
        raise ConfigurationError(
            "selection.primary_tolerance must be finite and nonnegative"
        )
    return SelectionRule(
        primary=primary,
        tie_breakers=tie_breakers,
        gates=gates,
        primary_tolerance=primary_tolerance,
    )


def _metric_rule(value: Mapping[str, Any], section: str) -> MetricRule:
    _reject_unknown(value, {"path", "direction"}, section)
    path = _metric_path(_required(value, "path", section), f"{section}.path")
    direction = str(_required(value, "direction", section))
    if direction not in _DIRECTIONS:
        raise ConfigurationError(
            f"{section}.direction must be one of {sorted(_DIRECTIONS)}"
        )
    return MetricRule(path=path, direction=direction)


def _gate_rule(value: Mapping[str, Any], position: int) -> GateRule:
    section = f"selection.gates[{position}]"
    _reject_unknown(value, {"path", "operator", "value"}, section)
    path = _metric_path(_required(value, "path", section), f"{section}.path")
    operator = str(_required(value, "operator", section))
    if operator not in _OPERATORS:
        raise ConfigurationError(
            f"{section}.operator must be one of {sorted(_OPERATORS)}"
        )
    return GateRule(
        path=path,
        operator=operator,
        value=_required(value, "value", section),
    )


def _trial(value: Any, position: int) -> StudyTrial:
    section = f"trials[{position}]"
    mapping = _mapping(value, section)
    _reject_unknown(mapping, {"id", "purpose", "overrides"}, section)
    trial_id = _simple_name(_required(mapping, "id", section), f"{section}.id")
    purpose = str(_required(mapping, "purpose", section)).strip()
    if not purpose:
        raise ConfigurationError(f"{section}.purpose cannot be empty")
    overrides = _mapping(mapping.get("overrides", {}), f"{section}.overrides")
    return StudyTrial(trial_id=trial_id, purpose=purpose, overrides=dict(overrides))


def _component_override(
    value: Any,
    current: ComponentSettings,
    section: str,
) -> ComponentSettings:
    mapping = _mapping(value, section)
    _reject_unknown(mapping, {"name", "params"}, section)
    name = str(mapping.get("name", current.name)).strip()
    if not name:
        raise ConfigurationError(f"{section}.name cannot be empty")
    params = mapping.get("params", current.params)
    return ComponentSettings(
        name=name,
        params=_mapping(params, f"{section}.params"),
    )


def _metric_paths(rule: SelectionRule) -> tuple[str, ...]:
    ordered = [
        rule.primary.path,
        *(item.path for item in rule.tie_breakers),
        *(item.path for item in rule.gates),
    ]
    return tuple(dict.fromkeys(ordered))


def _metric_column(path: str) -> str:
    return f"metric__{path.replace('.', '__')}"


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


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
                f"gate_failed={gate.path}{gate.operator}{gate.value!r};observed={observed!r}"
            )
    return failures


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _simple_name(value: Any, section: str) -> str:
    name = str(value).strip()
    if not name or not name[0].isalnum() or any(
        not (character.isalnum() or character in "-_.") for character in name
    ):
        raise ConfigurationError(
            f"{section} must contain only letters, digits, '-', '_' or '.'"
        )
    return name


def _metric_path(value: Any, section: str) -> str:
    path = str(value).strip()
    if not path or any(not part for part in path.split(".")):
        raise ConfigurationError(f"{section} must be a non-empty dotted path")
    return path


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
