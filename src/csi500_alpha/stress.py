from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from csi500_alpha.config import AppConfig
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import enrich_active_performance
from csi500_alpha.utils import canonical_json, sha256_text, utc_now

_SUMMARY_PATHS = (
    "metrics.information_ratio",
    "metrics.max_drawdown",
    "metrics.active_max_drawdown",
    "metrics.portfolio_max_drawdown",
    "metrics.average_turnover",
    "metrics.annualized_active_return",
    "metrics.transaction_cost",
    "metrics.target_configured_breach_fraction",
    "metrics.post_trade_configured_breach_fraction",
    "metrics.post_trade_policy_violation_fraction",
    "metrics.post_trade_material_configured_breach_fraction",
    "metrics.post_trade_material_policy_violation_fraction",
    "metrics.post_trade_material_execution_deterioration_fraction",
    "metrics.maximum_target_configured_constraint_breach",
    "metrics.maximum_post_trade_policy_violation",
    "metrics.maximum_execution_constraint_deterioration",
    "metrics.maximum_post_trade_active_beta_deviation",
    "metrics.maximum_post_trade_industry_active_exposure",
    "metrics.maximum_post_trade_tracking_error",
    "metrics.p95_actual_active_risk_utilization",
    "metrics.beta_audit_complete_fraction",
    "evaluation.yearly.minimum_information_ratio",
    "evaluation.yearly.positive_active_year_fraction",
    "evaluation.execution.notional_fill_ratio",
    "evaluation.execution.cost_bps_of_executed_notional",
    "optimizer_solve_rate",
    "source_metric_max_abs_difference",
    "source_metric_parity_passed",
)


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    purpose: str
    execution_mode: str = "reoptimized"
    cost_multiplier: float = 1.0
    portfolio_aum_cny: float | None = None
    max_adv_participation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.scenario_id,
            "purpose": self.purpose,
            "execution_mode": self.execution_mode,
            "cost_multiplier": self.cost_multiplier,
            "portfolio_aum_cny": self.portfolio_aum_cny,
            "max_adv_participation": self.max_adv_participation,
        }


@dataclass(frozen=True)
class StressSpec:
    config_path: Path
    stress_id: str
    source_study_path: Path
    source_study_reference: str
    max_scenarios: int
    baseline_parity_tolerance: float
    scenarios: tuple[StressScenario, ...]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> StressSpec:
        path = Path(config_path).resolve()
        if not path.is_file():
            raise ConfigurationError(f"Stress configuration does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        root = _mapping(raw, "stress config")
        _reject_unknown(root, {"stress", "scenarios"}, "stress config")
        settings = _mapping(_required(root, "stress", "stress config"), "stress")
        _reject_unknown(
            settings,
            {
                "id",
                "source_study",
                "max_scenarios",
                "baseline_parity_tolerance",
            },
            "stress",
        )
        stress_id = _simple_name(_required(settings, "id", "stress"), "stress.id")
        source_reference = str(_required(settings, "source_study", "stress"))
        source_path = (path.parent / source_reference).resolve()
        if not source_path.is_file():
            raise ConfigurationError(
                f"stress.source_study does not exist: {source_path}"
            )
        max_scenarios = int(settings.get("max_scenarios", 12))
        if not 1 <= max_scenarios <= 25:
            raise ConfigurationError("stress.max_scenarios must be between 1 and 25")
        tolerance = float(settings.get("baseline_parity_tolerance", 1e-8))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ConfigurationError(
                "stress.baseline_parity_tolerance must be finite and nonnegative"
            )
        raw_scenarios = _sequence(
            _required(root, "scenarios", "stress config"),
            "stress.scenarios",
        )
        if not raw_scenarios:
            raise ConfigurationError("Stress configuration requires at least one scenario")
        if len(raw_scenarios) > max_scenarios:
            raise ConfigurationError(
                f"Stress configuration declares {len(raw_scenarios)} scenarios but "
                f"max_scenarios={max_scenarios}"
            )
        scenarios = tuple(
            _scenario(value, position) for position, value in enumerate(raw_scenarios)
        )
        ids = [scenario.scenario_id for scenario in scenarios]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("Stress scenario ids must be unique")
        baseline = [scenario for scenario in scenarios if scenario.scenario_id == "baseline"]
        if len(baseline) != 1 or baseline[0] != StressScenario(
            scenario_id="baseline",
            purpose=baseline[0].purpose if baseline else "",
        ):
            raise ConfigurationError(
                "Stress configuration requires one unmodified scenario with id 'baseline'"
            )
        return cls(
            config_path=path,
            stress_id=stress_id,
            source_study_path=source_path,
            source_study_reference=source_reference,
            max_scenarios=max_scenarios,
            baseline_parity_tolerance=tolerance,
            scenarios=scenarios,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stress": {
                "id": self.stress_id,
                "source_study": self.source_study_reference,
                "max_scenarios": self.max_scenarios,
                "baseline_parity_tolerance": self.baseline_parity_tolerance,
            },
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    @property
    def spec_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass(frozen=True)
class StressContext:
    source_study_id: str
    source_trial_id: str
    source_resolved_config_hash: str
    source_run_manifest_hash: str
    source_hash: str
    data_snapshot_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_study_id": self.source_study_id,
            "source_trial_id": self.source_trial_id,
            "source_resolved_config_hash": self.source_resolved_config_hash,
            "source_run_manifest_hash": self.source_run_manifest_hash,
            "source_hash": self.source_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
        }


@dataclass(frozen=True)
class StressExecution:
    summary: dict[str, Any]


@dataclass(frozen=True)
class StressResult:
    stress_id: str
    status: str
    stress_root: Path
    source_trial_id: str
    scenario_count: int
    completed_count: int
    failed_count: int
    skipped_count: int
    baseline_parity_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stress_id": self.stress_id,
            "status": self.status,
            "stress_root": str(self.stress_root),
            "source_trial_id": self.source_trial_id,
            "scenario_count": self.scenario_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "baseline_parity_passed": self.baseline_parity_passed,
        }


StressExecutor = Callable[[StressScenario, Path], StressExecution]


class StressRunner:
    """Run bounded one-way stresses against one immutable selected signal stream."""

    def __init__(
        self,
        spec: StressSpec,
        *,
        stress_root: Path,
        context: StressContext,
        executor: StressExecutor,
    ) -> None:
        self.spec = spec
        self.stress_root = stress_root.resolve()
        self.context = context
        self.executor = executor

    def run(self) -> StressResult:
        manifest = self._initialize_manifest()
        skipped = 0
        for scenario in self.spec.scenarios:
            path = self._scenario_manifest_path(scenario)
            existing = self._read_json(path)
            scenario_hash = self._scenario_hash(scenario)
            if existing:
                self._assert_scenario_identity(existing, scenario, scenario_hash)
                if existing.get("status") == "completed":
                    skipped += 1
                    continue
            attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
            attempt_root = path.parent / f"attempt-{attempts:03d}"
            running = {
                "schema_version": 1,
                "stress_id": self.spec.stress_id,
                "scenario": scenario.to_dict(),
                "scenario_hash": scenario_hash,
                **self.context.to_dict(),
                "status": "running",
                "attempts": attempts,
                "started_at": utc_now(),
                "completed_at": None,
                "artifact_root": attempt_root.relative_to(
                    self.stress_root
                ).as_posix(),
                "summary": None,
                "error": None,
            }
            write_json_atomic(running, path)
            try:
                execution = self.executor(scenario, attempt_root)
            except Exception as exc:  # noqa: BLE001 - failures are research evidence
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
                continue
            write_json_atomic(
                {
                    **running,
                    "status": "completed",
                    "completed_at": utc_now(),
                    "summary": execution.summary,
                },
                path,
            )

        scenario_manifests = [
            self._read_json(self._scenario_manifest_path(scenario))
            for scenario in self.spec.scenarios
        ]
        results = self._result_table(scenario_manifests)
        write_parquet_atomic(results, self.stress_root / "stress-results.parquet")
        comparison = self._comparison(scenario_manifests)
        write_json_atomic(comparison, self.stress_root / "stress-summary.json")
        completed_count = sum(
            item.get("status") == "completed" for item in scenario_manifests
        )
        failed_count = sum(
            item.get("status") == "failed" for item in scenario_manifests
        )
        baseline_parity = comparison["baseline_parity_passed"] is True
        if failed_count:
            status = "completed_with_failures"
        elif not baseline_parity:
            status = "baseline_parity_failed"
        else:
            status = "completed"
        write_json_atomic(
            {
                **manifest,
                "status": status,
                "updated_at": utc_now(),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "baseline_parity_passed": baseline_parity,
                "artifacts": {
                    "results": "stress-results.parquet",
                    "summary": "stress-summary.json",
                    "scenario_root": "scenarios",
                },
            },
            self.stress_root / "stress-manifest.json",
        )
        return StressResult(
            stress_id=self.spec.stress_id,
            status=status,
            stress_root=self.stress_root,
            source_trial_id=self.context.source_trial_id,
            scenario_count=len(self.spec.scenarios),
            completed_count=completed_count,
            failed_count=failed_count,
            skipped_count=skipped,
            baseline_parity_passed=baseline_parity,
        )

    def _initialize_manifest(self) -> dict[str, Any]:
        path = self.stress_root / "stress-manifest.json"
        existing = self._read_json(path)
        identity = {"spec_hash": self.spec.spec_hash, **self.context.to_dict()}
        if existing:
            changed = sorted(
                key for key, value in identity.items() if existing.get(key) != value
            )
            if changed:
                raise ConfigurationError(
                    "Existing stress identity differs; create a new stress id. "
                    f"Changed fields: {changed}"
                )
            return existing
        manifest = {
            "schema_version": 1,
            "stress_id": self.spec.stress_id,
            "stress_config_path": str(self.spec.config_path),
            **identity,
            "scenario_count": len(self.spec.scenarios),
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        write_json_atomic(manifest, path)
        return manifest

    def _scenario_manifest_path(self, scenario: StressScenario) -> Path:
        return (
            self.stress_root
            / "scenarios"
            / scenario.scenario_id
            / "scenario-manifest.json"
        )

    def _scenario_hash(self, scenario: StressScenario) -> str:
        return sha256_text(
            canonical_json(
                {
                    "source_resolved_config_hash": (
                        self.context.source_resolved_config_hash
                    ),
                    "scenario": scenario.to_dict(),
                }
            )
        )

    def _assert_scenario_identity(
        self,
        existing: Mapping[str, Any],
        scenario: StressScenario,
        scenario_hash: str,
    ) -> None:
        expected = {
            "scenario_hash": scenario_hash,
            **self.context.to_dict(),
        }
        changed = sorted(
            key for key, value in expected.items() if existing.get(key) != value
        )
        if changed:
            raise ConfigurationError(
                f"Stress scenario {scenario.scenario_id!r} cannot resume because "
                f"identity fields changed: {changed}"
            )

    def _result_table(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for manifest in manifests:
            scenario = _mapping(manifest.get("scenario", {}), "scenario manifest")
            summary = manifest.get("summary")
            error = manifest.get("error")
            row: dict[str, Any] = {
                "stress_id": self.spec.stress_id,
                "scenario_id": str(scenario.get("id", "")),
                "purpose": str(scenario.get("purpose", "")),
                "execution_mode": scenario.get("execution_mode"),
                "cost_multiplier": scenario.get("cost_multiplier"),
                "portfolio_aum_cny": scenario.get("portfolio_aum_cny"),
                "max_adv_participation": scenario.get("max_adv_participation"),
                "status": manifest.get("status"),
                "attempts": int(manifest.get("attempts", 0)),
                "artifact_root": manifest.get("artifact_root"),
                "error_type": error.get("type") if isinstance(error, dict) else None,
                "error_message": (
                    error.get("message") if isinstance(error, dict) else None
                ),
                "summary_json": canonical_json(summary)
                if summary is not None
                else None,
            }
            for path in _SUMMARY_PATHS:
                row[_metric_column(path)] = _resolve_path(summary, path)
            rows.append(row)
        return pd.DataFrame(rows)

    def _comparison(
        self,
        manifests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        summaries: dict[str, Mapping[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        for manifest in manifests:
            scenario = _mapping(manifest.get("scenario", {}), "scenario manifest")
            scenario_id = str(scenario.get("id", ""))
            summary = manifest.get("summary")
            if manifest.get("status") == "completed" and isinstance(summary, Mapping):
                summaries[scenario_id] = summary
            else:
                error = manifest.get("error")
                failures.append(
                    {
                        "scenario_id": scenario_id,
                        "status": manifest.get("status"),
                        "error": error.get("message")
                        if isinstance(error, dict)
                        else None,
                    }
                )
        baseline = summaries.get("baseline")
        baseline_parity = (
            _resolve_path(baseline, "source_metric_parity_passed")
            if baseline is not None
            else None
        )
        comparison_rows: list[dict[str, Any]] = []
        for scenario_id, summary in sorted(summaries.items()):
            metrics: dict[str, Any] = {}
            for path in _SUMMARY_PATHS:
                value = _finite_number(_resolve_path(summary, path))
                baseline_value = _finite_number(_resolve_path(baseline, path))
                metrics[path] = value
                metrics[f"delta.{path}"] = (
                    value - baseline_value
                    if value is not None and baseline_value is not None
                    else None
                )
            comparison_rows.append(
                {"scenario_id": scenario_id, "metrics": metrics}
            )
        return {
            "schema_version": 1,
            "stress_id": self.spec.stress_id,
            "created_at": utc_now(),
            "source_trial_id": self.context.source_trial_id,
            "baseline_parity_passed": baseline_parity,
            "completed_scenarios": sorted(summaries),
            "failed_scenarios": failures,
            "comparison": comparison_rows,
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def resolve_stress_config(base: AppConfig, scenario: StressScenario) -> AppConfig:
    research = replace(
        base.research,
        linear_cost_bps=base.research.linear_cost_bps * scenario.cost_multiplier,
        stamp_duty_before=(
            base.research.stamp_duty_before * scenario.cost_multiplier
        ),
        stamp_duty_after=base.research.stamp_duty_after * scenario.cost_multiplier,
    )
    optimizer = replace(
        base.optimizer,
        portfolio_aum_cny=(
            scenario.portfolio_aum_cny
            if scenario.portfolio_aum_cny is not None
            else base.optimizer.portfolio_aum_cny
        ),
        max_adv_participation=(
            scenario.max_adv_participation
            if scenario.max_adv_participation is not None
            else base.optimizer.max_adv_participation
        ),
        impact_bps_at_max_participation=(
            base.optimizer.impact_bps_at_max_participation
            * scenario.cost_multiplier
        ),
    )
    resolved = replace(base, research=research, optimizer=optimizer)
    resolved.validate()
    return resolved


def replay_frozen_trade_costs(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    cost_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay identical gross trades while changing only modeled cash costs.

    Holdings and gross trade values remain fixed.  With a zero cash return, the
    counterfactual NAV differs from the source NAV by the cumulative change in
    linear, stamp-duty and impact costs.
    """

    if not np.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be finite and positive")
    required_daily = {
        "trade_date",
        "nav",
        "benchmark_nav",
        "transaction_cost",
    }
    required_trades = {
        "trade_date",
        "linear_cost",
        "stamp_duty",
        "impact_cost",
    }
    missing_daily = sorted(required_daily.difference(daily.columns))
    missing_trades = sorted(required_trades.difference(trades.columns))
    if missing_daily or missing_trades:
        raise ValueError(
            "Frozen-trade replay inputs are incomplete: "
            f"daily={missing_daily}, trades={missing_trades}"
        )

    replay_trades = trades.copy()
    cost_columns = ("linear_cost", "stamp_duty", "impact_cost")
    for column in cost_columns:
        replay_trades[column] = (
            pd.to_numeric(replay_trades[column], errors="raise").astype(float)
            * cost_multiplier
        )

    source_cost_by_date = (
        trades.assign(
            _total_cost=sum(
                (
                    pd.to_numeric(trades[column], errors="raise").astype(float)
                    for column in cost_columns
                ),
                start=pd.Series(0.0, index=trades.index),
            )
        )
        .groupby("trade_date")["_total_cost"]
        .sum()
    )
    replay = daily.copy()
    recorded_cost = pd.to_numeric(
        replay["transaction_cost"],
        errors="raise",
    ).astype(float)
    trade_cost = replay["trade_date"].map(source_cost_by_date).fillna(0.0)
    if not np.allclose(recorded_cost, trade_cost, rtol=1e-10, atol=1e-12):
        maximum_difference = float((recorded_cost - trade_cost).abs().max())
        raise ValueError(
            "Daily and trade-level source costs do not reconcile: "
            f"maximum_difference={maximum_difference}"
        )

    scenario_cost = recorded_cost * cost_multiplier
    cumulative_cost_delta = (scenario_cost - recorded_cost).cumsum()
    replay["nav"] = (
        pd.to_numeric(replay["nav"], errors="raise").astype(float)
        - cumulative_cost_delta
    )
    if (replay["nav"] <= 0).any() or not np.isfinite(replay["nav"]).all():
        raise ValueError("Frozen-trade cost replay exhausted portfolio NAV")
    if "cash" in replay:
        replay["cash"] = (
            pd.to_numeric(replay["cash"], errors="raise").astype(float)
            - cumulative_cost_delta
        )
    replay["transaction_cost"] = scenario_cost
    replay["cost_replay_cumulative_delta"] = cumulative_cost_delta
    replay["cost_replay_mode"] = "frozen_trades"
    return enrich_active_performance(replay), replay_trades


def stress_plan(config_path: str | Path) -> dict[str, Any]:
    spec = StressSpec.from_yaml(config_path)
    return {
        **spec.to_dict(),
        "stress_config_path": str(spec.config_path),
        "source_study_path": str(spec.source_study_path),
        "scenario_count": len(spec.scenarios),
        "spec_hash": spec.spec_hash,
    }


def _scenario(value: Any, position: int) -> StressScenario:
    section = f"scenarios[{position}]"
    mapping = _mapping(value, section)
    _reject_unknown(
        mapping,
        {
            "id",
            "purpose",
            "execution_mode",
            "cost_multiplier",
            "portfolio_aum_cny",
            "max_adv_participation",
        },
        section,
    )
    scenario_id = _simple_name(_required(mapping, "id", section), f"{section}.id")
    purpose = str(_required(mapping, "purpose", section)).strip()
    if not purpose:
        raise ConfigurationError(f"{section}.purpose cannot be empty")
    cost_multiplier = float(mapping.get("cost_multiplier", 1.0))
    execution_mode = str(mapping.get("execution_mode", "reoptimized"))
    if execution_mode not in {"reoptimized", "frozen_trades"}:
        raise ConfigurationError(
            f"{section}.execution_mode must be 'reoptimized' or 'frozen_trades'"
        )
    aum = _optional_positive(mapping.get("portfolio_aum_cny"), f"{section}.portfolio_aum_cny")
    participation = _optional_positive(
        mapping.get("max_adv_participation"),
        f"{section}.max_adv_participation",
    )
    if not np.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ConfigurationError(f"{section}.cost_multiplier must be positive")
    if participation is not None and participation > 1:
        raise ConfigurationError(
            f"{section}.max_adv_participation must be at most 1"
        )
    if execution_mode == "frozen_trades" and (
        aum is not None or participation is not None
    ):
        raise ConfigurationError(
            f"{section} frozen-trade replay can change costs only"
        )
    return StressScenario(
        scenario_id=scenario_id,
        purpose=purpose,
        execution_mode=execution_mode,
        cost_multiplier=cost_multiplier,
        portfolio_aum_cny=aum,
        max_adv_participation=participation,
    )


def _optional_positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return numeric


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be a list")
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{section}.{key} is required")
    return mapping[key]


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ConfigurationError(f"Unknown {section} keys: {unknown}")


def _simple_name(value: Any, name: str) -> str:
    text = str(value).strip()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not text or any(character not in allowed for character in text):
        raise ConfigurationError(
            f"{name} must contain only lowercase letters, digits, '-' or '_'"
        )
    return text


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _metric_column(path: str) -> str:
    return f"metric__{path.replace('.', '__')}"


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
