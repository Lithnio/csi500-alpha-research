from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from csi500_alpha.data.storage import write_json_atomic
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import (
    calculate_backtest_metrics,
    enrich_active_performance,
)
from csi500_alpha.portfolio.audit import summarize_constraint_audits
from csi500_alpha.reporting import (
    _BLUE,
    _BLUE_OPEN,
    _GRID,
    _INK,
    _MUTED,
    _RED,
    _RED_OPEN,
    _matplotlib,
    _save_figure,
    _validate_daily,
)
from csi500_alpha.utils import canonical_json, sha256_file, sha256_text

_JSON_ARTIFACTS = {"metrics", "research_evaluation", "workflow_summary"}
_REQUIRED_ARTIFACTS = {
    "calibration_bins",
    "calibration_fits",
    "daily",
    "evaluation_signals",
    "factor_weight_history",
    "metrics",
    "model_fits",
    "optimization",
    "positions",
    "constraint_audits",
    "research_evaluation",
    "signals",
    "trades",
    "workflow_summary",
}
_RANGE_SPECS = {
    "features": ("gold/features.parquet", "decision_date"),
    "labels": ("gold/labels.parquet", "decision_date"),
    "label_availability": ("gold/labels.parquet", "label_available_date"),
    "factor_diagnostics": ("factor-ic.parquet", "decision_date"),
    "signals": ("signals.parquet", "decision_date"),
    "evaluation_signals": ("evaluation-signals.parquet", "decision_date"),
    "backtest": ("daily.parquet", "trade_date"),
    "positions": ("positions.parquet", "execution_date"),
    "constraint_audits": ("constraint-audits.parquet", "execution_date"),
}
_KEY_SPECS = {
    "daily.parquet": ("trade_date",),
    "model-fits.parquet": ("fit_date",),
    "calibration-fits.parquet": ("fit_date",),
    "evaluation-signals.parquet": ("decision_date", "instrument"),
    "signals.parquet": ("decision_date", "instrument"),
    "factor-ic.parquet": ("decision_date", "factor"),
    "factor-weight-history.parquet": ("fit_date", "factor"),
    "targets.parquet": ("execution_date", "instrument"),
    "trades.parquet": ("signal_date", "trade_date", "instrument"),
    "optimization.parquet": ("decision_date",),
    "positions.parquet": ("execution_date", "instrument"),
    "constraint-audits.parquet": ("execution_date",),
    "gold/features.parquet": ("decision_date", "instrument"),
    "gold/raw_features.parquet": ("decision_date", "instrument"),
    "gold/labels.parquet": ("decision_date", "instrument"),
    "gold/research_panel.parquet": ("decision_date", "instrument"),
    "gold/feature_quality.parquet": ("decision_date", "factor"),
}


@dataclass(frozen=True)
class FinalHoldoutReportResult:
    output_root: Path
    summary_path: Path
    audit_path: Path
    manifest_path: Path
    figure_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "summary": str(self.summary_path),
            "audit": str(self.audit_path),
            "manifest": str(self.manifest_path),
            "figure": str(self.figure_path),
        }


def build_final_holdout_report(
    *,
    run_root: str | Path,
    output_root: str | Path,
) -> FinalHoldoutReportResult:
    """Audit and publish one completed frozen-test run without row-level exports."""

    run_path = Path(run_root).resolve()
    destination = Path(output_root).resolve()
    if not run_path.is_dir():
        raise ConfigurationError(f"Final holdout run does not exist: {run_path}")

    manifest_path = run_path / "run-manifest.json"
    run_manifest = _read_object(manifest_path, "Final run manifest")
    experiment = _mapping(run_manifest.get("experiment"), "Final experiment")
    summary = _mapping(run_manifest.get("summary"), "Final run summary")
    if experiment.get("stage") != "frozen_test":
        raise ConfigurationError("Final report requires a frozen_test run")
    run_id = str(run_manifest.get("run_id", ""))
    if not run_id or run_path.name != run_id:
        raise ConfigurationError("Final run directory and manifest run_id disagree")

    protocol_id = str(experiment.get("protocol_id", ""))
    if not protocol_id:
        raise ConfigurationError("Final run does not identify an experiment protocol")
    runs_root = run_path.parent
    repository_root = runs_root.parent
    protocol_root = runs_root / "_protocols" / protocol_id
    frozen_lock_path = protocol_root / "frozen-test.json"
    validation_lock_path = protocol_root / "validation-lock.json"
    frozen_lock = _read_object(frozen_lock_path, "Frozen-test lock")
    validation_lock = _read_object(validation_lock_path, "Validation lock")
    _validate_lineage(
        run_manifest=run_manifest,
        experiment=experiment,
        frozen_lock=frozen_lock,
        validation_lock=validation_lock,
    )

    artifact_fingerprints = _mapping(
        run_manifest.get("artifact_fingerprints"),
        "Final artifact fingerprints",
    )
    missing_required = sorted(_REQUIRED_ARTIFACTS.difference(artifact_fingerprints))
    if missing_required:
        raise ConfigurationError(
            f"Final run lacks required artifact fingerprints: {missing_required}"
        )
    artifact_mismatches = _artifact_mismatches(run_path, artifact_fingerprints)
    gold_mismatches = _gold_mismatches(run_path, run_manifest)
    silver_mismatches, actual_silver = _silver_mismatches(
        repository_root,
        run_manifest,
    )
    if artifact_mismatches or gold_mismatches or silver_mismatches:
        raise ConfigurationError(
            "Final holdout input fingerprints differ: "
            f"artifacts={artifact_mismatches}, gold={gold_mismatches}, "
            f"silver={silver_mismatches}"
        )
    expected_snapshot = sha256_text(canonical_json(dict(sorted(actual_silver.items()))))
    if expected_snapshot != experiment.get("data_snapshot_hash"):
        raise ConfigurationError("Final Silver snapshot hash does not match the run")

    metrics_path = run_path / "metrics.json"
    evaluation_path = run_path / "research-evaluation.json"
    workflow_summary_path = run_path / "workflow-summary.json"
    metrics = _read_object(metrics_path, "Final metrics")
    evaluation = _read_object(evaluation_path, "Final research evaluation")
    workflow_summary = _read_object(workflow_summary_path, "Final workflow summary")
    if canonical_json(workflow_summary) != canonical_json(summary):
        raise ConfigurationError("Workflow summary differs from the final run manifest")
    if canonical_json(metrics) != canonical_json(summary.get("metrics")):
        raise ConfigurationError("Metrics differ from the final run manifest")
    if canonical_json(evaluation) != canonical_json(summary.get("evaluation")):
        raise ConfigurationError("Research evaluation differs from the final run manifest")

    daily_path = run_path / "daily.parquet"
    trades_path = run_path / "trades.parquet"
    daily = pd.read_parquet(daily_path)
    trades = pd.read_parquet(trades_path)
    constraint_audits = pd.read_parquet(run_path / "constraint-audits.parquet")
    _validate_daily(daily)
    recalculated = _recalculate_metrics(daily, trades)
    constraint_summary = summarize_constraint_audits(constraint_audits)
    recalculated.update(constraint_summary)
    metric_differences = _metric_differences(metrics, recalculated)
    maximum_metric_difference = max(metric_differences.values(), default=0.0)
    if maximum_metric_difference > 1.0e-12:
        raise ConfigurationError(
            "Final metrics do not reproduce from daily and trade artifacts"
        )

    range_mismatches, stage_crossings = _audit_ranges(
        run_path,
        summary,
        test_end=str(experiment.get("test_end", "")),
    )
    duplicate_rows = _audit_keys(run_path)
    point_in_time_violations = _audit_point_in_time(
        run_path,
        test_start=str(experiment.get("test_start", "")),
        test_end=str(experiment.get("test_end", "")),
    )
    critical_failures, warnings = _quality_findings(run_manifest)
    if (
        range_mismatches
        or stage_crossings
        or sum(duplicate_rows.values())
        or sum(point_in_time_violations.values())
        or critical_failures
    ):
        raise ConfigurationError(
            "Final holdout audit failed: "
            f"ranges={range_mismatches}, crossings={stage_crossings}, "
            f"duplicates={duplicate_rows}, point_in_time={point_in_time_violations}, "
            f"critical_quality_failures={critical_failures}"
        )

    optimization = pd.read_parquet(run_path / "optimization.parquet")
    if optimization.empty:
        raise ConfigurationError("Final holdout has no optimization attempts")
    solved = int(
        optimization["status"].isin(["optimal", "optimal_inaccurate"]).sum()
    )
    maximum_violation = float(
        pd.to_numeric(optimization["maximum_violation"], errors="coerce").max()
    )
    calibration = _mapping(evaluation.get("calibration"), "Calibration evaluation")
    execution = _mapping(evaluation.get("execution"), "Execution evaluation")
    reported_constraints = _mapping(
        evaluation.get("constraints"),
        "Constraint evaluation",
    )
    if canonical_json(reported_constraints) != canonical_json(constraint_summary):
        raise ConfigurationError(
            "Constraint evaluation does not reproduce from post-trade audits"
        )
    top_weights = _final_factor_weights(run_path)
    if not top_weights:
        raise ConfigurationError("Final model has no material factor weights")
    calibration_bins = _public_calibration_bins(run_path)

    validation_run_id = str(frozen_lock.get("validation_run_id", ""))
    validation_manifest_path = runs_root / validation_run_id / "run-manifest.json"
    validation_manifest = _read_object(
        validation_manifest_path,
        "Validation run manifest",
    )
    validation_gold_path = validation_manifest_path.parent / "gold"
    final_gold_path = run_path / "gold"
    gold_paths_distinct = validation_gold_path.resolve() != final_gold_path.resolve()
    if not gold_paths_distinct:
        raise ConfigurationError("Validation and final runs share one Gold directory")
    validation_gold = _mapping(
        validation_manifest.get("gold_fingerprints"),
        "Validation Gold fingerprints",
    )
    final_gold = _mapping(
        run_manifest.get("gold_fingerprints"),
        "Final Gold fingerprints",
    )
    identical_gold = sum(
        final_gold.get(name) == validation_gold.get(name) for name in final_gold
    )

    audit = {
        "schema_version": 1,
        "data_source": "Tushare Pro",
        "evaluation_role": "final_holdout",
        "protocol_id": protocol_id,
        "run": {
            "run_id": run_id,
            "completed_at": frozen_lock.get("completed_at"),
            "git_commit": _mapping(run_manifest.get("git"), "Git metadata").get(
                "commit"
            ),
            "git_dirty": _mapping(run_manifest.get("git"), "Git metadata").get(
                "dirty"
            ),
        },
        "identity": {
            "research_spec_hash": experiment.get("research_spec_hash"),
            "research_source_hash": experiment.get("research_source_hash"),
            "data_snapshot_hash": experiment.get("data_snapshot_hash"),
            "validation_run_id": validation_run_id,
            "lock_matches_run": True,
            "validation_lineage_matches": True,
        },
        "scope": {
            "test_start": experiment.get("test_start"),
            "test_end": experiment.get("test_end"),
            "evaluation_start": metrics.get("start_date"),
            "evaluation_end": metrics.get("end_date"),
            "observations": metrics.get("observations"),
            "final_holdout": True,
        },
        "quality_audit": {
            "artifact_hash_mismatches": len(artifact_mismatches),
            "gold_hash_mismatches": len(gold_mismatches),
            "silver_hash_mismatches": len(silver_mismatches),
            "maximum_metric_recalculation_difference": maximum_metric_difference,
            "range_mismatches": len(range_mismatches),
            "stage_boundary_crossings": len(stage_crossings),
            "duplicate_key_rows": sum(duplicate_rows.values()),
            "point_in_time_violations": sum(point_in_time_violations.values()),
            "critical_data_quality_failures": critical_failures,
            "warning_count": len(warnings),
            "artifact_count": len(artifact_fingerprints),
            "gold_artifact_count": len(final_gold),
            "silver_table_count": len(actual_silver),
            "validation_and_frozen_gold_paths_distinct": gold_paths_distinct,
            "identical_validation_and_frozen_gold_fingerprints": identical_gold,
            "optimizer_attempts": int(len(optimization)),
            "optimizer_solved": solved,
            "maximum_optimizer_constraint_violation": maximum_violation,
            "post_trade_audit_count": constraint_summary.get(
                "post_trade_audit_count"
            ),
            "post_trade_policy_violation_fraction": constraint_summary.get(
                "post_trade_policy_violation_fraction"
            ),
            "maximum_post_trade_policy_violation": constraint_summary.get(
                "maximum_post_trade_policy_violation"
            ),
            "beta_audit_complete_fraction": constraint_summary.get(
                "beta_audit_complete_fraction"
            ),
        },
        "warnings": warnings,
        "limitations": [
            "The final holdout covers only the first half of 2026.",
            (
                "Historical industry coverage has one non-blocking warning; "
                "missing industries use an explicit unknown group."
            ),
            "Transaction costs and capacity are modeled rather than live fills.",
        ],
    }

    return_gap = float(metrics["total_return"]) - float(
        metrics["benchmark_total_return"]
    )
    relative_wealth_gap = float(metrics["final_nav"]) / float(
        metrics["final_benchmark_nav"]
    ) - 1.0
    performance_status = (
        "negative_active_holdout"
        if float(metrics["annualized_active_return"]) < 0.0
        else "positive_active_holdout"
    )
    final_effective_factor_count = 1.0 / sum(
        float(item["weight"]) ** 2 for item in top_weights
    )
    public_summary = {
        "schema_version": 1,
        "data_source": "Tushare Pro",
        "evaluation_role": "final_holdout",
        "scope": audit["scope"],
        "method": {
            "selector": summary.get("selector"),
            "model": summary.get("model"),
            "calibrator": summary.get("calibrator"),
            "candidate_factor_count": len(summary.get("factors", [])),
            "final_effective_factor_count": final_effective_factor_count,
            "final_material_factor_weights": top_weights,
        },
        "headline_metrics": {
            "total_return": metrics.get("total_return"),
            "benchmark_total_return": metrics.get("benchmark_total_return"),
            "return_gap_percentage_points": return_gap * 100.0,
            "relative_wealth_gap": relative_wealth_gap,
            "annualized_return": metrics.get("annualized_return"),
            "annualized_active_return": metrics.get("annualized_active_return"),
            "tracking_error": metrics.get("tracking_error"),
            "information_ratio": metrics.get("information_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "active_max_drawdown": metrics.get("active_max_drawdown"),
            "portfolio_max_drawdown": metrics.get("portfolio_max_drawdown"),
            "capm_alpha_annualized": metrics.get("capm_alpha_annualized"),
            "capm_beta": metrics.get("capm_beta"),
            "post_trade_audit_count": metrics.get("post_trade_audit_count"),
            "maximum_post_trade_active_beta_deviation": metrics.get(
                "maximum_post_trade_active_beta_deviation"
            ),
            "maximum_post_trade_industry_active_exposure": metrics.get(
                "maximum_post_trade_industry_active_exposure"
            ),
            "post_trade_policy_violation_fraction": metrics.get(
                "post_trade_policy_violation_fraction"
            ),
            "beta_audit_complete_fraction": metrics.get(
                "beta_audit_complete_fraction"
            ),
            "average_turnover": metrics.get("average_turnover"),
            "execution_cost_bps": execution.get("cost_bps_of_executed_notional"),
            "notional_fill_ratio": execution.get("notional_fill_ratio"),
            "optimizer_solve_rate": solved / len(optimization),
            "daily_rank_ic": calibration.get("mean_daily_rank_ic"),
            "calibration_slope": calibration.get("slope"),
            "quintile_monotonicity": calibration.get("quintile_monotonicity"),
            "top_minus_bottom_realized_return": calibration.get(
                "top_minus_bottom_realized_return"
            ),
        },
        "calibration_bins": calibration_bins,
        "interpretation": {
            "status": performance_status,
            "active_performance_supported": performance_status
            == "positive_active_holdout",
            "finding": (
                "The frozen method made money in absolute terms but did not deliver "
                "positive active performance against the CSI 500 in the final holdout."
            ),
            "diagnostic": (
                "Average daily rank IC remained positive, but the highest expected-return "
                "bin realized a lower active return than the lowest bin. The concentrated "
                "low-risk/liquidity factor mix is a plausible sensitivity, not a proven cause."
            ),
        },
        "audit": audit["quality_audit"],
        "limitations": audit["limitations"],
    }

    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "final-holdout-summary.json"
    audit_path = destination / "final-holdout-audit.json"
    figure_path = destination / "final-holdout.png"
    public_manifest_path = destination / "final-holdout-report-manifest.json"
    write_json_atomic(public_summary, summary_path)
    write_json_atomic(audit, audit_path)
    pyplot, mdates = _matplotlib()
    _plot_final_holdout(
        pyplot,
        mdates,
        daily,
        metrics,
        figure_path,
    )

    generator_path = Path(__file__).resolve()
    shared_reporting_path = generator_path.with_name("reporting.py")
    storage_path = generator_path.parent / "data" / "storage.py"
    source_artifacts = {
        "generator/final_reporting.py": generator_path,
        "generator/reporting.py": shared_reporting_path,
        "generator/storage.py": storage_path,
        "final/run-manifest.json": manifest_path,
        "final/daily.parquet": daily_path,
        "final/trades.parquet": trades_path,
        "final/metrics.json": metrics_path,
        "final/research-evaluation.json": evaluation_path,
        "protocol/frozen-test.json": frozen_lock_path,
        "protocol/validation-lock.json": validation_lock_path,
    }
    outputs = {
        path.name: sha256_file(path)
        for path in (summary_path, audit_path, figure_path)
    }
    public_manifest = {
        "schema_version": 2,
        "public_boundary": (
            "Aggregate statistics and one raster figure only. Signals, holdings, "
            "orders, trades, index weights, vendor rows and local paths are excluded."
        ),
        "evaluation_role": "final_holdout",
        "protocol_id": protocol_id,
        "run_id": run_id,
        "chart_map": {
            "final-holdout.png": {
                "title": "Final holdout NAV, relative wealth gap and drawdown",
                "note": (
                    f"The portfolio returned {float(metrics['total_return']):.2%} "
                    f"versus the benchmark's "
                    f"{float(metrics['benchmark_total_return']):.2%}; the final "
                    "holdout did not support positive active performance."
                ),
                "type": "three-panel line and filled-area chart",
                "fields": ["trade_date", "nav", "benchmark_nav"],
                "observations": int(len(daily)),
                "scope": "one-time 2026 final holdout; no post-holdout retuning",
            }
        },
        "source_artifacts": {
            name: sha256_file(path)
            for name, path in sorted(source_artifacts.items())
        },
        "outputs": dict(sorted(outputs.items())),
    }
    write_json_atomic(public_manifest, public_manifest_path)
    return FinalHoldoutReportResult(
        output_root=destination,
        summary_path=summary_path,
        audit_path=audit_path,
        manifest_path=public_manifest_path,
        figure_path=figure_path,
    )


def _validate_lineage(
    *,
    run_manifest: Mapping[str, Any],
    experiment: Mapping[str, Any],
    frozen_lock: Mapping[str, Any],
    validation_lock: Mapping[str, Any],
) -> None:
    run_id = run_manifest.get("run_id")
    keys = (
        "protocol_id",
        "research_spec_hash",
        "research_source_hash",
        "data_snapshot_hash",
    )
    if frozen_lock.get("stage") != "frozen_test" or frozen_lock.get("run_id") != run_id:
        raise ConfigurationError("Frozen-test lock does not match the selected run")
    if validation_lock.get("stage") != "validation":
        raise ConfigurationError("Protocol validation lock has the wrong stage")
    if frozen_lock.get("validation_run_id") != validation_lock.get("run_id"):
        raise ConfigurationError("Frozen-test lock does not descend from validation")
    changed = [
        key
        for key in keys
        if experiment.get(key)
        != frozen_lock.get(key)
        or experiment.get(key) != validation_lock.get(key)
    ]
    if changed:
        raise ConfigurationError(f"Final protocol lineage differs: {changed}")
    run_data = _mapping(run_manifest.get("data_fingerprints"), "Run Silver fingerprints")
    if canonical_json(run_data) != canonical_json(frozen_lock.get("data_fingerprints")):
        raise ConfigurationError("Frozen lock and run Silver fingerprints differ")
    if canonical_json(run_data) != canonical_json(validation_lock.get("data_fingerprints")):
        raise ConfigurationError("Validation and final Silver fingerprints differ")


def _artifact_mismatches(
    run_root: Path,
    fingerprints: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for name, expected in sorted(fingerprints.items()):
        suffix = ".json" if name in _JSON_ARTIFACTS else ".parquet"
        path = run_root / f"{name.replace('_', '-')}{suffix}"
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or sha256_file(path) != expected
        ):
            mismatches.append(name)
    return mismatches


def _gold_mismatches(
    run_root: Path,
    run_manifest: Mapping[str, Any],
) -> list[str]:
    fingerprints = _mapping(run_manifest.get("gold_fingerprints"), "Gold fingerprints")
    mismatches: list[str] = []
    for name, expected in sorted(fingerprints.items()):
        path = run_root / "gold" / f"{name}.parquet"
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or sha256_file(path) != expected
        ):
            mismatches.append(name)
    return mismatches


def _silver_mismatches(
    repository_root: Path,
    run_manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    dataset = str(run_manifest.get("dataset", ""))
    if not dataset:
        raise ConfigurationError("Final run does not identify its Silver dataset")
    fingerprints = _mapping(
        run_manifest.get("data_fingerprints"),
        "Silver fingerprints",
    )
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected in sorted(fingerprints.items()):
        path = repository_root / "data" / "silver" / dataset / f"{name}.parquet"
        if not path.is_file() or not isinstance(expected, str):
            mismatches.append(name)
            continue
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            mismatches.append(name)
    return mismatches, actual


def _recalculate_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    required_daily = {
        "trade_date",
        "nav",
        "benchmark_nav",
        "turnover",
    }
    required_trades = {
        "status",
        "linear_cost",
        "stamp_duty",
        "impact_cost",
    }
    missing_daily = sorted(required_daily.difference(daily.columns))
    missing_trades = sorted(required_trades.difference(trades.columns))
    if missing_daily or missing_trades:
        raise ConfigurationError(
            f"Final metric inputs lack columns: daily={missing_daily}, trades={missing_trades}"
        )
    return calculate_backtest_metrics(
        enrich_active_performance(daily),
        trades,
    )


def _metric_differences(
    reported: Mapping[str, Any],
    recalculated: Mapping[str, Any],
) -> dict[str, float]:
    differences: dict[str, float] = {}
    for name, expected in recalculated.items():
        actual = reported.get(name)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)):
                differences[name] = math.inf
            elif math.isnan(expected) and math.isnan(float(actual)):
                differences[name] = 0.0
            elif not math.isfinite(expected) or not math.isfinite(float(actual)):
                differences[name] = math.inf
            else:
                differences[name] = abs(float(actual) - expected)
        else:
            differences[name] = 0.0 if actual == expected else 1.0
    return differences


def _audit_ranges(
    run_root: Path,
    summary: Mapping[str, Any],
    *,
    test_end: str,
) -> tuple[list[str], list[str]]:
    declared = _mapping(summary.get("artifact_date_ranges"), "Artifact date ranges")
    mismatches: list[str] = []
    crossings: list[str] = []
    for name, (relative, column) in _RANGE_SPECS.items():
        frame = pd.read_parquet(run_root / relative, columns=[column])
        values = frame[column].dropna().astype(str)
        actual = {
            "start": None if values.empty else str(values.min()),
            "end": None if values.empty else str(values.max()),
        }
        if canonical_json(actual) != canonical_json(declared.get(name)):
            mismatches.append(name)
        if actual["end"] is not None and actual["end"] > test_end:
            crossings.append(name)
    for relative, column in (
        ("targets.parquet", "execution_date"),
        ("trades.parquet", "trade_date"),
        ("optimization.parquet", "execution_date"),
    ):
        frame = pd.read_parquet(run_root / relative, columns=[column])
        values = frame[column].dropna().astype(str)
        if not values.empty and str(values.max()) > test_end:
            crossings.append(relative)
    return mismatches, crossings


def _audit_keys(run_root: Path) -> dict[str, int]:
    duplicates: dict[str, int] = {}
    for relative, keys in _KEY_SPECS.items():
        path = run_root / relative
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=list(keys))
        duplicates[relative] = int(frame.duplicated(list(keys)).sum())
    return duplicates


def _audit_point_in_time(
    run_root: Path,
    *,
    test_start: str,
    test_end: str,
) -> dict[str, int]:
    model = pd.read_parquet(
        run_root / "model-fits.parquet",
        columns=["fit_date", "max_label_available_date"],
    )
    calibration = pd.read_parquet(
        run_root / "calibration-fits.parquet",
        columns=["fit_date", "max_label_available_date"],
    )
    signals = pd.read_parquet(
        run_root / "signals.parquet",
        columns=["decision_date", "model_fit_date", "calibrator_fit_date"],
    )
    evaluation = pd.read_parquet(
        run_root / "evaluation-signals.parquet",
        columns=["decision_date"],
    )
    daily = pd.read_parquet(run_root / "daily.parquet", columns=["trade_date"])

    def not_strictly_before(frame: pd.DataFrame) -> int:
        available = frame["max_label_available_date"].fillna("").astype(str)
        fit = frame["fit_date"].fillna("").astype(str)
        return int(((available != "") & (fit != "") & (available >= fit)).sum())

    decisions = signals["decision_date"].astype(str)
    evaluation_dates = evaluation["decision_date"].astype(str)
    trade_dates = daily["trade_date"].astype(str)
    return {
        "model_training_label_not_before_fit": not_strictly_before(model),
        "calibration_training_label_not_before_fit": not_strictly_before(calibration),
        "model_fit_after_signal": int(
            (
                signals["model_fit_date"].fillna("").astype(str)
                > decisions
            ).sum()
        ),
        "calibrator_fit_after_signal": int(
            (
                signals["calibrator_fit_date"].fillna("").astype(str)
                > decisions
            ).sum()
        ),
        "evaluation_signal_outside_test": int(
            ((evaluation_dates < test_start) | (evaluation_dates > test_end)).sum()
        ),
        "backtest_row_outside_test": int(
            ((trade_dates < test_start) | (trade_dates > test_end)).sum()
        ),
    }


def _quality_findings(
    run_manifest: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    quality = _mapping(run_manifest.get("quality"), "Run quality")
    checks = quality.get("checks")
    if not isinstance(checks, list):
        raise ConfigurationError("Run quality checks must be a list")
    critical = 0
    warnings: list[dict[str, Any]] = []
    for item in checks:
        check = _mapping(item, "Quality check")
        if check.get("passed") is False and check.get("severity") == "error":
            critical += 1
        if check.get("passed") is False and check.get("severity") == "warning":
            details = _mapping(check.get("details", {}), "Quality warning details")
            warnings.append(
                {
                    "name": check.get("name"),
                    "minimum_coverage": details.get("minimum_coverage"),
                    "minimum_date": details.get("minimum_date"),
                }
            )
    return critical, warnings


def _final_factor_weights(run_root: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(
        run_root / "factor-weight-history.parquet",
        columns=["fit_date", "factor", "allocation_weight"],
    )
    latest = str(frame["fit_date"].astype(str).max())
    selected = frame[
        (frame["fit_date"].astype(str) == latest)
        & (pd.to_numeric(frame["allocation_weight"], errors="coerce") > 1.0e-6)
    ].copy()
    selected["allocation_weight"] = pd.to_numeric(
        selected["allocation_weight"],
        errors="raise",
    )
    selected = selected.sort_values("allocation_weight", ascending=False)
    return [
        {"factor": str(row.factor), "weight": float(row.allocation_weight)}
        for row in selected.itertuples(index=False)
    ]


def _public_calibration_bins(run_root: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(run_root / "calibration-bins.parquet")
    columns = (
        "bin",
        "observations",
        "dates",
        "mean_expected_return",
        "mean_realized_active_return",
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ConfigurationError(f"Calibration bins lack columns: {missing}")
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("bin").itertuples(index=False):
        rows.append(
            {
                "bin": int(row.bin),
                "observations": int(row.observations),
                "dates": int(row.dates),
                "mean_expected_return": float(row.mean_expected_return),
                "mean_realized_active_return": float(
                    row.mean_realized_active_return
                ),
            }
        )
    return rows


def _plot_final_holdout(
    pyplot: Any,
    mdates: Any,
    frame: pd.DataFrame,
    metrics: Mapping[str, Any],
    path: Path,
) -> None:
    daily = frame.copy()
    dates = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    nav = pd.to_numeric(daily["nav"], errors="raise").astype(float)
    benchmark = pd.to_numeric(daily["benchmark_nav"], errors="raise").astype(float)
    relative_gap = nav / benchmark - 1.0
    drawdown = nav / nav.cummax() - 1.0

    fig, axes = pyplot.subplots(
        3,
        1,
        figsize=(12, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 0.9, 0.9]},
    )
    axes[0].plot(dates, nav, color=_BLUE, linewidth=2.0, label="冻结组合")
    axes[0].plot(
        dates,
        benchmark,
        color=_INK,
        linewidth=1.4,
        linestyle="--",
        label="中证 500",
    )
    axes[0].axhline(1.0, color=_GRID, linewidth=0.9)
    axes[0].set_ylabel("累计净值")
    axes[0].legend(loc="upper left", frameon=False, ncols=2)
    axes[0].grid(axis="y", color=_GRID, linewidth=0.8)

    axes[1].fill_between(
        dates,
        relative_gap,
        0.0,
        where=relative_gap <= 0.0,
        color=_RED_OPEN,
        edgecolor="none",
    )
    axes[1].plot(dates, relative_gap, color=_RED, linewidth=1.4)
    axes[1].axhline(0.0, color=_INK, linewidth=0.9)
    axes[1].set_ylabel("相对净值差")
    axes[1].yaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].grid(axis="y", color=_GRID, linewidth=0.8)

    axes[2].fill_between(
        dates,
        drawdown,
        0.0,
        color=_BLUE_OPEN,
        edgecolor="none",
    )
    axes[2].set_ylabel("组合回撤")
    axes[2].yaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[2].grid(axis="y", color=_GRID, linewidth=0.8)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=9)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_GRID)
        axis.set_axisbelow(True)

    fig.suptitle(
        "图 4  最终留出期：冻结组合未取得正主动收益",
        x=0.085,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.895,
        (
            f"{pd.to_datetime(str(metrics['start_date'])).date()}—"
            f"{pd.to_datetime(str(metrics['end_date'])).date()}｜"
            f"{len(daily)} 个交易日｜组合 {float(metrics['total_return']):+.2%}｜"
            f"基准 {float(metrics['benchmark_total_return']):+.2%}｜"
            f"IR {float(metrics['information_ratio']):.2f}"
        ),
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.085,
        0.022,
        (
            "注：组合仅在留出期首个决策日拟合一次，期内不更新；净值已扣除模型化交易成本。"
            "资料来源：Tushare Pro，本项目计算。"
        ),
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.12, hspace=0.14)
    _save_figure(fig, path, pyplot)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigurationError(f"{label} is not valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must contain a JSON object")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    return dict(value)
