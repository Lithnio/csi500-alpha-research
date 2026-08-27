from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.client import TushareClient
from csi500_alpha.data.downloader import (
    DownloadSummary,
    SmokeDownloader,
    build_download_plan,
)
from csi500_alpha.data.eligibility import (
    EligibilityDownloader,
    EligibilityDownloadSummary,
    build_eligibility_download_plan,
)
from csi500_alpha.data.manifest import RequestManifest
from csi500_alpha.data.normalize import build_market_panel
from csi500_alpha.data.quality import QualityReport, load_silver, validate_smoke
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.environment import load_project_environment, require_token
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import BacktestResult, SmokeEventBacktester
from csi500_alpha.features.labels import build_forward_labels
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.evaluation import (
    evaluate_portfolio_run,
    evaluate_research_run,
)
from csi500_alpha.research.factors import compute_reversal_5d
from csi500_alpha.risk.model import LedoitWolfRiskModel
from csi500_alpha.stress import (
    StressContext,
    StressExecution,
    StressResult,
    StressRunner,
    StressScenario,
    StressSpec,
    resolve_stress_config,
    stress_plan,
)
from csi500_alpha.study import (
    StudyContext,
    StudyResult,
    StudyRunner,
    StudySpec,
    StudyTrial,
    TrialExecution,
    resolve_trial_config,
    study_plan,
)
from csi500_alpha.utils import canonical_json, sha256_file, sha256_text, utc_now
from csi500_alpha.workflow.calibration import WalkForwardReturnCalibrationEngine
from csi500_alpha.workflow.components import default_component_registry
from csi500_alpha.workflow.orchestrator import (
    ResearchWorkflow,
    WorkflowResult,
    _industry_exposures,
    _name_history_restrictions,
)
from csi500_alpha.workflow.samples import ResearchSamplePolicy


def _client(config: AppConfig) -> TushareClient:
    load_project_environment(config.paths.root)
    token = require_token(config.source.token_env)
    manifest = RequestManifest(config.paths.data_root / "manifest.sqlite")
    return TushareClient(
        token=token,
        raw_root=config.paths.data_root / "raw",
        manifest=manifest,
        max_attempts=config.source.max_attempts,
        request_timeout_seconds=config.source.request_timeout_seconds,
        backoff_base_seconds=config.source.backoff_base_seconds,
        backoff_max_seconds=config.source.backoff_max_seconds,
        min_request_interval_seconds=(
            config.source.effective_min_request_interval_seconds
        ),
    )


def doctor(config_path: str | Path, *, probe: bool = False) -> dict[str, Any]:
    config = AppConfig.from_yaml(config_path)
    load_project_environment(config.paths.root)
    token_configured = bool(os.environ.get(config.source.token_env, "").strip())
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "tushare", "cvxpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_supported": sys.version_info[:2] == (3, 12),
        "token_configured": token_configured,
        "request_policy": {
            "timeout_seconds": config.source.request_timeout_seconds,
            "calls_per_minute_limit": config.source.calls_per_minute_limit,
            "effective_min_request_interval_seconds": (
                config.source.effective_min_request_interval_seconds
            ),
        },
        "packages": versions,
        "probe": "not-requested",
    }
    if probe:
        client = _client(config)
        frame = client.fetch(
            "trade_cal",
            params={
                "exchange": config.source.exchange,
                "start_date": config.dates.end,
                "end_date": config.dates.end,
            },
            fields=("exchange", "cal_date", "is_open", "pretrade_date"),
        ).frame
        result["probe"] = {
            "api": "trade_cal",
            "date": config.dates.end,
            "rows": len(frame),
            "ok": True,
        }
    return result


def download_smoke(config_path: str | Path, *, force: bool = False) -> DownloadSummary:
    config = AppConfig.from_yaml(config_path)
    return SmokeDownloader(config, _client(config)).run(force=force)


def plan_data_download(config_path: str | Path) -> dict[str, Any]:
    config = AppConfig.from_yaml(config_path)
    return build_download_plan(config).to_dict()


def plan_eligibility_download(config_path: str | Path) -> dict[str, Any]:
    config = AppConfig.from_yaml(config_path)
    return build_eligibility_download_plan(config).to_dict()


def download_eligibility_data(
    config_path: str | Path,
    *,
    force: bool = False,
    refresh_names_from: str | None = None,
) -> EligibilityDownloadSummary:
    config = AppConfig.from_yaml(config_path)
    return EligibilityDownloader(config, _client(config)).run(
        force=force,
        refresh_names_from=refresh_names_from,
    )


def download_data(
    config_path: str | Path,
    *,
    force: bool = False,
    refresh_reference: bool = False,
) -> tuple[DownloadSummary, QualityReport]:
    config = AppConfig.from_yaml(config_path)
    summary = SmokeDownloader(config, _client(config)).run(
        force=force,
        refresh_reference=refresh_reference,
    )
    quality = validate_smoke(config, load_silver(config))
    quality.raise_for_failures()
    return summary, quality


def validate_existing_smoke(config_path: str | Path) -> QualityReport:
    config = AppConfig.from_yaml(config_path)
    report = validate_smoke(config, load_silver(config))
    report.raise_for_failures()
    return report


def run_smoke(config_path: str | Path, *, force: bool = False) -> tuple[str, BacktestResult]:
    config = AppConfig.from_yaml(config_path)
    download_summary = SmokeDownloader(config, _client(config)).run(force=force)
    tables = load_silver(config)
    quality = validate_smoke(config, tables)
    quality.raise_for_failures()
    market_panel = build_market_panel(
        tables["stock_bars"], tables["adjustments"], tables["price_limits"]
    )
    open_dates = tables["calendar"].loc[
        tables["calendar"]["is_open"] == 1, "trade_date"
    ].astype(str).tolist()
    scores = compute_reversal_5d(
        market_panel,
        open_dates,
        window=config.research.factor_window,
    )
    score_features = scores.rename(columns={"trade_date": "decision_date"})[
        ["decision_date", "instrument"]
    ]
    labels = build_forward_labels(
        features=score_features,
        market_panel=market_panel,
        index_bars=tables["index_bars"],
        open_dates=open_dates,
        horizon=config.features.label_horizon,
        suspensions=tables.get("suspensions"),
    )
    registry = default_component_registry()
    calibrator_spec = config.workflow.calibrator
    calibration = WalkForwardReturnCalibrationEngine(
        calibrator_factory=lambda: registry.create_calibrator(
            calibrator_spec.name,
            calibrator_spec.params,
        ),
        refit_every=config.workflow.refit_every,
    ).run(
        scores.rename(columns={"trade_date": "decision_date"}),
        labels,
        prediction_start=config.dates.backtest_start,
        prediction_end=config.dates.end,
    )
    portfolio_signals = calibration.signals.rename(
        columns={"decision_date": "trade_date"}
    )[["trade_date", "instrument", "score", "expected_return"]]
    risk_model = None
    optimizer = None
    if config.optimizer.enabled:
        risk_model = LedoitWolfRiskModel(config.risk, market_panel, open_dates)
        optimizer = ActivePortfolioOptimizer(
            config.optimizer,
            config.research,
            config.risk,
        )
    result = SmokeEventBacktester(
        config.research,
        risk_model=risk_model,
        optimizer=optimizer,
    ).run(
        calendar=tables["calendar"],
        benchmark_weights=tables["benchmark_weights"],
        index_bars=tables["index_bars"],
        market_panel=market_panel,
        signals=portfolio_signals,
        start_date=config.dates.backtest_start,
        end_date=config.dates.end,
    )

    config_hash = sha256_file(config.config_path)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash[:8]}"
    run_root = config.paths.run_root / run_id
    artifact_fingerprints = {
        "signals": write_parquet_atomic(
            portfolio_signals,
            run_root / "signals.parquet",
        ),
        "daily": write_parquet_atomic(result.daily, run_root / "daily.parquet"),
        "trades": write_parquet_atomic(result.trades, run_root / "trades.parquet"),
        "targets": write_parquet_atomic(result.targets, run_root / "targets.parquet"),
        "optimization": write_parquet_atomic(
            result.optimization,
            run_root / "optimization.parquet",
        ),
        "calibration_fits": write_parquet_atomic(
            calibration.calibration_fits,
            run_root / "calibration-fits.parquet",
        ),
        "metrics": write_json_atomic(result.metrics, run_root / "metrics.json"),
    }

    fingerprints = _loaded_silver_fingerprints(config, tables)
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "pipeline": (
            "smoke-reversal5-active-optimizer-event-backtest"
            if config.optimizer.enabled
            else "smoke-reversal5-topn-event-backtest"
        ),
        "config_path": str(config.config_path),
        "config_hash": config_hash,
        "calibrator": {
            "name": config.workflow.calibrator.name,
            "params": config.workflow.calibrator.params,
        },
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": _package_versions(),
        "git": _git_state(config.paths.root),
        "source_tree_hash": _source_tree_hash(config.paths.root),
        "data_fingerprints": fingerprints,
        "artifact_fingerprints": artifact_fingerprints,
        "download": {
            "rows": download_summary.rows,
            "loaded_rows": _loaded_silver_rows(tables),
            "cache_hits": download_summary.cache_hits,
            "network_requests": download_summary.network_requests,
        },
        "quality": quality.to_dict(),
        "metrics_hash": sha256_text(json.dumps(result.metrics, sort_keys=True, default=str)),
    }
    write_json_atomic(manifest, run_root / "run-manifest.json")
    return run_id, result


def run_research_workflow(
    config_path: str | Path,
    *,
    force: bool = False,
    experiment_stage: str | None = None,
    confirm_final_holdout: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run the pluggable research-to-portfolio framework end to end."""
    config = AppConfig.from_yaml(config_path)
    if experiment_stage is not None:
        config = replace(
            config,
            experiment=replace(config.experiment, stage=experiment_stage),
        )
        config.validate()
    _require_final_protocol_authorization(
        config,
        confirmed=confirm_final_holdout,
        git_state=_git_state(config.paths.root),
    )
    protocol = _prepare_experiment_run(config)
    download_summary, tables, quality, data_fingerprints = _research_inputs(
        config,
        force=force,
    )
    protocol = _bind_experiment_data_snapshot(
        config,
        protocol,
        data_fingerprints,
    )
    config_hash = _effective_config_hash(config)
    run_id = _research_run_id(config_hash)
    run_id, summary = _materialize_research_run(
        config,
        run_id=run_id,
        run_root=config.paths.run_root / run_id,
        protocol=protocol,
        download_summary=download_summary,
        tables=tables,
        quality=quality,
        data_fingerprints=data_fingerprints,
    )
    _complete_experiment_run(config, protocol, run_id)
    return run_id, summary


def plan_study(config_path: str | Path) -> dict[str, Any]:
    """Validate and describe a study without loading data or executing trials."""

    spec = StudySpec.from_yaml(config_path)
    base_config = AppConfig.from_yaml(spec.base_config_path)
    if base_config.experiment.stage == "frozen_test":
        raise ConfigurationError("Studies cannot open the frozen-test stage")
    payload = study_plan(config_path)
    resolved_trials: list[dict[str, Any]] = []
    for trial in spec.trials:
        resolved = resolve_trial_config(
            base_config,
            study_id=spec.study_id,
            trial=trial,
        )
        _validate_research_components(resolved)
        resolved_trials.append(
            {
                "trial_id": trial.trial_id,
                "config_hash": _effective_config_hash(resolved),
                "protocol_id": resolved.experiment.protocol_id,
                "components": {
                    "feature_provider": resolved.workflow.feature_provider.name,
                    "selector": resolved.workflow.selector.name,
                    "model": resolved.workflow.model.name,
                    "calibrator": resolved.workflow.calibrator.name,
                },
            }
        )
    payload["resolved_trials"] = resolved_trials
    return payload


def run_study(
    config_path: str | Path,
    *,
    force: bool = False,
) -> StudyResult:
    """Run a bounded, resumable study against one immutable Silver snapshot."""

    spec = StudySpec.from_yaml(config_path)
    base_config = AppConfig.from_yaml(spec.base_config_path)
    if base_config.experiment.stage == "frozen_test":
        raise ConfigurationError("Studies cannot open the frozen-test stage")
    resolved_configs: dict[str, AppConfig] = {}
    for trial in spec.trials:
        resolved = resolve_trial_config(
            base_config,
            study_id=spec.study_id,
            trial=trial,
        )
        _validate_research_components(resolved)
        resolved_configs[trial.trial_id] = resolved
    download_summary, tables, quality, data_fingerprints = _research_inputs(
        base_config,
        force=force,
    )
    source_hash = _research_source_hash(base_config.paths.root)
    data_snapshot_hash = _data_snapshot_hash(data_fingerprints)
    context = StudyContext(
        base_config_hash=sha256_file(base_config.config_path),
        source_hash=source_hash,
        data_snapshot_hash=data_snapshot_hash,
        data_fingerprints=data_fingerprints,
        git=_git_state(base_config.paths.root),
    )

    def execute(trial: StudyTrial, attempt_root: Path) -> TrialExecution:
        trial_config = resolved_configs[trial.trial_id]
        protocol = {
            "research_spec_hash": sha256_text(
                canonical_json(_research_spec(trial_config))
            ),
            "research_source_hash": source_hash,
            "data_snapshot_hash": data_snapshot_hash,
            "data_fingerprints": dict(sorted(data_fingerprints.items())),
            "study_id": spec.study_id,
            "trial_id": trial.trial_id,
        }
        config_hash = _effective_config_hash(trial_config)
        run_id = _research_run_id(config_hash)
        run_id, summary = _materialize_research_run(
            trial_config,
            run_id=run_id,
            run_root=attempt_root,
            protocol=protocol,
            download_summary=download_summary,
            tables=tables,
            quality=quality,
            data_fingerprints=data_fingerprints,
        )
        return TrialExecution(
            run_id=run_id,
            config_hash=config_hash,
            summary=summary,
        )

    return StudyRunner(
        spec,
        study_root=base_config.paths.root / "studies" / spec.study_id,
        context=context,
        executor=execute,
    ).run()


def plan_portfolio_stress(config_path: str | Path) -> dict[str, Any]:
    """Validate a bounded one-way stress matrix without requiring study outputs."""

    spec = StressSpec.from_yaml(config_path)
    source_spec = StudySpec.from_yaml(spec.source_study_path)
    base_config = AppConfig.from_yaml(source_spec.base_config_path)
    if base_config.experiment.stage == "frozen_test":
        raise ConfigurationError("Stress analysis cannot open the frozen-test stage")
    scenarios: list[dict[str, Any]] = []
    for scenario in spec.scenarios:
        resolved = resolve_stress_config(base_config, scenario)
        scenarios.append(
            {
                **scenario.to_dict(),
                "resolved": {
                    "linear_cost_bps": resolved.research.linear_cost_bps,
                    "stamp_duty_before": resolved.research.stamp_duty_before,
                    "stamp_duty_after": resolved.research.stamp_duty_after,
                    "portfolio_aum_cny": resolved.optimizer.portfolio_aum_cny,
                    "max_adv_participation": (
                        resolved.optimizer.max_adv_participation
                    ),
                    "impact_bps_at_max_participation": (
                        resolved.optimizer.impact_bps_at_max_participation
                    ),
                },
            }
        )
    return {
        **stress_plan(config_path),
        "source_study_id": source_spec.study_id,
        "scenarios": scenarios,
    }


def run_portfolio_stress(config_path: str | Path) -> StressResult:
    """Stress the selected trial while freezing its signal and calibration stream."""

    spec = StressSpec.from_yaml(config_path)
    source = _selected_study_run(spec)
    current_source_hash = _research_source_hash(source.base_config.paths.root)
    recorded_source_hash = str(source.trial_manifest.get("source_hash", ""))
    if current_source_hash != recorded_source_hash:
        raise ConfigurationError(
            "Stress source code differs from the selected Study trial; rerun the "
            "Study before stressing it"
        )
    tables = load_silver(source.base_config)
    quality = validate_smoke(source.base_config, tables)
    quality.raise_for_failures()
    data_fingerprints = _loaded_silver_fingerprints(source.base_config, tables)
    data_snapshot_hash = _data_snapshot_hash(data_fingerprints)
    recorded_snapshot = str(source.trial_manifest.get("data_snapshot_hash", ""))
    if data_snapshot_hash != recorded_snapshot:
        raise ConfigurationError(
            "Stress Silver snapshot differs from the selected Study trial"
        )

    evaluation_signals = pd.read_parquet(
        source.attempt_root / "evaluation-signals.parquet"
    )
    features = pd.read_parquet(source.attempt_root / "gold" / "features.parquet")
    source_metrics = _read_json_object(source.attempt_root / "metrics.json")
    market_panel = build_market_panel(
        tables["stock_bars"],
        tables["adjustments"],
        tables["price_limits"],
    )
    open_dates = (
        tables["calendar"]
        .loc[tables["calendar"]["is_open"] == 1, "trade_date"]
        .astype(str)
        .tolist()
    )
    policy = ResearchSamplePolicy(source.config.experiment, tuple(open_dates))
    portfolio_signals = evaluation_signals.rename(
        columns={"decision_date": "trade_date"}
    )[["trade_date", "instrument", "score", "expected_return"]]
    signal_dates = sorted(portfolio_signals["trade_date"].astype(str).unique())
    if not signal_dates:
        raise ConfigurationError("Selected Study trial has no evaluation signals")
    exposures = _industry_exposures(features)
    restrictions = _name_history_restrictions(
        features,
        tables.get("name_history"),
    )
    source_manifest_hash = sha256_file(source.attempt_root / "run-manifest.json")
    context = StressContext(
        source_study_id=source.study_spec.study_id,
        source_trial_id=source.trial.trial_id,
        source_resolved_config_hash=str(
            source.trial_manifest["resolved_config_hash"]
        ),
        source_run_manifest_hash=source_manifest_hash,
        source_hash=current_source_hash,
        data_snapshot_hash=data_snapshot_hash,
    )

    def execute(scenario: StressScenario, attempt_root: Path) -> StressExecution:
        scenario_config = resolve_stress_config(source.config, scenario)
        risk_model = (
            LedoitWolfRiskModel(
                scenario_config.risk,
                market_panel,
                open_dates,
            )
            if scenario_config.optimizer.enabled
            else None
        )
        optimizer = (
            ActivePortfolioOptimizer(
                scenario_config.optimizer,
                scenario_config.research,
                scenario_config.risk,
            )
            if scenario_config.optimizer.enabled
            else None
        )
        backtest = SmokeEventBacktester(
            scenario_config.research,
            risk_model=risk_model,
            optimizer=optimizer,
        ).run(
            calendar=tables["calendar"],
            benchmark_weights=tables["benchmark_weights"],
            index_bars=tables["index_bars"],
            market_panel=market_panel,
            signals=portfolio_signals,
            portfolio_exposures=exposures,
            portfolio_restrictions=restrictions,
            suspensions=tables.get("suspensions"),
            start_date=signal_dates[0],
            end_date=policy.evaluation_end,
        )
        portfolio_evaluation = evaluate_portfolio_run(
            daily=backtest.daily,
            trades=backtest.trades,
        )
        optimization = backtest.optimization
        solved = (
            int(optimization["status"].isin(["optimal", "optimal_inaccurate"]).sum())
            if not optimization.empty
            else 0
        )
        metric_difference = _maximum_metric_difference(
            source_metrics,
            backtest.metrics,
        )
        parity = (
            metric_difference <= spec.baseline_parity_tolerance
            if scenario.scenario_id == "baseline"
            else None
        )
        artifact_fingerprints = {
            "daily": write_parquet_atomic(backtest.daily, attempt_root / "daily.parquet"),
            "trades": write_parquet_atomic(
                backtest.trades,
                attempt_root / "trades.parquet",
            ),
            "targets": write_parquet_atomic(
                backtest.targets,
                attempt_root / "targets.parquet",
            ),
            "optimization": write_parquet_atomic(
                optimization,
                attempt_root / "optimization.parquet",
            ),
            "yearly_metrics": write_parquet_atomic(
                portfolio_evaluation.yearly_metrics,
                attempt_root / "yearly-metrics.parquet",
            ),
            "metrics": write_json_atomic(
                backtest.metrics,
                attempt_root / "metrics.json",
            ),
        }
        summary = {
            "scenario": scenario.to_dict(),
            "resolved_assumptions": {
                "linear_cost_bps": scenario_config.research.linear_cost_bps,
                "stamp_duty_before": scenario_config.research.stamp_duty_before,
                "stamp_duty_after": scenario_config.research.stamp_duty_after,
                "portfolio_aum_cny": scenario_config.optimizer.portfolio_aum_cny,
                "max_adv_participation": (
                    scenario_config.optimizer.max_adv_participation
                ),
                "impact_bps_at_max_participation": (
                    scenario_config.optimizer.impact_bps_at_max_participation
                ),
            },
            "portfolio_dates": int(
                evaluation_signals["decision_date"].astype(str).nunique()
            ),
            "optimizer_attempts": int(len(optimization)),
            "optimizer_solved": solved,
            "optimizer_solve_rate": (
                solved / len(optimization) if not optimization.empty else 1.0
            ),
            "source_metric_max_abs_difference": metric_difference,
            "source_metric_parity_passed": parity,
            "metrics": backtest.metrics,
            "evaluation": portfolio_evaluation.summary,
            "artifact_fingerprints": artifact_fingerprints,
        }
        write_json_atomic(summary, attempt_root / "scenario-summary.json")
        return StressExecution(summary=summary)

    stress_root = (
        source.study_root
        / "stress"
        / spec.stress_id
    )
    return StressRunner(
        spec,
        stress_root=stress_root,
        context=context,
        executor=execute,
    ).run()


@dataclass(frozen=True)
class _SelectedStudyRun:
    study_spec: StudySpec
    base_config: AppConfig
    trial: StudyTrial
    config: AppConfig
    study_root: Path
    attempt_root: Path
    trial_manifest: dict[str, Any]
    run_manifest: dict[str, Any]


def _selected_study_run(spec: StressSpec) -> _SelectedStudyRun:
    study_spec = StudySpec.from_yaml(spec.source_study_path)
    base_config = AppConfig.from_yaml(study_spec.base_config_path)
    study_root = base_config.paths.root / "studies" / study_spec.study_id
    study_manifest = _read_json_object(study_root / "study-manifest.json")
    if study_manifest.get("spec_hash") != study_spec.spec_hash:
        raise ConfigurationError(
            "Stress source Study manifest does not match the current Study spec"
        )
    if not str(study_manifest.get("status", "")).startswith("completed"):
        raise ConfigurationError("Stress source Study is not complete")
    selection = _read_json_object(study_root / "selection.json")
    selected_trial_id = selection.get("selected_trial_id")
    if not isinstance(selected_trial_id, str) or not selected_trial_id:
        raise ConfigurationError(
            "Stress source Study has no selected completed trial"
        )
    trial = next(
        (item for item in study_spec.trials if item.trial_id == selected_trial_id),
        None,
    )
    if trial is None:
        raise ConfigurationError(
            f"Selected trial {selected_trial_id!r} is absent from the source Study spec"
        )
    trial_manifest_path = (
        study_root / "trials" / selected_trial_id / "trial-manifest.json"
    )
    trial_manifest = _read_json_object(trial_manifest_path)
    if trial_manifest.get("status") != "completed":
        raise ConfigurationError(
            f"Selected trial {selected_trial_id!r} is not completed"
        )
    artifact_reference = trial_manifest.get("artifact_root")
    if not isinstance(artifact_reference, str) or not artifact_reference:
        raise ConfigurationError("Selected trial manifest lacks artifact_root")
    attempt_root = (study_root / artifact_reference).resolve()
    if not attempt_root.is_relative_to(study_root.resolve()):
        raise ConfigurationError("Selected trial artifact_root escapes the Study root")
    required_artifacts = (
        attempt_root / "run-manifest.json",
        attempt_root / "metrics.json",
        attempt_root / "evaluation-signals.parquet",
        attempt_root / "gold" / "features.parquet",
    )
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise ConfigurationError(
            f"Selected trial is missing required stress artifacts: {missing}"
        )
    config = resolve_trial_config(
        base_config,
        study_id=study_spec.study_id,
        trial=trial,
    )
    expected_hash = _effective_config_hash(config)
    if trial_manifest.get("resolved_config_hash") != expected_hash:
        raise ConfigurationError(
            "Selected trial resolved configuration differs from its manifest"
        )
    run_manifest = _read_json_object(attempt_root / "run-manifest.json")
    if run_manifest.get("config_hash") != expected_hash:
        raise ConfigurationError(
            "Selected trial run manifest has a different resolved configuration"
        )
    artifact_fingerprints = run_manifest.get("artifact_fingerprints")
    gold_fingerprints = run_manifest.get("gold_fingerprints")
    if not isinstance(artifact_fingerprints, Mapping) or not isinstance(
        gold_fingerprints,
        Mapping,
    ):
        raise ConfigurationError(
            "Selected trial run manifest lacks artifact fingerprints"
        )
    _assert_artifact_fingerprint(
        attempt_root / "evaluation-signals.parquet",
        artifact_fingerprints.get("evaluation_signals"),
    )
    _assert_artifact_fingerprint(
        attempt_root / "metrics.json",
        artifact_fingerprints.get("metrics"),
    )
    _assert_artifact_fingerprint(
        attempt_root / "gold" / "features.parquet",
        gold_fingerprints.get("features"),
    )
    return _SelectedStudyRun(
        study_spec=study_spec,
        base_config=base_config,
        trial=trial,
        config=config,
        study_root=study_root,
        attempt_root=attempt_root,
        trial_manifest=trial_manifest,
        run_manifest=run_manifest,
    )


def _maximum_metric_difference(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> float:
    differences: list[float] = []
    for key, raw_source in source.items():
        if isinstance(raw_source, bool):
            continue
        try:
            source_value = float(raw_source)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(source_value):
            continue
        if key not in candidate:
            return float("inf")
        try:
            candidate_value = float(candidate[key])
        except (TypeError, ValueError):
            return float("inf")
        if not np.isfinite(candidate_value):
            if np.isnan(source_value) and np.isnan(candidate_value):
                continue
            return float("inf")
        differences.append(abs(source_value - candidate_value))
    return max(differences, default=0.0)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON artifact must contain an object: {path}")
    return value


def _assert_artifact_fingerprint(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or not expected:
        raise ConfigurationError(
            f"Selected trial manifest lacks a fingerprint for {path.name}"
        )
    if sha256_file(path) != expected:
        raise ConfigurationError(f"Selected trial artifact fingerprint differs: {path}")


def _research_inputs(
    config: AppConfig,
    *,
    force: bool,
) -> tuple[DownloadSummary, dict[str, Any], QualityReport, dict[str, str]]:
    if not config.download.include_daily_basic:
        raise ConfigurationError(
            "Research workflow requires download.include_daily_basic=true"
        )
    download_summary = SmokeDownloader(config, _client(config)).run(force=force)
    tables = load_silver(config)
    quality = validate_smoke(config, tables)
    quality.raise_for_failures()
    data_fingerprints = _loaded_silver_fingerprints(config, tables)
    return download_summary, tables, quality, data_fingerprints


def _validate_research_components(config: AppConfig) -> None:
    registry = default_component_registry()
    provider_spec = config.workflow.feature_provider
    provider = registry.create_feature_provider(
        provider_spec.name,
        provider_spec.params,
    )
    factor_names = config.workflow.factor_names or provider.factor_names
    missing = sorted(set(factor_names).difference(provider.factor_names))
    if missing:
        raise ConfigurationError(
            f"Feature provider lacks requested factors: {missing}"
        )
    directions = {name: int(provider.directions[name]) for name in factor_names}
    selector_spec = config.workflow.selector
    registry.create_selector(selector_spec.name, selector_spec.params, directions)
    model_spec = config.workflow.model
    registry.create_model(model_spec.name, model_spec.params, directions)
    calibrator_spec = config.workflow.calibrator
    registry.create_calibrator(calibrator_spec.name, calibrator_spec.params)


def _materialize_research_run(
    config: AppConfig,
    *,
    run_id: str,
    run_root: Path,
    protocol: dict[str, Any],
    download_summary: DownloadSummary,
    tables: dict[str, Any],
    quality: QualityReport,
    data_fingerprints: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    result = ResearchWorkflow(config).run(tables)
    _assert_stage_artifact_boundaries(config, result)
    evaluation = evaluate_research_run(
        signals=result.evaluation_signals,
        labels=result.labels,
        daily=result.backtest.daily,
        trades=result.backtest.trades,
        model_fits=result.model_fits,
        calibrator_name=config.workflow.calibrator.name,
        calibrator_params=config.workflow.calibrator.params,
    )

    config_hash = _effective_config_hash(config)
    gold_root = run_root / "gold"
    gold_fingerprints = {
        "raw_features": write_parquet_atomic(
            result.raw_features,
            gold_root / "raw_features.parquet",
        ),
        "features": write_parquet_atomic(
            result.processed.features,
            gold_root / "features.parquet",
        ),
        "feature_quality": write_parquet_atomic(
            result.processed.quality,
            gold_root / "feature_quality.parquet",
        ),
        "labels": write_parquet_atomic(result.labels, gold_root / "labels.parquet"),
        "research_panel": write_parquet_atomic(
            result.research_panel,
            gold_root / "research_panel.parquet",
        ),
    }

    correlation = result.diagnostics.correlation.rename_axis("factor").reset_index()
    artifact_fingerprints = {
        "signals": write_parquet_atomic(result.signals, run_root / "signals.parquet"),
        "evaluation_signals": write_parquet_atomic(
            result.evaluation_signals,
            run_root / "evaluation-signals.parquet",
        ),
        "model_fits": write_parquet_atomic(
            result.model_fits,
            run_root / "model-fits.parquet",
        ),
        "calibration_fits": write_parquet_atomic(
            result.calibration_fits,
            run_root / "calibration-fits.parquet",
        ),
        "factor_ic": write_parquet_atomic(
            result.diagnostics.ic_by_date,
            run_root / "factor-ic.parquet",
        ),
        "factor_summary": write_parquet_atomic(
            result.diagnostics.summary,
            run_root / "factor-summary.parquet",
        ),
        "quintile_returns": write_parquet_atomic(
            result.diagnostics.quintile_returns,
            run_root / "quintile-returns.parquet",
        ),
        "factor_correlation": write_parquet_atomic(
            correlation,
            run_root / "factor-correlation.parquet",
        ),
        "calibration_bins": write_parquet_atomic(
            evaluation.calibration_bins,
            run_root / "calibration-bins.parquet",
        ),
        "yearly_metrics": write_parquet_atomic(
            evaluation.yearly_metrics,
            run_root / "yearly-metrics.parquet",
        ),
        "factor_weight_history": write_parquet_atomic(
            evaluation.factor_weight_history,
            run_root / "factor-weight-history.parquet",
        ),
        "research_evaluation": write_json_atomic(
            evaluation.summary,
            run_root / "research-evaluation.json",
        ),
        "daily": write_parquet_atomic(
            result.backtest.daily,
            run_root / "daily.parquet",
        ),
        "trades": write_parquet_atomic(
            result.backtest.trades,
            run_root / "trades.parquet",
        ),
        "targets": write_parquet_atomic(
            result.backtest.targets,
            run_root / "targets.parquet",
        ),
        "optimization": write_parquet_atomic(
            result.backtest.optimization,
            run_root / "optimization.parquet",
        ),
        "metrics": write_json_atomic(
            result.backtest.metrics,
            run_root / "metrics.json",
        ),
    }
    fitted = (
        int((result.model_fits["status"] == "fitted").sum())
        if not result.model_fits.empty
        else 0
    )
    optimization = result.backtest.optimization
    solved = (
        int(optimization["status"].isin(["optimal", "optimal_inaccurate"]).sum())
        if not optimization.empty
        else 0
    )
    summary = {
        "dataset": config.paths.dataset,
        "feature_provider": config.workflow.feature_provider.name,
        "selector": config.workflow.selector.name,
        "model": config.workflow.model.name,
        "calibrator": config.workflow.calibrator.name,
        "factors": list(result.factor_names),
        "feature_dates": int(result.processed.features["decision_date"].nunique()),
        "experiment_stage": config.experiment.stage,
        "protocol_id": config.experiment.protocol_id,
        "portfolio_dates": int(
            result.evaluation_signals["decision_date"].nunique()
        ),
        "feature_rows": int(len(result.processed.features)),
        "valid_signals": int(result.evaluation_signals["score"].notna().sum()),
        "valid_expected_returns": int(
            result.evaluation_signals["expected_return"].notna().sum()
        ),
        "model_fits": fitted,
        "calibration_fits": int(
            (result.calibration_fits["status"] == "fitted").sum()
        ),
        "optimizer_attempts": int(len(optimization)),
        "optimizer_solved": solved,
        "optimizer_solve_rate": (
            solved / len(optimization) if not optimization.empty else 1.0
        ),
        "quality_passed": not quality.critical_failures,
        "network_requests": download_summary.network_requests,
        "cache_hits": download_summary.cache_hits,
        "artifact_date_ranges": {
            "features": _date_range(result.processed.features, "decision_date"),
            "labels": _date_range(result.labels, "decision_date"),
            "label_availability": _date_range(
                result.labels,
                "label_available_date",
            ),
            "factor_diagnostics": _date_range(
                result.diagnostics.ic_by_date,
                "decision_date",
            ),
            "signals": _date_range(result.signals, "decision_date"),
            "evaluation_signals": _date_range(
                result.evaluation_signals,
                "decision_date",
            ),
            "backtest": _date_range(result.backtest.daily, "trade_date"),
        },
        "metrics": result.backtest.metrics,
        "evaluation": evaluation.summary,
    }
    artifact_fingerprints["workflow_summary"] = write_json_atomic(
        summary,
        run_root / "workflow-summary.json",
    )
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "pipeline": "pluggable-point-in-time-research-workflow",
        "config_path": str(config.config_path),
        "config_hash": config_hash,
        "config_file_hash": sha256_file(config.config_path),
        "dataset": config.paths.dataset,
        "experiment": {
            **result.sample_policy,
            "research_spec_hash": protocol["research_spec_hash"],
            "research_source_hash": protocol["research_source_hash"],
            "data_snapshot_hash": protocol["data_snapshot_hash"],
        },
        **(
            {
                "study": {
                    "study_id": protocol["study_id"],
                    "trial_id": protocol["trial_id"],
                }
            }
            if "study_id" in protocol
            else {}
        ),
        "components": {
            "feature_provider": {
                "name": config.workflow.feature_provider.name,
                "params": config.workflow.feature_provider.params,
            },
            "selector": {
                "name": config.workflow.selector.name,
                "params": config.workflow.selector.params,
            },
            "model": {
                "name": config.workflow.model.name,
                "params": config.workflow.model.params,
            },
            "calibrator": {
                "name": config.workflow.calibrator.name,
                "params": config.workflow.calibrator.params,
            },
        },
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": _package_versions(),
        "git": _git_state(config.paths.root),
        "source_tree_hash": _source_tree_hash(config.paths.root),
        "data_fingerprints": data_fingerprints,
        "gold_root": "gold",
        "gold_fingerprints": gold_fingerprints,
        "artifact_fingerprints": artifact_fingerprints,
        "download": {
            "rows": download_summary.rows,
            "loaded_rows": _loaded_silver_rows(tables),
            "cache_hits": download_summary.cache_hits,
            "network_requests": download_summary.network_requests,
        },
        "quality": quality.to_dict(),
        "summary": summary,
    }
    write_json_atomic(manifest, run_root / "run-manifest.json")
    return run_id, summary


def _effective_config_hash(config: AppConfig) -> str:
    return sha256_text(canonical_json(_research_spec(config)))


def _research_run_id(config_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{config_hash[:8]}"


def _research_spec(config: AppConfig) -> dict[str, Any]:
    experiment = asdict(config.experiment)
    experiment.pop("stage", None)
    return {
        "dataset": config.paths.dataset,
        "index_code": config.source.index_code,
        "dates": asdict(config.dates),
        "research": asdict(config.research),
        "risk": asdict(config.risk),
        "optimizer": asdict(config.optimizer),
        "features": asdict(config.features),
        "workflow": asdict(config.workflow),
        "experiment": experiment,
    }


def _assert_stage_artifact_boundaries(
    config: AppConfig,
    result: WorkflowResult,
) -> None:
    """Reject any formal run whose artifacts cross the visible stage boundary."""

    if not config.experiment.enabled:
        return
    stage_end = (
        config.experiment.validation_end
        if config.experiment.stage == "validation"
        else config.experiment.test_end
    )
    artifacts = {
        "raw_features": (result.raw_features, "decision_date"),
        "features": (result.processed.features, "decision_date"),
        "feature_quality": (result.processed.quality, "decision_date"),
        "labels": (result.labels, "decision_date"),
        "label_availability": (result.labels, "label_available_date"),
        "research_panel": (result.research_panel, "decision_date"),
        "factor_ic": (result.diagnostics.ic_by_date, "decision_date"),
        "quintile_returns": (
            result.diagnostics.quintile_returns,
            "decision_date",
        ),
        "signals": (result.signals, "decision_date"),
        "evaluation_signals": (result.evaluation_signals, "decision_date"),
        "backtest_daily": (result.backtest.daily, "trade_date"),
        "targets": (result.backtest.targets, "execution_date"),
        "trades": (result.backtest.trades, "trade_date"),
    }
    crossed = {
        name: bounds["end"]
        for name, (frame, column) in artifacts.items()
        if (bounds := _date_range(frame, column))["end"] is not None
        and str(bounds["end"]) > stage_end
    }
    if crossed:
        raise ConfigurationError(
            f"{config.experiment.stage} artifacts cross stage_end={stage_end}: {crossed}"
        )


def _date_range(frame: Any, column: str) -> dict[str, str | None]:
    if not hasattr(frame, "columns") or column not in frame.columns or frame.empty:
        return {"start": None, "end": None}
    values = frame[column].dropna().astype(str)
    if values.empty:
        return {"start": None, "end": None}
    return {"start": str(values.min()), "end": str(values.max())}


def _protocol_paths(config: AppConfig) -> tuple[Path, Path]:
    root = config.paths.run_root / "_protocols" / config.experiment.protocol_id
    return root / "validation-lock.json", root / "frozen-test.json"


def _prepare_experiment_run(config: AppConfig) -> dict[str, Any]:
    if (
        config.experiment.stage == "frozen_test"
        and not config.experiment.allow_frozen_test
    ):
        raise ConfigurationError(
            "frozen_test is disabled for this protocol config; use the dedicated "
            "final-holdout config"
        )
    spec_hash = sha256_text(canonical_json(_research_spec(config)))
    source_hash = _research_source_hash(config.paths.root)
    context = {
        "research_spec_hash": spec_hash,
        "research_source_hash": source_hash,
    }
    if not config.experiment.enabled:
        return context

    validation_path, frozen_path = _protocol_paths(config)
    if frozen_path.exists():
        raise ConfigurationError(
            "This protocol already has a completed frozen test; create a new "
            "experiment.protocol_id for another research iteration"
        )
    if config.experiment.stage == "validation":
        return context
    if not validation_path.exists():
        raise ConfigurationError(
            "Frozen test requires a completed validation run for the same protocol_id"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("research_spec_hash") != spec_hash:
        raise ConfigurationError(
            "Frozen-test research settings differ from the validated protocol"
        )
    if validation.get("research_source_hash") != source_hash:
        raise ConfigurationError(
            "Frozen-test source code differs from the validated protocol"
        )
    context["validation_run_id"] = validation.get("run_id")
    return context


def _require_final_protocol_authorization(
    config: AppConfig,
    *,
    confirmed: bool,
    git_state: Mapping[str, Any],
) -> None:
    """Require a committed protocol and an explicit second key for final holdout."""

    is_frozen = config.experiment.stage == "frozen_test"
    if not is_frozen and not config.experiment.allow_frozen_test:
        return
    if is_frozen and not confirmed:
        raise ConfigurationError(
            "Frozen test requires explicit --confirm-final-holdout authorization"
        )
    commit = git_state.get("commit")
    if (
        git_state.get("inside_work_tree") is not True
        or git_state.get("dirty") is not False
        or not isinstance(commit, str)
        or not commit
    ):
        raise ConfigurationError(
            "Final protocol requires a clean Git worktree at a recorded commit"
        )


def _complete_experiment_run(
    config: AppConfig,
    protocol: dict[str, Any],
    run_id: str,
) -> None:
    if not config.experiment.enabled:
        return
    if "data_snapshot_hash" not in protocol or "data_fingerprints" not in protocol:
        raise ConfigurationError(
            "Experiment completion requires a bound Silver data snapshot"
        )
    validation_path, frozen_path = _protocol_paths(config)
    payload = {
        "protocol_id": config.experiment.protocol_id,
        "stage": config.experiment.stage,
        "run_id": run_id,
        "completed_at": utc_now(),
        **protocol,
    }
    destination = (
        validation_path
        if config.experiment.stage == "validation"
        else frozen_path
    )
    write_json_atomic(payload, destination)


def _bind_experiment_data_snapshot(
    config: AppConfig,
    protocol: dict[str, Any],
    data_fingerprints: dict[str, str],
) -> dict[str, Any]:
    """Bind the exact consumed Silver snapshot before any research evaluation."""

    ordered_fingerprints = dict(sorted(data_fingerprints.items()))
    snapshot_hash = _data_snapshot_hash(ordered_fingerprints)
    context = {
        **protocol,
        "data_snapshot_hash": snapshot_hash,
        "data_fingerprints": ordered_fingerprints,
    }
    if not config.experiment.enabled or config.experiment.stage == "validation":
        return context

    validation_path, _ = _protocol_paths(config)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validated_hash = validation.get("data_snapshot_hash")
    if not validated_hash:
        raise ConfigurationError(
            "Frozen-test validation lock predates Silver snapshot locking; "
            "rerun validation before opening the frozen test"
        )
    if validated_hash != snapshot_hash:
        validated_tables = validation.get("data_fingerprints", {})
        changed_tables = sorted(
            name
            for name in set(validated_tables).union(ordered_fingerprints)
            if validated_tables.get(name) != ordered_fingerprints.get(name)
        )
        raise ConfigurationError(
            "Frozen-test Silver data differs from the validated snapshot; "
            f"changed tables: {changed_tables}"
        )
    return context


def _data_snapshot_hash(data_fingerprints: Mapping[str, str]) -> str:
    return sha256_text(canonical_json(dict(sorted(data_fingerprints.items()))))


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "tushare", "cvxpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_state(root: Path) -> dict[str, Any]:
    try:
        inside_work_tree = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == "true"
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = head.stdout.strip() if head.returncode == 0 else None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        result: dict[str, Any] = {
            "inside_work_tree": inside_work_tree,
            "commit": commit,
            "dirty": dirty,
        }
        if commit is None:
            result["note"] = "unborn-git-worktree"
        return result
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "inside_work_tree": False,
            "commit": None,
            "dirty": None,
            "note": "not-a-git-worktree",
        }


def _source_tree_hash(root: Path) -> str:
    """Fingerprint tracked project inputs even before the first Git commit."""
    candidates = [root / "pyproject.toml"]
    candidates.extend(sorted((root / "configs").glob("*.yaml")))
    candidates.extend(sorted((root / "src").rglob("*.py")))
    fingerprints = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    return sha256_text(json.dumps(fingerprints, sort_keys=True))


def _loaded_silver_fingerprints(
    config: AppConfig,
    tables: dict[str, Any],
) -> dict[str, str]:
    """Fingerprint every Silver table actually consumed by a run."""

    return {
        name: sha256_file(config.paths.silver_root / f"{name}.parquet")
        for name in sorted(tables)
    }


def _loaded_silver_rows(tables: dict[str, Any]) -> dict[str, int]:
    return {name: int(len(table)) for name, table in sorted(tables.items())}


def _research_source_hash(root: Path) -> str:
    """Fingerprint executable research code without unrelated experiment YAMLs."""
    candidates = [root / "pyproject.toml"]
    candidates.extend(sorted((root / "src").rglob("*.py")))
    fingerprints = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    return sha256_text(json.dumps(fingerprints, sort_keys=True))
