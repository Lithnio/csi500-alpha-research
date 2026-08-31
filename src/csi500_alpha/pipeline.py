from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.annual import (
    AnnualFold,
    AnnualFoldExecution,
    AnnualStudyResult,
    AnnualStudyRunner,
    AnnualStudySpec,
    AnnualTrialAggregate,
    annual_cost_contract,
    annual_feature_contract,
    annual_study_plan,
    build_annual_folds,
    resolve_annual_fold_config,
)
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
from csi500_alpha.data.financial import (
    FINANCIAL_CONTRACT_VERSION,
    FinancialDownloader,
    FinancialDownloadSpec,
    FinancialDownloadSummary,
    build_financial_download_plan,
)
from csi500_alpha.data.manifest import RequestManifest
from csi500_alpha.data.normalize import build_market_panel
from csi500_alpha.data.quality import QualityReport, load_silver, validate_smoke
from csi500_alpha.data.storage import write_json_atomic, write_parquet_atomic
from csi500_alpha.environment import load_project_environment, require_token
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import (
    BacktestResult,
    SmokeEventBacktester,
    calculate_backtest_metrics,
    enrich_active_performance,
)
from csi500_alpha.features.builder import ProcessedFactors
from csi500_alpha.features.labels import build_forward_labels
from csi500_alpha.logging_utils import ProgressCallback, emit_progress
from csi500_alpha.portfolio.audit import summarize_constraint_audits
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.diagnostics import FactorDiagnostics
from csi500_alpha.research.evaluation import (
    evaluate_portfolio_run,
    evaluate_research_run,
)
from csi500_alpha.research.factor_audit import (
    FACTOR_AUDIT_CONTRACT_VERSION,
    FactorAuditSpec,
    build_factor_audit_tables,
    combined_factor_families,
    factor_catalog_frame,
    finite_or_none,
)
from csi500_alpha.research.factors import compute_reversal_5d
from csi500_alpha.risk.model import build_risk_model
from csi500_alpha.stress import (
    StressContext,
    StressExecution,
    StressResult,
    StressRunner,
    StressScenario,
    StressSpec,
    replay_frozen_trade_costs,
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
    PreparedResearchData,
    ResearchWorkflow,
    WorkflowResult,
    _industry_exposures,
    _name_history_restrictions,
    _style_exposures,
)
from csi500_alpha.workflow.samples import ResearchSamplePolicy

LOGGER = logging.getLogger(__name__)


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


def plan_financial_download(config_path: str | Path) -> dict[str, Any]:
    spec = FinancialDownloadSpec.from_yaml(config_path)
    manifest_path = (
        spec.quality_root / f"{spec.output_subdirectory}-dataset-manifest.json"
    )
    existing_contract: str | None = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_contract = str(manifest.get("contract_version") or "") or None
        except (json.JSONDecodeError, OSError):
            existing_contract = None
    return {
        "contract_version": FINANCIAL_CONTRACT_VERSION,
        "spec": spec.to_dict(),
        "plan": build_financial_download_plan(spec).to_dict(),
        "existing_snapshot": {
            "manifest_path": str(manifest_path),
            "contract_version": existing_contract,
            "rematerialization_required": (
                existing_contract != FINANCIAL_CONTRACT_VERSION
            ),
        },
    }


def download_financial_data(
    config_path: str | Path,
    *,
    force: bool = False,
) -> FinancialDownloadSummary:
    spec = FinancialDownloadSpec.from_yaml(config_path)
    return FinancialDownloader(spec, _client(spec.base_config)).run(force=force)


def plan_factor_audit(config_path: str | Path) -> dict[str, Any]:
    """Validate and describe the offline factor audit without loading Silver data."""

    spec = FactorAuditSpec.from_yaml(config_path)
    config = spec.resolved_config()
    registry = default_component_registry()
    provider_spec = config.workflow.feature_provider
    provider = registry.create_feature_provider(
        provider_spec.name,
        provider_spec.params,
    )
    factor_names = config.workflow.factor_names or provider.factor_names
    missing = sorted(set(factor_names).difference(provider.factor_names))
    if missing:
        raise ConfigurationError(f"Factor audit provider lacks factors: {missing}")
    families = combined_factor_families()
    financial_quality_path = config.paths.quality_root / "financial-data-quality.json"
    financial_contract: str | None = None
    financial_validation_passed = False
    if financial_quality_path.is_file():
        try:
            financial_quality = json.loads(
                financial_quality_path.read_text(encoding="utf-8")
            )
            financial_contract = (
                str(financial_quality.get("contract_version") or "") or None
            )
            financial_validation_passed = bool(
                financial_quality.get("status") == "success"
                and financial_quality.get("validation", {}).get("passed", False)
            )
        except (json.JSONDecodeError, OSError):
            financial_contract = None
    return {
        "contract_version": FACTOR_AUDIT_CONTRACT_VERSION,
        "spec": spec.to_dict()["factor_audit"],
        "dataset": config.paths.dataset,
        "factor_count": len(factor_names),
        "factor_family_count": len({families[name] for name in factor_names}),
        "factors": list(factor_names),
        "rebalance_every_open_days": config.research.rebalance_every,
        "label_horizon_open_days": config.features.label_horizon,
        "network_requests": 0,
        "output_parent": str(config.paths.run_root / spec.output_subdirectory),
        "financial_snapshot": {
            "quality_path": str(financial_quality_path),
            "contract_version": financial_contract,
            "required_contract_version": FINANCIAL_CONTRACT_VERSION,
            "ready": (
                financial_contract == FINANCIAL_CONTRACT_VERSION
                and financial_validation_passed
            ),
        },
    }


def run_factor_audit(config_path: str | Path) -> dict[str, Any]:
    """Run a point-in-time, model-free audit of all configured factor candidates."""

    spec = FactorAuditSpec.from_yaml(config_path)
    config = spec.resolved_config()
    tables = load_silver(config)
    quality = validate_smoke(config, tables)
    quality.raise_for_failures()
    _require_valid_financial_snapshot(config, tables)
    prepared = ResearchWorkflow(config).prepare(
        tables,
        research_end_override=spec.end_date,
    )
    audit = build_factor_audit_tables(
        raw_features=prepared.raw_features,
        processed_features=prepared.processed.features,
        labels=prepared.labels,
        feature_quality=prepared.processed.quality,
        diagnostics=prepared.diagnostics,
        market_panel=prepared.market_panel,
        open_dates=prepared.open_dates,
        factor_names=prepared.factor_names,
        directions=prepared.directions,
        families=combined_factor_families(),
        gates=spec.gates,
        label_horizon=config.features.label_horizon,
        linear_cost_bps=config.research.linear_cost_bps,
        stamp_duty_change_date=config.research.stamp_duty_change_date,
        stamp_duty_before=config.research.stamp_duty_before,
        stamp_duty_after=config.research.stamp_duty_after,
        adv_window=config.optimizer.adv_lookback,
        max_adv_participation=config.optimizer.max_adv_participation,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{spec.audit_id}-{spec.spec_hash[:8]}"
    run_root = config.paths.run_root / spec.output_subdirectory / run_id
    correlation = prepared.diagnostics.correlation.rename_axis("factor").reset_index()
    artifacts = {
        "factor_catalog": write_parquet_atomic(
            factor_catalog_frame(prepared.factor_names),
            run_root / "factor-catalog.parquet",
        ),
        "raw_features": write_parquet_atomic(
            prepared.raw_features,
            run_root / "raw-features.parquet",
        ),
        "processed_features": write_parquet_atomic(
            prepared.processed.features,
            run_root / "processed-features.parquet",
        ),
        "feature_quality": write_parquet_atomic(
            prepared.processed.quality,
            run_root / "feature-quality.parquet",
        ),
        "labels": write_parquet_atomic(
            prepared.labels,
            run_root / "labels.parquet",
        ),
        "factor_ic": write_parquet_atomic(
            prepared.diagnostics.ic_by_date,
            run_root / "factor-ic.parquet",
        ),
        "quintile_returns": write_parquet_atomic(
            prepared.diagnostics.quintile_returns,
            run_root / "quintile-returns.parquet",
        ),
        "factor_correlation": write_parquet_atomic(
            correlation,
            run_root / "factor-correlation.parquet",
        ),
        "rebalance_spreads": write_parquet_atomic(
            audit.rebalance_spreads,
            run_root / "rebalance-spreads.parquet",
        ),
        "yearly_audit": write_parquet_atomic(
            audit.yearly,
            run_root / "yearly-audit.parquet",
        ),
        "factor_summary": write_parquet_atomic(
            audit.summary,
            run_root / "factor-summary.parquet",
        ),
        "distribution_audit": write_parquet_atomic(
            audit.distribution,
            run_root / "distribution-audit.parquet",
        ),
        "industry_dependence": write_parquet_atomic(
            audit.industry_dependence,
            run_root / "industry-dependence.parquet",
        ),
        "industry_distribution": write_parquet_atomic(
            audit.industry_distribution,
            run_root / "industry-distribution.parquet",
        ),
    }
    eligible = audit.summary.loc[audit.summary["eligible"].astype(bool), "factor"]
    point_in_time_violations = int(
        audit.summary[
            [
                "lookahead_violations",
                "stale_value_violations",
                "future_report_period_violations",
            ]
        ].sum().sum()
    )
    summary = {
        "contract_version": FACTOR_AUDIT_CONTRACT_VERSION,
        "status": "success",
        "audit_id": spec.audit_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "dataset": config.paths.dataset,
        "start_date": str(prepared.raw_features["decision_date"].min()),
        "end_date": str(prepared.raw_features["decision_date"].max()),
        "decision_dates": int(prepared.raw_features["decision_date"].nunique()),
        "panel_rows": int(len(prepared.raw_features)),
        "factor_count": len(prepared.factor_names),
        "eligible_factor_count": int(len(eligible)),
        "eligible_factors": eligible.astype(str).tolist(),
        "point_in_time_violations": point_in_time_violations,
        "all_data_quality_checks_passed": not quality.critical_failures,
        "network_requests": 0,
        "gates": asdict(spec.gates),
    }
    summary_fingerprint = write_json_atomic(
        summary,
        run_root / "factor-audit-summary.json",
    )
    data_fingerprints = _loaded_silver_fingerprints(config, tables)
    manifest = {
        **summary,
        "created_at": utc_now(),
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash,
        "base_config_hash": sha256_file(spec.base_config_path),
        "source_tree_hash": _source_tree_hash(config.paths.root),
        "git": _git_state(config.paths.root),
        "data_fingerprints": data_fingerprints,
        "artifact_fingerprints": {
            **artifacts,
            "factor_audit_summary": summary_fingerprint,
        },
        "summary_preview": [
            {str(key): finite_or_none(value) for key, value in row.items()}
            for row in audit.summary.head(10).to_dict(orient="records")
        ],
    }
    write_json_atomic(manifest, run_root / "factor-audit-manifest.json")
    return summary


def _require_valid_financial_snapshot(
    config: AppConfig,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    required = {
        "financial_income",
        "financial_balancesheet",
        "financial_cashflow",
    }
    missing = sorted(required.difference(tables))
    if missing:
        raise ConfigurationError(
            f"Factor audit requires point-in-time financial tables: {missing}"
        )
    quality_path = config.paths.quality_root / "financial-data-quality.json"
    if not quality_path.is_file():
        raise ConfigurationError(
            f"Factor audit requires financial quality evidence: {quality_path}"
        )
    payload = json.loads(quality_path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != FINANCIAL_CONTRACT_VERSION:
        raise ConfigurationError(
            "Factor audit requires a financial snapshot materialized under "
            f"{FINANCIAL_CONTRACT_VERSION}; rerun scripts/download_financial.ps1"
        )
    if (
        payload.get("status") != "success"
        or not payload.get("validation", {}).get("passed", False)
    ):
        raise ConfigurationError(
            "Factor audit cannot use a financial snapshot that failed validation"
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
        risk_model = build_risk_model(
            config.risk,
            market_panel,
            open_dates,
            index_bars=tables["index_bars"],
        )
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
        benchmark_membership_intervals=tables.get("benchmark_membership_intervals"),
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
        "positions": write_parquet_atomic(
            result.positions,
            run_root / "positions.parquet",
        ),
        "constraint_audits": write_parquet_atomic(
            result.constraint_audits,
            run_root / "constraint-audits.parquet",
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


def plan_annual_study(config_path: str | Path) -> dict[str, Any]:
    """Resolve an annual candidate-by-fold plan without running research."""

    spec = AnnualStudySpec.from_yaml(config_path)
    base_config = AppConfig.from_yaml(spec.method_study.base_config_path)
    method_configs = _annual_method_configs(spec, base_config)
    feature_hash, cost_hash = _annual_shared_contracts(method_configs)
    calendar_path = base_config.paths.silver_root / "calendar.parquet"
    if not calendar_path.is_file():
        raise ConfigurationError(
            "Annual planning requires the frozen Silver calendar: "
            f"{calendar_path}"
        )
    calendar = pd.read_parquet(calendar_path)
    open_dates = _open_dates(calendar)
    folds = build_annual_folds(spec, base_config, open_dates)
    payload = annual_study_plan(config_path, open_dates=open_dates)
    payload.update(
        {
            "shared_feature_contract_hash": feature_hash,
            "cost_contract_hash": cost_hash,
            "resolved_tasks": [
                {
                    "trial_id": trial.trial_id,
                    "fold_year": fold.year,
                    "method_hash": _effective_config_hash(
                        method_configs[trial.trial_id]
                    ),
                    "config_hash": _effective_config_hash(
                        resolve_annual_fold_config(
                            method_configs[trial.trial_id],
                            annual_id=spec.annual_id,
                            trial_id=trial.trial_id,
                            fold=fold,
                            embargo_days=spec.embargo_days,
                        )
                    ),
                }
                for trial in spec.method_study.trials
                for fold in folds
            ],
        }
    )
    return payload


def run_annual_study(
    config_path: str | Path,
    *,
    force: bool = False,
    max_workers: int | None = None,
    years: Sequence[int] | None = None,
    trial_ids: Sequence[str] | None = None,
) -> AnnualStudyResult:
    """Run or resume annual folds while sharing one immutable feature layer."""

    spec = AnnualStudySpec.from_yaml(config_path)
    base_config = AppConfig.from_yaml(spec.method_study.base_config_path)
    annual_root = base_config.paths.root / "annual-studies" / spec.annual_id

    def annual_progress(event: Mapping[str, Any]) -> None:
        write_json_atomic(
            {
                "schema_version": 1,
                "annual_id": spec.annual_id,
                "updated_at": utc_now(),
                **dict(event),
            },
            annual_root / "progress.json",
        )

    def announce(stage: str, status: str, **details: Any) -> None:
        suffix = "".join(
            f" | {key}={value}" for key, value in sorted(details.items())
        )
        LOGGER.info(
            "annual=%s | stage=%s | status=%s%s",
            spec.annual_id,
            stage,
            status,
            suffix,
        )
        emit_progress(
            annual_progress,
            stage=stage,
            status=status,
            **details,
        )

    method_configs = _annual_method_configs(spec, base_config)
    feature_contract_hash, cost_contract_hash = _annual_shared_contracts(
        method_configs
    )
    announce("research_inputs", "running")
    download_summary, tables, quality, data_fingerprints = _research_inputs(
        base_config,
        force=force,
    )
    announce(
        "research_inputs",
        "completed",
        silver_tables=len(tables),
        network_requests=download_summary.network_requests,
    )
    open_dates = _open_dates(tables["calendar"])
    folds = build_annual_folds(spec, base_config, open_dates)
    source_hash = _research_source_hash(base_config.paths.root)
    data_snapshot_hash = _data_snapshot_hash(data_fingerprints)
    context = StudyContext(
        base_config_hash=sha256_file(base_config.config_path),
        source_hash=source_hash,
        data_snapshot_hash=data_snapshot_hash,
        data_fingerprints=data_fingerprints,
        git=_git_state(base_config.paths.root),
    )
    preparation_config = _annual_preparation_config(
        next(iter(method_configs.values())),
        spec.annual_id,
    )
    announce("shared_features", "running")
    shared = _load_or_prepare_annual_features(
        config=preparation_config,
        tables=tables,
        annual_root=annual_root,
        feature_contract_hash=feature_contract_hash,
        source_hash=source_hash,
        data_snapshot_hash=data_snapshot_hash,
        progress_callback=annual_progress,
    )
    announce(
        "shared_features",
        "completed",
        decision_dates=int(shared.processed.features["decision_date"].nunique()),
        rows=len(shared.processed.features),
    )
    requested_years = set(int(value) for value in years) if years else set(spec.years)
    fold_views: dict[int, PreparedResearchData] = {}
    first_trial_id = spec.method_study.trials[0].trial_id
    announce("fold_views", "running", total_years=len(requested_years))
    for fold in folds:
        if fold.year not in requested_years:
            continue
        fold_config = resolve_annual_fold_config(
            method_configs[first_trial_id],
            annual_id=spec.annual_id,
            trial_id=first_trial_id,
            fold=fold,
            embargo_days=spec.embargo_days,
        )
        fold_views[fold.year] = ResearchWorkflow(fold_config).fold_view(
            tables,
            shared,
            progress_callback=annual_progress,
        )
        announce(
            "fold_views",
            "running",
            completed_years=len(fold_views),
            current_year=fold.year,
            total_years=len(requested_years),
        )
    announce("fold_views", "completed", completed_years=len(fold_views))

    fold_plan_hash = sha256_text(
        canonical_json([fold.to_dict() for fold in folds])
    )

    def execute(
        trial: StudyTrial,
        fold: AnnualFold,
        attempt_root: Path,
    ) -> AnnualFoldExecution:
        def task_progress(event: Mapping[str, Any]) -> None:
            write_json_atomic(
                {
                    "schema_version": 1,
                    "annual_id": spec.annual_id,
                    "trial_id": trial.trial_id,
                    "fold_year": fold.year,
                    "updated_at": utc_now(),
                    **dict(event),
                },
                attempt_root / "progress.json",
            )

        method_config = method_configs[trial.trial_id]
        fold_config = resolve_annual_fold_config(
            method_config,
            annual_id=spec.annual_id,
            trial_id=trial.trial_id,
            fold=fold,
            embargo_days=spec.embargo_days,
        )
        protocol = {
            "research_spec_hash": sha256_text(
                canonical_json(_research_spec(fold_config))
            ),
            "research_source_hash": source_hash,
            "data_snapshot_hash": data_snapshot_hash,
            "data_fingerprints": dict(sorted(data_fingerprints.items())),
            "study_id": spec.method_study.study_id,
            "trial_id": trial.trial_id,
            "annual_id": spec.annual_id,
            "fold_year": fold.year,
            "fold_plan_hash": fold_plan_hash,
            "shared_feature_contract_hash": feature_contract_hash,
        }
        config_hash = _effective_config_hash(fold_config)
        run_id = _research_run_id(config_hash)
        emit_progress(
            task_progress,
            stage="annual_fold",
            status="running",
        )
        try:
            run_id, summary = _materialize_research_run(
                fold_config,
                run_id=run_id,
                run_root=attempt_root,
                protocol=protocol,
                download_summary=download_summary,
                tables=tables,
                quality=quality,
                data_fingerprints=data_fingerprints,
                prepared=fold_views[fold.year],
                progress_callback=task_progress,
            )
            artifact_fingerprints = _annual_fold_artifact_fingerprints(
                attempt_root
            )
        except Exception as exc:
            emit_progress(
                task_progress,
                stage="annual_fold",
                status="failed",
                error_type=type(exc).__name__,
            )
            raise
        emit_progress(
            task_progress,
            stage="annual_fold",
            status="completed",
        )
        return AnnualFoldExecution(
            run_id=run_id,
            config_hash=config_hash,
            method_hash=_effective_config_hash(method_config),
            cost_hash=cost_contract_hash,
            summary=summary,
            artifact_fingerprints=artifact_fingerprints,
        )

    def aggregate(
        trial: StudyTrial,
        task_manifests: Sequence[Mapping[str, Any]],
        aggregate_root: Path,
    ) -> AnnualTrialAggregate:
        return _aggregate_annual_trial(
            trial=trial,
            task_manifests=task_manifests,
            annual_root=annual_root,
            aggregate_root=aggregate_root,
            calibrator_name=method_configs[trial.trial_id].workflow.calibrator.name,
            calibrator_params=method_configs[
                trial.trial_id
            ].workflow.calibrator.params,
        )

    announce("annual_tasks", "running")
    try:
        result = AnnualStudyRunner(
            spec,
            folds=folds,
            annual_root=annual_root,
            context=context,
            executor=execute,
            aggregator=aggregate,
            max_workers=max_workers,
            selected_years=years,
            selected_trials=trial_ids,
        ).run()
    except Exception as exc:
        announce("annual_tasks", "failed", error_type=type(exc).__name__)
        raise
    announce(
        "annual_tasks",
        result.status,
        completed_tasks=result.completed_task_count,
        pending_tasks=result.pending_task_count,
    )
    return result


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
    source_daily = pd.read_parquet(source.attempt_root / "daily.parquet")
    source_trades = pd.read_parquet(source.attempt_root / "trades.parquet")
    source_targets = pd.read_parquet(source.attempt_root / "targets.parquet")
    source_optimization = pd.read_parquet(
        source.attempt_root / "optimization.parquet"
    )
    source_positions = pd.read_parquet(source.attempt_root / "positions.parquet")
    source_constraint_audits = pd.read_parquet(
        source.attempt_root / "constraint-audits.parquet"
    )
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
    styles = _style_exposures(features)
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
        if scenario.execution_mode == "frozen_trades":
            daily, trades = replay_frozen_trade_costs(
                source_daily,
                source_trades,
                cost_multiplier=scenario.cost_multiplier,
            )
            targets = source_targets.copy()
            optimization = source_optimization.copy()
            positions = source_positions.copy()
            constraint_audits = source_constraint_audits.copy()
            metrics = calculate_backtest_metrics(daily, trades)
            metrics.update(summarize_constraint_audits(constraint_audits))
        else:
            risk_model = (
                build_risk_model(
                    scenario_config.risk,
                    market_panel,
                    open_dates,
                    index_bars=tables["index_bars"],
                    industry_exposures=exposures,
                    style_exposures=styles,
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
                benchmark_membership_intervals=tables.get(
                    "benchmark_membership_intervals"
                ),
                index_bars=tables["index_bars"],
                market_panel=market_panel,
                signals=portfolio_signals,
                portfolio_exposures=exposures,
                portfolio_styles=styles,
                portfolio_restrictions=restrictions,
                suspensions=tables.get("suspensions"),
                start_date=signal_dates[0],
                end_date=policy.evaluation_end,
            )
            daily = backtest.daily
            trades = backtest.trades
            targets = backtest.targets
            optimization = backtest.optimization
            positions = backtest.positions
            constraint_audits = backtest.constraint_audits
            metrics = backtest.metrics
        portfolio_evaluation = evaluate_portfolio_run(
            daily=daily,
            trades=trades,
            constraint_audits=constraint_audits,
        )
        solved = (
            int(optimization["status"].isin(["optimal", "optimal_inaccurate"]).sum())
            if not optimization.empty
            else 0
        )
        metric_difference = _maximum_metric_difference(
            source_metrics,
            metrics,
        )
        parity = (
            metric_difference <= spec.baseline_parity_tolerance
            if scenario.scenario_id == "baseline"
            else None
        )
        artifact_fingerprints = {
            "daily": write_parquet_atomic(daily, attempt_root / "daily.parquet"),
            "trades": write_parquet_atomic(
                trades,
                attempt_root / "trades.parquet",
            ),
            "targets": write_parquet_atomic(
                targets,
                attempt_root / "targets.parquet",
            ),
            "optimization": write_parquet_atomic(
                optimization,
                attempt_root / "optimization.parquet",
            ),
            "positions": write_parquet_atomic(
                positions,
                attempt_root / "positions.parquet",
            ),
            "constraint_audits": write_parquet_atomic(
                constraint_audits,
                attempt_root / "constraint-audits.parquet",
            ),
            "yearly_metrics": write_parquet_atomic(
                portfolio_evaluation.yearly_metrics,
                attempt_root / "yearly-metrics.parquet",
            ),
            "metrics": write_json_atomic(
                metrics,
                attempt_root / "metrics.json",
            ),
        }
        summary = {
            "scenario": scenario.to_dict(),
            "execution_mode": scenario.execution_mode,
            "optimizer_reoptimized": scenario.execution_mode == "reoptimized",
            "constraint_audit_mode": (
                "recalculated"
                if scenario.execution_mode == "reoptimized"
                else "source_gross_holdings_reused"
            ),
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
            "metrics": metrics,
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
        attempt_root / "daily.parquet",
        attempt_root / "trades.parquet",
        attempt_root / "targets.parquet",
        attempt_root / "optimization.parquet",
        attempt_root / "positions.parquet",
        attempt_root / "constraint-audits.parquet",
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
    for name, filename in (
        ("daily", "daily.parquet"),
        ("trades", "trades.parquet"),
        ("targets", "targets.parquet"),
        ("optimization", "optimization.parquet"),
        ("positions", "positions.parquet"),
        ("constraint_audits", "constraint-audits.parquet"),
    ):
        _assert_artifact_fingerprint(
            attempt_root / filename,
            artifact_fingerprints.get(name),
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
    prepared: PreparedResearchData | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, dict[str, Any]]:
    workflow = ResearchWorkflow(config)
    result = (
        workflow.run_prepared(
            tables,
            prepared,
            progress_callback=progress_callback,
        )
        if prepared is not None
        else workflow.run(tables, progress_callback=progress_callback)
    )
    _assert_stage_artifact_boundaries(config, result)
    evaluation = evaluate_research_run(
        signals=result.evaluation_signals,
        labels=result.labels,
        daily=result.backtest.daily,
        trades=result.backtest.trades,
        constraint_audits=result.backtest.constraint_audits,
        optimization=result.backtest.optimization,
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
        "positions": write_parquet_atomic(
            result.backtest.positions,
            run_root / "positions.parquet",
        ),
        "constraint_audits": write_parquet_atomic(
            result.backtest.constraint_audits,
            run_root / "constraint-audits.parquet",
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
            "positions": _date_range(
                result.backtest.positions,
                "execution_date",
            ),
            "constraint_audits": _date_range(
                result.backtest.constraint_audits,
                "execution_date",
            ),
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
        **(
            {
                "annual_fold": {
                    "annual_id": protocol["annual_id"],
                    "fold_year": protocol["fold_year"],
                    "fold_plan_hash": protocol["fold_plan_hash"],
                    "shared_feature_contract_hash": protocol[
                        "shared_feature_contract_hash"
                    ],
                }
            }
            if "annual_id" in protocol
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


def _annual_method_configs(
    spec: AnnualStudySpec,
    base_config: AppConfig,
) -> dict[str, AppConfig]:
    if base_config.experiment.stage == "frozen_test":
        raise ConfigurationError("Annual studies cannot open the frozen-test stage")
    configs: dict[str, AppConfig] = {}
    for trial in spec.method_study.trials:
        resolved = resolve_trial_config(
            base_config,
            study_id=spec.method_study.study_id,
            trial=trial,
        )
        _validate_research_components(resolved)
        configs[trial.trial_id] = resolved
    if not configs:
        raise ConfigurationError("Annual study has no candidate methods")
    return configs


def _annual_shared_contracts(
    configs: Mapping[str, AppConfig],
) -> tuple[str, str]:
    feature_contracts = {
        sha256_text(canonical_json(annual_feature_contract(config)))
        for config in configs.values()
    }
    if len(feature_contracts) != 1:
        raise ConfigurationError(
            "Annual candidates must share one feature and label contract"
        )
    cost_contracts = {
        sha256_text(canonical_json(annual_cost_contract(config)))
        for config in configs.values()
    }
    if len(cost_contracts) != 1:
        raise ConfigurationError(
            "Annual candidates must share one execution-cost contract"
        )
    return next(iter(feature_contracts)), next(iter(cost_contracts))


def _annual_preparation_config(config: AppConfig, annual_id: str) -> AppConfig:
    prepared = replace(
        config,
        experiment=replace(
            config.experiment,
            stage="walk_forward",
            protocol_id=f"{config.experiment.protocol_id}-{annual_id}-shared",
            allow_frozen_test=False,
        ),
    )
    prepared.validate()
    return prepared


def _load_or_prepare_annual_features(
    *,
    config: AppConfig,
    tables: Mapping[str, pd.DataFrame],
    annual_root: Path,
    feature_contract_hash: str,
    source_hash: str,
    data_snapshot_hash: str,
    progress_callback: ProgressCallback | None = None,
) -> PreparedResearchData:
    shared_root = annual_root / "shared"
    manifest_path = shared_root / "shared-manifest.json"
    identity = {
        "feature_contract_hash": feature_contract_hash,
        "source_hash": source_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "research_end": config.dates.end,
    }
    if manifest_path.is_file():
        manifest = _read_json_object(manifest_path)
        changed = sorted(
            key for key, value in identity.items() if manifest.get(key) != value
        )
        if changed:
            raise ConfigurationError(
                "Existing shared annual feature layer has a different identity: "
                f"{changed}"
            )
        artifacts = manifest.get("artifact_fingerprints")
        if not isinstance(artifacts, Mapping):
            raise ConfigurationError("Shared annual manifest lacks artifact hashes")
        for reference, expected in artifacts.items():
            _assert_artifact_fingerprint(shared_root / str(reference), expected)
        raw_features = pd.read_parquet(shared_root / "raw-features.parquet")
        features = pd.read_parquet(shared_root / "features.parquet")
        feature_quality = pd.read_parquet(shared_root / "feature-quality.parquet")
        labels = pd.read_parquet(shared_root / "labels.parquet")
        research_panel = pd.read_parquet(shared_root / "research-panel.parquet")
        correlation = pd.read_parquet(
            shared_root / "factor-correlation.parquet"
        ).set_index("factor")
        open_dates = _open_dates(tables["calendar"])
        return PreparedResearchData(
            factor_names=tuple(str(value) for value in manifest["factor_names"]),
            directions={
                str(name): int(value)
                for name, value in manifest["directions"].items()
            },
            market_panel=build_market_panel(
                tables["stock_bars"],
                tables["adjustments"],
                tables["price_limits"],
            ),
            open_dates=open_dates,
            research_end=str(manifest["research_end"]),
            sample_policy=ResearchSamplePolicy(
                config.experiment,
                tuple(open_dates),
            ),
            raw_features=raw_features,
            processed=ProcessedFactors(
                features=features,
                quality=feature_quality,
            ),
            labels=labels,
            research_panel=research_panel,
            diagnostics=FactorDiagnostics(
                ic_by_date=pd.read_parquet(shared_root / "factor-ic.parquet"),
                summary=pd.read_parquet(shared_root / "factor-summary.parquet"),
                quintile_returns=pd.read_parquet(
                    shared_root / "quintile-returns.parquet"
                ),
                correlation=correlation,
            ),
        )

    prepared = ResearchWorkflow(config).prepare(
        tables,
        progress_callback=progress_callback,
    )
    correlation = prepared.diagnostics.correlation.rename_axis("factor").reset_index()
    artifact_fingerprints = {
        "raw-features.parquet": write_parquet_atomic(
            prepared.raw_features,
            shared_root / "raw-features.parquet",
        ),
        "features.parquet": write_parquet_atomic(
            prepared.processed.features,
            shared_root / "features.parquet",
        ),
        "feature-quality.parquet": write_parquet_atomic(
            prepared.processed.quality,
            shared_root / "feature-quality.parquet",
        ),
        "labels.parquet": write_parquet_atomic(
            prepared.labels,
            shared_root / "labels.parquet",
        ),
        "research-panel.parquet": write_parquet_atomic(
            prepared.research_panel,
            shared_root / "research-panel.parquet",
        ),
        "factor-ic.parquet": write_parquet_atomic(
            prepared.diagnostics.ic_by_date,
            shared_root / "factor-ic.parquet",
        ),
        "factor-summary.parquet": write_parquet_atomic(
            prepared.diagnostics.summary,
            shared_root / "factor-summary.parquet",
        ),
        "quintile-returns.parquet": write_parquet_atomic(
            prepared.diagnostics.quintile_returns,
            shared_root / "quintile-returns.parquet",
        ),
        "factor-correlation.parquet": write_parquet_atomic(
            correlation,
            shared_root / "factor-correlation.parquet",
        ),
    }
    write_json_atomic(
        {
            "schema_version": 1,
            **identity,
            "created_at": utc_now(),
            "factor_names": list(prepared.factor_names),
            "directions": dict(sorted(prepared.directions.items())),
            "artifact_fingerprints": artifact_fingerprints,
        },
        manifest_path,
    )
    return prepared


def _annual_fold_artifact_fingerprints(run_root: Path) -> dict[str, str]:
    required = (
        "run-manifest.json",
        "workflow-summary.json",
        "evaluation-signals.parquet",
        "model-fits.parquet",
        "calibration-fits.parquet",
        "daily.parquet",
        "trades.parquet",
        "targets.parquet",
        "optimization.parquet",
        "positions.parquet",
        "constraint-audits.parquet",
        "gold/labels.parquet",
    )
    missing = [reference for reference in required if not (run_root / reference).is_file()]
    if missing:
        raise ConfigurationError(f"Annual fold lacks required artifacts: {missing}")
    return {reference: sha256_file(run_root / reference) for reference in required}


def _aggregate_annual_trial(
    *,
    trial: StudyTrial,
    task_manifests: Sequence[Mapping[str, Any]],
    annual_root: Path,
    aggregate_root: Path,
    calibrator_name: str,
    calibrator_params: Mapping[str, Any],
) -> AnnualTrialAggregate:
    if not task_manifests:
        raise ConfigurationError(f"Annual trial {trial.trial_id} has no folds")
    ordered = sorted(
        task_manifests,
        key=lambda item: int(_annual_fold_mapping(item)["year"]),
    )
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    model_fit_frames: list[pd.DataFrame] = []
    calibration_fit_frames: list[pd.DataFrame] = []
    constraint_frames: list[pd.DataFrame] = []
    optimization_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    expected_years: list[int] = []
    quality_passed = True
    optimizer_attempts = 0
    optimizer_solved = 0
    valid_expected_returns = 0

    for task in ordered:
        fold = _annual_fold_mapping(task)
        year = int(fold["year"])
        expected_years.append(year)
        attempt_root = _annual_attempt_root(annual_root, task)
        _audit_annual_run_manifest(task, fold, attempt_root)
        daily = pd.read_parquet(attempt_root / "daily.parquet")
        trades = pd.read_parquet(attempt_root / "trades.parquet")
        signals = pd.read_parquet(attempt_root / "evaluation-signals.parquet")
        labels = pd.read_parquet(attempt_root / "gold" / "labels.parquet")
        model_fits = pd.read_parquet(attempt_root / "model-fits.parquet")
        calibration_fits = pd.read_parquet(
            attempt_root / "calibration-fits.parquet"
        )
        constraints = pd.read_parquet(attempt_root / "constraint-audits.parquet")
        optimization_path = attempt_root / "optimization.parquet"
        optimization = (
            pd.read_parquet(optimization_path)
            if optimization_path.is_file()
            else pd.DataFrame()
        )
        _audit_annual_fold_artifacts(
            trial_id=trial.trial_id,
            fold=fold,
            daily=daily,
            signals=signals,
            model_fits=model_fits,
            calibration_fits=calibration_fits,
        )
        evaluation_start = str(fold["evaluation_start"])
        evaluation_end = str(fold["evaluation_end"])
        labels = labels.loc[
            labels["decision_date"].astype(str).between(
                evaluation_start,
                evaluation_end,
            )
        ].copy()
        validation_model_fits = model_fits.loc[
            model_fits["experiment_phase"].astype(str).eq("validation")
        ].copy()
        validation_calibration_fits = calibration_fits.loc[
            calibration_fits["experiment_phase"].astype(str).eq("validation")
        ].copy()
        for frame in (
            daily,
            trades,
            signals,
            labels,
            validation_model_fits,
            validation_calibration_fits,
            constraints,
            optimization,
        ):
            frame["fold_year"] = year
        daily_frames.append(daily)
        trade_frames.append(trades)
        signal_frames.append(signals)
        label_frames.append(labels)
        model_fit_frames.append(validation_model_fits)
        calibration_fit_frames.append(validation_calibration_fits)
        constraint_frames.append(constraints)
        optimization_frames.append(optimization)
        summary = task.get("summary")
        if not isinstance(summary, Mapping):
            raise ConfigurationError(
                f"Annual fold {trial.trial_id}/{year} lacks a summary"
            )
        quality_passed &= bool(summary.get("quality_passed", False))
        optimizer_attempts += int(summary.get("optimizer_attempts", 0))
        optimizer_solved += int(summary.get("optimizer_solved", 0))
        valid_expected_returns += int(summary.get("valid_expected_returns", 0))
        fold_rows.append(
            {
                "trial_id": trial.trial_id,
                "fold_year": year,
                "run_id": task.get("run_id"),
                "train_end": fold["train_end"],
                "last_mature_label_date": fold["last_mature_label_date"],
                "embargo_start": fold["embargo_start"],
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "observations": len(daily),
                "valid_expected_returns": int(
                    summary.get("valid_expected_returns", 0)
                ),
                "optimizer_attempts": int(summary.get("optimizer_attempts", 0)),
                "optimizer_solved": int(summary.get("optimizer_solved", 0)),
                "quality_passed": bool(summary.get("quality_passed", False)),
            }
        )

    if expected_years != sorted(set(expected_years)):
        raise ConfigurationError(
            f"Annual trial {trial.trial_id} has duplicate or unordered fold years"
        )
    combined_daily = _stitch_annual_daily(pd.concat(daily_frames, ignore_index=True))
    combined_trades = pd.concat(trade_frames, ignore_index=True)
    combined_signals = pd.concat(signal_frames, ignore_index=True)
    combined_labels = pd.concat(label_frames, ignore_index=True)
    combined_model_fits = pd.concat(model_fit_frames, ignore_index=True)
    combined_calibration_fits = pd.concat(
        calibration_fit_frames,
        ignore_index=True,
    )
    combined_constraints = pd.concat(constraint_frames, ignore_index=True)
    combined_optimization = pd.concat(optimization_frames, ignore_index=True)
    _require_unique_keys(combined_daily, ["trade_date"], "annual daily")
    _require_unique_keys(
        combined_signals,
        ["decision_date", "instrument"],
        "annual evaluation signals",
    )
    _require_unique_keys(
        combined_labels,
        ["decision_date", "instrument"],
        "annual evaluation labels",
    )
    metrics = calculate_backtest_metrics(combined_daily, combined_trades)
    evaluation = evaluate_research_run(
        signals=combined_signals,
        labels=combined_labels,
        daily=combined_daily,
        trades=combined_trades,
        constraint_audits=combined_constraints,
        optimization=combined_optimization,
        model_fits=combined_model_fits,
        calibrator_name=calibrator_name,
        calibrator_params=calibrator_params,
    )
    aggregate_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "fold-summary.parquet": write_parquet_atomic(
            pd.DataFrame(fold_rows),
            aggregate_root / "fold-summary.parquet",
        ),
        "daily.parquet": write_parquet_atomic(
            combined_daily,
            aggregate_root / "daily.parquet",
        ),
        "trades.parquet": write_parquet_atomic(
            combined_trades,
            aggregate_root / "trades.parquet",
        ),
        "optimization.parquet": write_parquet_atomic(
            combined_optimization,
            aggregate_root / "optimization.parquet",
        ),
        "evaluation-signals.parquet": write_parquet_atomic(
            combined_signals,
            aggregate_root / "evaluation-signals.parquet",
        ),
        "evaluation-labels.parquet": write_parquet_atomic(
            combined_labels,
            aggregate_root / "evaluation-labels.parquet",
        ),
        "model-fits.parquet": write_parquet_atomic(
            combined_model_fits,
            aggregate_root / "model-fits.parquet",
        ),
        "calibration-fits.parquet": write_parquet_atomic(
            combined_calibration_fits,
            aggregate_root / "calibration-fits.parquet",
        ),
        "constraint-audits.parquet": write_parquet_atomic(
            combined_constraints,
            aggregate_root / "constraint-audits.parquet",
        ),
        "yearly-metrics.parquet": write_parquet_atomic(
            evaluation.yearly_metrics,
            aggregate_root / "yearly-metrics.parquet",
        ),
        "factor-weight-history.parquet": write_parquet_atomic(
            evaluation.factor_weight_history,
            aggregate_root / "factor-weight-history.parquet",
        ),
        "metrics.json": write_json_atomic(
            metrics,
            aggregate_root / "metrics.json",
        ),
        "research-evaluation.json": write_json_atomic(
            evaluation.summary,
            aggregate_root / "research-evaluation.json",
        ),
    }
    summary = {
        "trial_id": trial.trial_id,
        "fold_years": expected_years,
        "fold_count": len(expected_years),
        "quality_passed": quality_passed,
        "valid_expected_returns": valid_expected_returns,
        "optimizer_attempts": optimizer_attempts,
        "optimizer_solved": optimizer_solved,
        "optimizer_solve_rate": (
            optimizer_solved / optimizer_attempts
            if optimizer_attempts
            else 1.0
        ),
        "metrics": metrics,
        "evaluation": evaluation.summary,
    }
    artifacts["aggregate-summary.json"] = write_json_atomic(
        summary,
        aggregate_root / "aggregate-summary.json",
    )
    return AnnualTrialAggregate(
        summary=summary,
        artifact_fingerprints=artifacts,
    )


def _annual_fold_mapping(task: Mapping[str, Any]) -> dict[str, Any]:
    value = task.get("fold")
    if not isinstance(value, dict):
        raise ConfigurationError("Annual task manifest lacks a fold definition")
    required = {
        "year",
        "train_end",
        "embargo_start",
        "last_mature_label_date",
        "evaluation_start",
        "evaluation_end",
        "first_decision_date",
        "last_decision_date",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ConfigurationError(f"Annual fold definition is incomplete: {missing}")
    return dict(value)


def _annual_attempt_root(
    annual_root: Path,
    task: Mapping[str, Any],
) -> Path:
    reference = task.get("artifact_root")
    if not isinstance(reference, str) or not reference:
        raise ConfigurationError("Annual task manifest lacks artifact_root")
    root = (annual_root / reference).resolve()
    if not root.is_relative_to(annual_root.resolve()):
        raise ConfigurationError("Annual task artifact_root escapes its parent")
    return root


def _audit_annual_run_manifest(
    task: Mapping[str, Any],
    fold: Mapping[str, Any],
    attempt_root: Path,
) -> None:
    manifest = _read_json_object(attempt_root / "run-manifest.json")
    summary = _read_json_object(attempt_root / "workflow-summary.json")
    if manifest.get("run_id") != task.get("run_id"):
        raise ConfigurationError("Annual fold run_id differs from its task manifest")
    if manifest.get("config_hash") != task.get("resolved_config_hash"):
        raise ConfigurationError(
            "Annual fold configuration hash differs from its task manifest"
        )
    if canonical_json(summary) != canonical_json(task.get("summary")):
        raise ConfigurationError("Annual fold summary differs from its task manifest")
    annual = manifest.get("annual_fold")
    experiment = manifest.get("experiment")
    if not isinstance(annual, Mapping) or not isinstance(experiment, Mapping):
        raise ConfigurationError("Annual run manifest lacks fold protocol metadata")
    expected_annual = {
        "annual_id": task.get("annual_id"),
        "fold_year": int(fold["year"]),
    }
    changed_annual = sorted(
        key for key, value in expected_annual.items() if annual.get(key) != value
    )
    expected_experiment = {
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "validation_start": fold["evaluation_start"],
        "validation_end": fold["evaluation_end"],
        "data_snapshot_hash": task.get("data_snapshot_hash"),
    }
    changed_experiment = sorted(
        key
        for key, value in expected_experiment.items()
        if experiment.get(key) != value
    )
    if changed_annual or changed_experiment:
        raise ConfigurationError(
            "Annual run manifest differs from its frozen fold definition: "
            f"annual={changed_annual}, experiment={changed_experiment}"
        )


def _audit_annual_fold_artifacts(
    *,
    trial_id: str,
    fold: Mapping[str, Any],
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    model_fits: pd.DataFrame,
    calibration_fits: pd.DataFrame,
) -> None:
    year = int(fold["year"])
    evaluation_start = str(fold["evaluation_start"])
    evaluation_end = str(fold["evaluation_end"])
    if daily.empty or "trade_date" not in daily:
        raise ConfigurationError(f"Annual fold {trial_id}/{year} has no daily rows")
    daily_dates = daily["trade_date"].astype(str)
    if not daily_dates.between(evaluation_start, evaluation_end).all():
        raise ConfigurationError(
            f"Annual fold {trial_id}/{year} daily data cross fold boundaries"
        )
    if daily_dates.duplicated().any():
        raise ConfigurationError(
            f"Annual fold {trial_id}/{year} has duplicate daily dates"
        )
    if signals.empty or "decision_date" not in signals:
        raise ConfigurationError(
            f"Annual fold {trial_id}/{year} has no evaluation signals"
        )
    signal_dates = signals["decision_date"].astype(str)
    if not signal_dates.between(evaluation_start, evaluation_end).all():
        raise ConfigurationError(
            f"Annual fold {trial_id}/{year} signals cross fold boundaries"
        )
    _audit_fold_fits(
        frame=model_fits,
        label=f"model fits {trial_id}/{year}",
        fold=fold,
    )
    _audit_fold_fits(
        frame=calibration_fits,
        label=f"calibration fits {trial_id}/{year}",
        fold=fold,
    )


def _audit_fold_fits(
    *,
    frame: pd.DataFrame,
    label: str,
    fold: Mapping[str, Any],
) -> None:
    required = {
        "fit_date",
        "experiment_phase",
        "max_label_available_date",
        "max_training_decision_date",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConfigurationError(f"{label} lack columns: {missing}")
    if frame.empty:
        raise ConfigurationError(f"{label} are empty")
    phases = frame["experiment_phase"].astype(str)
    if not phases.isin(["train", "validation"]).all():
        raise ConfigurationError(f"{label} contain an unexpected experiment phase")
    validation = frame.loc[phases.eq("validation")]
    validation_dates = validation["fit_date"].astype(str)
    first_decision_date = str(fold["first_decision_date"])
    repeated_closed_model_checks = (
        len(validation) > 1
        and "status" in validation
        and validation["status"].astype(str).eq("no_selected_factors").all()
        and validation_dates.is_unique
    )
    frozen_fit = len(validation) == 1
    if (
        validation.empty
        or validation_dates.min() != first_decision_date
        or not (frozen_fit or repeated_closed_model_checks)
    ):
        raise ConfigurationError(
            f"{label} were not frozen at the first fold decision"
        )
    training = frame.loc[phases.eq("train")]
    if not training.empty and training["fit_date"].astype(str).gt(
        str(fold["train_end"])
    ).any():
        raise ConfigurationError(f"{label} contain a training fit after train_end")
    available = frame["max_label_available_date"].dropna().astype(str)
    if not available.empty and available.gt(
        str(fold["last_mature_label_date"])
    ).any():
        raise ConfigurationError(f"{label} use labels inside the embargo")
    decisions = frame["max_training_decision_date"].dropna().astype(str)
    if not decisions.empty and decisions.gt(str(fold["train_end"])).any():
        raise ConfigurationError(f"{label} use decisions after train_end")


def _stitch_annual_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "portfolio_return", "benchmark_return"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ConfigurationError(f"Annual daily rows lack columns: {missing}")
    result = daily.sort_values("trade_date").reset_index(drop=True).copy()
    if result["trade_date"].astype(str).duplicated().any():
        raise ConfigurationError("Annual daily fold dates overlap")
    portfolio_return = pd.to_numeric(
        result["portfolio_return"], errors="raise"
    ).astype(float)
    benchmark_return = pd.to_numeric(
        result["benchmark_return"], errors="raise"
    ).astype(float)
    if not np.isfinite(portfolio_return).all() or not np.isfinite(
        benchmark_return
    ).all():
        raise ConfigurationError("Annual daily returns must be finite")
    result["nav"] = (1.0 + portfolio_return).cumprod()
    result["benchmark_nav"] = (1.0 + benchmark_return).cumprod()
    return enrich_active_performance(result)


def _require_unique_keys(
    frame: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise ConfigurationError(f"{label} lack key columns: {missing}")
    if frame.duplicated(list(keys)).any():
        raise ConfigurationError(f"{label} natural key is not unique: {list(keys)}")


def _open_dates(calendar: pd.DataFrame) -> list[str]:
    required = {"trade_date", "is_open"}
    missing = sorted(required.difference(calendar.columns))
    if missing:
        raise ConfigurationError(f"Silver calendar lacks columns: {missing}")
    dates = sorted(
        calendar.loc[calendar["is_open"].eq(1), "trade_date"]
        .astype(str)
        .unique()
    )
    if not dates:
        raise ConfigurationError("Silver calendar has no open dates")
    return dates


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
        "total_return_index_code": config.source.total_return_index_code,
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
        "positions": (result.backtest.positions, "execution_date"),
        "constraint_audits": (
            result.backtest.constraint_audits,
            "execution_date",
        ),
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

    fingerprints: dict[str, str] = {}
    for name in sorted(tables):
        path = (
            config.paths.silver_root
            / "financial"
            / f"{name.removeprefix('financial_')}.parquet"
            if name.startswith("financial_")
            else config.paths.silver_root / f"{name}.parquet"
        )
        fingerprints[name] = sha256_file(path)
    return fingerprints


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
