from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from csi500_alpha.data.storage import write_json_atomic
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.utils import sha256_file

_BLUE = "#244A64"
_BLUE_OPEN = "#DCE7ED"
_GOLD = "#B98945"
_INK = "#202A33"
_MUTED = "#626C75"
_GRID = "#D9DEE3"
_WHITE = "#FBFBFA"

_FAMILY_LABELS = {
    "accrual": "应计",
    "growth": "成长",
    "interaction": "交互",
    "investment": "投资",
    "liquidity": "流动性",
    "momentum": "动量",
    "quality": "质量",
    "reversal": "反转",
    "risk": "风险",
    "size": "规模",
    "value": "估值",
}


@dataclass(frozen=True)
class V2ReadmeReportResult:
    output_root: Path
    summary_path: Path
    manifest_path: Path
    figure_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "summary": str(self.summary_path),
            "manifest": str(self.manifest_path),
            "figures": [str(path) for path in self.figure_paths],
        }


@dataclass(frozen=True)
class _AnnualEvidence:
    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    daily: pd.DataFrame
    yearly: pd.DataFrame


@dataclass(frozen=True)
class _FactorEvidence:
    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    factors: pd.DataFrame
    yearly: pd.DataFrame
    correlation: pd.DataFrame


@dataclass(frozen=True)
class _SelectionEvidence:
    root: Path
    annual_manifest: dict[str, Any]
    progress: dict[str, Any]
    selection: dict[str, Any]
    publication_gates: dict[str, Any]
    released: _AnnualEvidence
    challenger: _AnnualEvidence


def build_v2_readme_report(
    *,
    baseline_root: str | Path,
    expanded_root: str | Path,
    selection_root: str | Path,
    factor_audit_root: str | Path,
    output_root: str | Path,
) -> V2ReadmeReportResult:
    """Build repository-facing v2 figures from fingerprinted aggregate evidence."""

    comparison_baseline = _load_annual_evidence(
        Path(baseline_root).resolve(),
        "Factor-expansion baseline",
    )
    expanded = _load_annual_evidence(Path(expanded_root).resolve(), "Expanded pool")
    factor_audit = _load_factor_evidence(Path(factor_audit_root).resolve())
    _validate_comparable_annual_evidence(comparison_baseline, expanded)
    selection = _load_selection_evidence(
        Path(selection_root).resolve(),
        comparison_baseline=comparison_baseline,
    )
    released = selection.released

    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary = _public_summary(
        released,
        comparison_baseline,
        expanded,
        factor_audit,
        selection,
    )
    summary_path = destination / "v2-public-summary.json"
    write_json_atomic(summary, summary_path)

    pyplot, mdates = _matplotlib()
    figures = (
        destination / "v2-backtest-overview.png",
        destination / "v2-factor-audit.png",
        destination / "v2-freeze-rationale.png",
    )
    _plot_backtest(pyplot, mdates, released, figures[0])
    _plot_factor_audit(pyplot, factor_audit, figures[1])
    _plot_freeze_rationale(pyplot, comparison_baseline, expanded, figures[2])

    source_artifacts = {
        "comparison-baseline/aggregate-manifest.json": (
            comparison_baseline.root / "aggregate-manifest.json"
        ),
        "comparison-baseline/aggregate-summary.json": (
            comparison_baseline.root / "aggregate-summary.json"
        ),
        "comparison-baseline/daily.parquet": comparison_baseline.root / "daily.parquet",
        "comparison-baseline/yearly-metrics.parquet": (
            comparison_baseline.root / "yearly-metrics.parquet"
        ),
        "expanded/aggregate-manifest.json": expanded.root / "aggregate-manifest.json",
        "expanded/aggregate-summary.json": expanded.root / "aggregate-summary.json",
        "expanded/yearly-metrics.parquet": expanded.root / "yearly-metrics.parquet",
        "selection/annual-study-manifest.json": (
            selection.root / "annual-study-manifest.json"
        ),
        "selection/progress.json": selection.root / "progress.json",
        "selection/selection.json": selection.root / "selection.json",
        "selection/publication-gates.json": selection.root / "publication-gates.json",
        "selection/released/aggregate-manifest.json": (
            released.root / "aggregate-manifest.json"
        ),
        "selection/released/aggregate-summary.json": (
            released.root / "aggregate-summary.json"
        ),
        "selection/released/daily.parquet": released.root / "daily.parquet",
        "selection/released/yearly-metrics.parquet": (
            released.root / "yearly-metrics.parquet"
        ),
        "selection/challenger/aggregate-manifest.json": (
            selection.challenger.root / "aggregate-manifest.json"
        ),
        "selection/challenger/model-fits.parquet": (
            selection.challenger.root / "model-fits.parquet"
        ),
        "factor-audit/factor-audit-manifest.json": (
            factor_audit.root / "factor-audit-manifest.json"
        ),
        "factor-audit/factor-audit-summary.json": (factor_audit.root / "factor-audit-summary.json"),
        "factor-audit/factor-summary.parquet": factor_audit.root / "factor-summary.parquet",
        "factor-audit/factor-correlation.parquet": (
            factor_audit.root / "factor-correlation.parquet"
        ),
        "factor-audit/yearly-audit.parquet": factor_audit.root / "yearly-audit.parquet",
    }
    chart_map = {
        figures[0].name: {
            "question": "冻结方案在 2020—2025 年滚动年度研究折中的净值与主动回撤如何？",
            "takeaway": (
                "六个研究年度折累计主动收益为 9.76%，五个年度为正；该区间不是新的独立最终留出样本。"
            ),
            "type": "three-panel line and drawdown chart",
            "fields": ["trade_date", "nav", "benchmark_nav", "active_nav"],
        },
        figures[1].name: {
            "question": "42 个候选因子的正式审计如何收缩因子池，证据是否随时间衰减？",
            "takeaway": (
                "14 个因子通过跨年覆盖率、IC 与净多空价差门槛；2025 年仅 7 个保持正净价差。"
            ),
            "type": "family count bars and yearly evidence bars",
            "fields": [
                "family",
                "eligible",
                "year",
                "mean_q5_minus_q1_net",
            ],
        },
        figures[2].name: {
            "question": "备选因子扩展是否带来稳定的年度增量？",
            "takeaway": (
                "扩池提高全期汇总收益，但年度增量中位数为负，正主动收益年份由五个降至四个。"
            ),
            "type": "paired annual bar charts",
            "fields": ["year", "annualized_active_return", "increment_vs_baseline"],
        },
    }
    outputs = {path.name: sha256_file(path) for path in (*figures, summary_path)}
    manifest = {
        "schema_version": 2,
        "public_boundary": (
            "Aggregate metrics and raster figures only. Holdings, trades, signals, "
            "vendor rows, credentials and local paths are excluded."
        ),
        "source_ids": {
            "release_annual_id": selection.annual_manifest.get("annual_id"),
            "released_trial_id": released.summary.get("trial_id"),
            "challenger_trial_id": selection.challenger.summary.get("trial_id"),
            "comparison_annual_id": comparison_baseline.manifest.get("annual_id"),
            "comparison_baseline_trial_id": comparison_baseline.summary.get("trial_id"),
            "expanded_trial_id": expanded.summary.get("trial_id"),
            "factor_audit_id": factor_audit.summary.get("audit_id"),
            "factor_audit_run_id": factor_audit.summary.get("run_id"),
        },
        "source_artifacts": {
            name: sha256_file(path) for name, path in sorted(source_artifacts.items())
        },
        "chart_map": chart_map,
        "outputs": dict(sorted(outputs.items())),
    }
    manifest_path = destination / "v2-report-manifest.json"
    write_json_atomic(manifest, manifest_path)
    return V2ReadmeReportResult(
        output_root=destination,
        summary_path=summary_path,
        manifest_path=manifest_path,
        figure_paths=figures,
    )


def _load_annual_evidence(root: Path, label: str) -> _AnnualEvidence:
    manifest_path = root / "aggregate-manifest.json"
    summary_path = root / "aggregate-summary.json"
    daily_path = root / "daily.parquet"
    yearly_path = root / "yearly-metrics.parquet"
    manifest = _read_object(manifest_path, f"{label} manifest")
    if manifest.get("status") != "completed" or manifest.get("error") is not None:
        raise ConfigurationError(f"{label} aggregate is not complete")
    fingerprints = _as_mapping(
        manifest.get("artifact_fingerprints"),
        f"{label} artifact fingerprints",
    )
    for name, path in (
        ("aggregate-summary.json", summary_path),
        ("daily.parquet", daily_path),
        ("yearly-metrics.parquet", yearly_path),
    ):
        _verify_fingerprint(path, fingerprints.get(name), f"{label} {name}")
    summary = _read_object(summary_path, f"{label} summary")
    if manifest.get("summary") != summary:
        raise ConfigurationError(f"{label} manifest summary does not match its artifact")
    if summary.get("quality_passed") is not True:
        raise ConfigurationError(f"{label} did not pass aggregate quality checks")
    daily = pd.read_parquet(daily_path)
    yearly = pd.read_parquet(yearly_path)
    _validate_daily(daily, label)
    _validate_yearly(yearly, label)
    if int(summary.get("fold_count", -1)) != len(yearly):
        raise ConfigurationError(f"{label} fold count does not match yearly metrics")
    return _AnnualEvidence(
        root=root,
        manifest=manifest,
        summary=summary,
        daily=daily,
        yearly=yearly,
    )


def _load_factor_evidence(root: Path) -> _FactorEvidence:
    manifest_path = root / "factor-audit-manifest.json"
    summary_path = root / "factor-audit-summary.json"
    factor_path = root / "factor-summary.parquet"
    correlation_path = root / "factor-correlation.parquet"
    yearly_path = root / "yearly-audit.parquet"
    manifest = _read_object(manifest_path, "Factor-audit manifest")
    summary = _read_object(summary_path, "Factor-audit summary")
    if manifest.get("status") != "success" or summary.get("status") != "success":
        raise ConfigurationError("Factor audit is not successful")
    if (
        manifest.get("all_data_quality_checks_passed") is not True
        or summary.get("all_data_quality_checks_passed") is not True
    ):
        raise ConfigurationError("Factor audit did not pass all data-quality checks")
    if int(summary.get("point_in_time_violations", -1)) != 0:
        raise ConfigurationError("Factor audit contains point-in-time violations")
    fingerprints = _as_mapping(
        manifest.get("artifact_fingerprints"),
        "Factor-audit artifact fingerprints",
    )
    for key, path in (
        ("factor_audit_summary", summary_path),
        ("factor_correlation", correlation_path),
        ("factor_summary", factor_path),
        ("yearly_audit", yearly_path),
    ):
        _verify_fingerprint(path, fingerprints.get(key), f"Factor-audit {key}")
    factors = pd.read_parquet(factor_path)
    correlation = pd.read_parquet(correlation_path)
    yearly = pd.read_parquet(yearly_path)
    required_factor = {
        "factor",
        "family",
        "eligible",
        "mean_coverage",
        "mean_directed_rank_ic",
        "median_yearly_q5_minus_q1_net",
    }
    required_yearly = {"year", "factor", "mean_q5_minus_q1_net"}
    _require_columns(factors, required_factor, "Factor summary")
    _require_columns(yearly, required_yearly, "Yearly factor audit")
    if factors["factor"].astype(str).duplicated().any():
        raise ConfigurationError("Factor summary contains duplicate factors")
    eligible_count = int(factors["eligible"].astype(bool).sum())
    if eligible_count != int(summary.get("eligible_factor_count", -1)):
        raise ConfigurationError("Factor-audit eligible count does not match its summary")
    if len(factors) != int(summary.get("factor_count", -1)):
        raise ConfigurationError("Factor-audit candidate count does not match its summary")
    _require_columns(correlation, {"factor"}, "Factor correlation")
    factor_names = set(factors["factor"].astype(str))
    if set(correlation["factor"].astype(str)) != factor_names:
        raise ConfigurationError("Factor-correlation rows do not match the factor summary")
    if not factor_names.issubset(correlation.columns):
        raise ConfigurationError("Factor-correlation columns do not match the factor summary")
    return _FactorEvidence(
        root=root,
        manifest=manifest,
        summary=summary,
        factors=factors,
        yearly=yearly,
        correlation=correlation,
    )


def _load_selection_evidence(
    root: Path,
    *,
    comparison_baseline: _AnnualEvidence,
) -> _SelectionEvidence:
    manifest = _read_object(root / "annual-study-manifest.json", "Release annual manifest")
    progress = _read_object(root / "progress.json", "Release annual progress")
    selection = _read_object(root / "selection.json", "Release annual selection")
    publication_gates = _read_object(
        root / "publication-gates.json",
        "Release publication gates",
    )
    expected_status = "completed_without_publishable_candidate"
    if manifest.get("status") != expected_status or progress.get("status") != expected_status:
        raise ConfigurationError(
            "V2 release evidence must be complete and non-publishable by its gates"
        )
    task_count = int(manifest.get("task_count", -1))
    if (
        task_count <= 0
        or int(manifest.get("completed_task_count", -1)) != task_count
        or int(manifest.get("failed_task_count", -1)) != 0
        or int(manifest.get("pending_task_count", -1)) != 0
        or int(progress.get("completed_tasks", -1)) != task_count
        or int(progress.get("pending_tasks", -1)) != 0
    ):
        raise ConfigurationError("V2 release annual tasks are incomplete")

    selected_id = selection.get("selected_trial_id")
    if not isinstance(selected_id, str) or not selected_id:
        raise ConfigurationError("V2 release selection lacks a selected trial")
    if manifest.get("selected_trial_id") != selected_id:
        raise ConfigurationError("V2 release manifest and selection disagree")
    registry = _as_mapping(manifest.get("registry"), "Release experiment registry")
    current = registry.get("current_candidates")
    if not isinstance(current, list):
        raise ConfigurationError("Release experiment registry lacks current candidates")
    included_ids = [
        str(item.get("id"))
        for item in current
        if isinstance(item, Mapping) and item.get("included_in_selection") is True
    ]
    challengers = [trial_id for trial_id in included_ids if trial_id != selected_id]
    if len(included_ids) != 2 or len(challengers) != 1:
        raise ConfigurationError(
            "V2 release reporting requires one selected method and one challenger"
        )

    released = _load_annual_evidence(root / "aggregates" / selected_id, "Released method")
    challenger = _load_annual_evidence(
        root / "aggregates" / challengers[0],
        "Release challenger",
    )
    _validate_comparable_annual_evidence(released, challenger)
    _validate_equivalent_release_baseline(comparison_baseline, released)
    if not _portfolio_paths_equal(released, challenger):
        raise ConfigurationError("Release challenger changed the economic portfolio path")

    challenger_fits = challenger.root / "model-fits.parquet"
    fingerprints = _as_mapping(
        challenger.manifest.get("artifact_fingerprints"),
        "Release challenger artifact fingerprints",
    )
    _verify_fingerprint(
        challenger_fits,
        fingerprints.get("model-fits.parquet"),
        "Release challenger model fits",
    )
    gate_trials = publication_gates.get("trials")
    if not isinstance(gate_trials, list):
        raise ConfigurationError("Release publication-gate results are missing")
    selected_gate = next(
        (
            item
            for item in gate_trials
            if isinstance(item, Mapping) and item.get("trial_id") == selected_id
        ),
        None,
    )
    if not isinstance(selected_gate, Mapping) or selected_gate.get("passed") is not False:
        raise ConfigurationError("Released method must retain its failed publication-gate result")
    return _SelectionEvidence(
        root=root,
        annual_manifest=manifest,
        progress=progress,
        selection=selection,
        publication_gates=publication_gates,
        released=released,
        challenger=challenger,
    )


def _validate_equivalent_release_baseline(
    comparison_baseline: _AnnualEvidence,
    released: _AnnualEvidence,
) -> None:
    baseline_snapshot = _annual_data_snapshot(comparison_baseline)
    released_snapshot = _annual_data_snapshot(released)
    if not baseline_snapshot or baseline_snapshot != released_snapshot:
        raise ConfigurationError("Released method does not use the comparison data snapshot")
    _validate_evaluation_alignment(comparison_baseline, released)
    if not _portfolio_paths_equal(comparison_baseline, released):
        raise ConfigurationError(
            "Released method does not reproduce the frozen comparison baseline"
        )


def _annual_data_snapshot(evidence: _AnnualEvidence) -> Any:
    snapshot = evidence.manifest.get("data_snapshot_hash")
    if snapshot:
        return snapshot
    annual_manifest_path = evidence.root.parents[1] / "annual-study-manifest.json"
    if not annual_manifest_path.is_file():
        return None
    annual_manifest = _read_object(annual_manifest_path, "Parent annual manifest")
    if annual_manifest.get("annual_id") != evidence.manifest.get("annual_id"):
        raise ConfigurationError("Aggregate and parent annual manifest disagree")
    return annual_manifest.get("data_snapshot_hash")


def _portfolio_paths_equal(left: _AnnualEvidence, right: _AnnualEvidence) -> bool:
    if left.summary.get("metrics") != right.summary.get("metrics"):
        return False
    if not left.daily.reset_index(drop=True).equals(right.daily.reset_index(drop=True)):
        return False
    return left.yearly.reset_index(drop=True).equals(right.yearly.reset_index(drop=True))


def _validate_comparable_annual_evidence(
    baseline: _AnnualEvidence,
    expanded: _AnnualEvidence,
) -> None:
    if baseline.manifest.get("annual_id") != expanded.manifest.get("annual_id"):
        raise ConfigurationError("Annual comparisons must come from the same annual study")
    _validate_evaluation_alignment(baseline, expanded)


def _validate_evaluation_alignment(
    baseline: _AnnualEvidence,
    comparison: _AnnualEvidence,
) -> None:
    baseline_dates = baseline.daily["trade_date"].astype(str).reset_index(drop=True)
    comparison_dates = comparison.daily["trade_date"].astype(str).reset_index(drop=True)
    if not baseline_dates.equals(comparison_dates):
        raise ConfigurationError("Annual comparisons do not share the same evaluation dates")
    baseline_years = baseline.yearly["year"].astype(int).tolist()
    comparison_years = comparison.yearly["year"].astype(int).tolist()
    if baseline_years != comparison_years:
        raise ConfigurationError("Annual comparisons do not share the same fold years")
    benchmark_gap = (
        pd.to_numeric(baseline.daily["benchmark_nav"], errors="coerce")
        - pd.to_numeric(comparison.daily["benchmark_nav"], errors="coerce")
    ).abs()
    if benchmark_gap.isna().any() or float(benchmark_gap.max()) > 1e-10:
        raise ConfigurationError("Annual comparisons do not share the same benchmark path")


def _public_summary(
    released: _AnnualEvidence,
    comparison_baseline: _AnnualEvidence,
    expanded: _AnnualEvidence,
    factor_audit: _FactorEvidence,
    selection: _SelectionEvidence,
) -> dict[str, Any]:
    released_metrics = _as_mapping(released.summary.get("metrics"), "Released metrics")
    expanded_metrics = _as_mapping(expanded.summary.get("metrics"), "Expanded metrics")
    released_evaluation = _as_mapping(
        released.summary.get("evaluation"),
        "Released evaluation",
    )
    expanded_evaluation = _as_mapping(
        expanded.summary.get("evaluation"),
        "Expanded evaluation",
    )
    released_yearly = _as_mapping(
        released_evaluation.get("yearly"),
        "Released yearly evaluation",
    )
    expanded_yearly = _as_mapping(
        expanded_evaluation.get("yearly"),
        "Expanded yearly evaluation",
    )
    released_execution = _as_mapping(
        released_evaluation.get("execution"),
        "Released execution evaluation",
    )
    expanded_execution = _as_mapping(
        expanded_evaluation.get("execution"),
        "Expanded execution evaluation",
    )
    baseline_rows = comparison_baseline.yearly[["year", "annualized_active_return"]].copy()
    expanded_rows = expanded.yearly[["year", "annualized_active_return"]].copy()
    annual = baseline_rows.merge(
        expanded_rows,
        on="year",
        validate="one_to_one",
        suffixes=("_baseline", "_expanded"),
    )
    annual["increment_vs_baseline"] = (
        annual["annualized_active_return_expanded"] - annual["annualized_active_return_baseline"]
    )

    factors = factor_audit.factors.copy()
    family = (
        factors.groupby("family", as_index=False)
        .agg(
            candidate_count=("factor", "size"),
            eligible_count=("eligible", "sum"),
            median_coverage=("mean_coverage", "median"),
        )
        .sort_values(
            ["eligible_count", "candidate_count", "family"],
            ascending=[False, False, True],
        )
    )
    eligible = factors.loc[factors["eligible"].astype(bool)].copy()
    eligible_components = _correlation_components(
        factor_audit.correlation,
        eligible["factor"].astype(str),
        threshold=0.75,
    )
    eligible_yearly = factor_audit.yearly.merge(
        eligible[["factor"]],
        on="factor",
        how="inner",
        validate="many_to_one",
    )
    yearly_factor = (
        eligible_yearly.groupby("year", as_index=False)
        .agg(
            factor_count=("factor", "size"),
            positive_net_spread_count=(
                "mean_q5_minus_q1_net",
                lambda values: int((values > 0).sum()),
            ),
            median_net_spread=("mean_q5_minus_q1_net", "median"),
        )
        .sort_values("year")
    )

    released_public = _annual_public_metrics(
        released_metrics,
        released_yearly,
        released_execution,
        released.summary,
    )
    expanded_public = _annual_public_metrics(
        expanded_metrics,
        expanded_yearly,
        expanded_execution,
        expanded.summary,
    )
    return {
        "schema_version": 2,
        "release": {
            "version": "2.0.0",
            "status": "research_release",
            "publication_gate_assessment": _publication_gate_summary(selection),
            "candidate_signal_check": _candidate_signal_summary(selection),
        },
        "scope": {
            "evaluation_role": "rolling_annual_research",
            "start_date": str(released_metrics.get("start_date")),
            "end_date": str(released_metrics.get("end_date")),
            "fold_years": [int(value) for value in released.yearly["year"]],
            "benchmark": "CSI 500 total-return proxy",
            "costs_included": True,
        },
        "released_method": released_public,
        "expanded_pool_ablation": {
            **expanded_public,
            "delta_annualized_active_return": (
                expanded_public["annualized_active_return"]
                - released_public["annualized_active_return"]
            ),
            "delta_information_ratio": (
                expanded_public["information_ratio"] - released_public["information_ratio"]
            ),
            "median_annual_increment": _finite(
                annual["increment_vs_baseline"].median(),
                "Median annual increment",
            ),
            "promoted": False,
        },
        "annual_comparison": [
            {
                "year": int(row.year),
                "baseline_annualized_active_return": _finite(
                    row.annualized_active_return_baseline,
                    "Baseline annualized active return",
                ),
                "expanded_annualized_active_return": _finite(
                    row.annualized_active_return_expanded,
                    "Expanded annualized active return",
                ),
                "increment_vs_baseline": _finite(
                    row.increment_vs_baseline,
                    "Annual increment",
                ),
            }
            for row in annual.itertuples(index=False)
        ],
        "factor_audit": {
            "start_date": str(factor_audit.summary.get("start_date")),
            "end_date": str(factor_audit.summary.get("end_date")),
            "decision_dates": int(factor_audit.summary.get("decision_dates", 0)),
            "panel_rows": int(factor_audit.summary.get("panel_rows", 0)),
            "candidate_count": int(factor_audit.summary.get("factor_count", 0)),
            "eligible_count": int(factor_audit.summary.get("eligible_factor_count", 0)),
            "mean_coverage": _finite(factors["mean_coverage"].mean(), "Mean coverage"),
            "point_in_time_violations": int(
                factor_audit.summary.get("point_in_time_violations", -1)
            ),
            "correlation_threshold": 0.75,
            "eligible_correlation_component_count": len(eligible_components),
            "family_rows": [
                {
                    "family": str(row.family),
                    "candidate_count": int(row.candidate_count),
                    "eligible_count": int(row.eligible_count),
                    "median_coverage": _finite(row.median_coverage, "Family coverage"),
                }
                for row in family.itertuples(index=False)
            ],
            "eligible_factors": [
                {
                    "factor": str(row.factor),
                    "family": str(row.family),
                    "mean_coverage": _finite(row.mean_coverage, "Factor coverage"),
                    "mean_directed_rank_ic": _finite(
                        row.mean_directed_rank_ic,
                        "Directed Rank IC",
                    ),
                    "median_yearly_net_spread": _finite(
                        row.median_yearly_q5_minus_q1_net,
                        "Median yearly net spread",
                    ),
                }
                for row in eligible.sort_values(["family", "factor"]).itertuples(index=False)
            ],
            "yearly_rows": [
                {
                    "year": int(row.year),
                    "factor_count": int(row.factor_count),
                    "positive_net_spread_count": int(row.positive_net_spread_count),
                    "median_net_spread": _finite(
                        row.median_net_spread,
                        "Yearly median net spread",
                    ),
                }
                for row in yearly_factor.itertuples(index=False)
            ],
        },
        "evidence_boundary": [
            (
                "The 2020-2025 results are rolling annual research folds, "
                "not a new independent holdout."
            ),
            (
                "The factor audit is aggregate diagnostic evidence and does not "
                "replace fold-internal selection."
            ),
            (
                "The 2026 H1 holdout is a revealed diagnostic for the prior method, "
                "not an independent holdout for v2."
            ),
            "Costs and capacity are modeled; the results are not live trading evidence.",
        ],
    }


def _publication_gate_summary(selection: _SelectionEvidence) -> dict[str, Any]:
    rules = selection.publication_gates.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ConfigurationError("Release publication-gate rules are missing")
    rows: list[dict[str, Any]] = []
    for item in rules:
        rule = _as_mapping(item, "Publication-gate rule")
        path = str(rule.get("path", ""))
        operator = str(rule.get("operator", ""))
        threshold = rule.get("value")
        observed = _resolve_metric_path(selection.released.summary, path)
        passed = _evaluate_gate(observed, operator, threshold)
        rows.append(
            {
                "path": path,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
            }
        )
    failed = [row for row in rows if not row["passed"]]
    return {
        "passed": not failed,
        "rule_count": len(rows),
        "passed_rule_count": len(rows) - len(failed),
        "failed_rule_count": len(failed),
        "failed_rules": failed,
    }


def _candidate_signal_summary(selection: _SelectionEvidence) -> dict[str, Any]:
    path = selection.challenger.root / "model-fits.parquet"
    fits = pd.read_parquet(path)
    _require_columns(
        fits,
        {"fold_year", "fit_date", "status", "model_parameters"},
        "Release challenger model fits",
    )
    fold_rows: list[dict[str, Any]] = []
    observed_shares: list[float] = []
    for year, group in fits.groupby(pd.to_numeric(fits["fold_year"], errors="raise")):
        evidence_rows: list[dict[str, Any]] = []
        for row in group.sort_values("fit_date").itertuples(index=False):
            parameters = _parse_json_mapping(row.model_parameters)
            if "candidate_share" not in parameters:
                continue
            sleeve = parameters.get("sleeve_evidence")
            sleeve_mapping = dict(sleeve) if isinstance(sleeve, Mapping) else {}
            share = _finite(parameters.get("candidate_share"), "Candidate share")
            observed_shares.append(share)
            oof_t = sleeve_mapping.get("oof_t")
            evidence_rows.append(
                {
                    "fit_date": str(row.fit_date),
                    "actual_candidate_share": share,
                    "oof_t": None if oof_t is None else _finite(oof_t, "Candidate OOF t"),
                    "oof_evidence_passed": sleeve_mapping.get("oof_evidence_passed") is True,
                }
            )
        if evidence_rows:
            final = evidence_rows[-1]
            fold_rows.append(
                {
                    "year": int(year),
                    "status": "evaluated",
                    "fit_count": len(evidence_rows),
                    "maximum_candidate_share": max(
                        float(item["actual_candidate_share"]) for item in evidence_rows
                    ),
                    "final_oof_t": final["oof_t"],
                    "oof_evidence_passed": any(
                        bool(item["oof_evidence_passed"]) for item in evidence_rows
                    ),
                }
            )
        else:
            no_factor_count = int((group["status"].astype(str) == "no_selected_factors").sum())
            if no_factor_count != len(group):
                raise ConfigurationError("Release challenger has unexplained model-fit rows")
            fold_rows.append(
                {
                    "year": int(year),
                    "status": "no_selected_factors",
                    "fit_count": len(group),
                    "maximum_candidate_share": 0.0,
                    "final_oof_t": None,
                    "oof_evidence_passed": False,
                }
            )
    maximum_share = max(observed_shares, default=0.0)
    return {
        "portfolio_path_identical": True,
        "maximum_actual_candidate_share": maximum_share,
        "active_allocation_years": sum(
            float(row["maximum_candidate_share"]) > 0.0 for row in fold_rows
        ),
        "year_count": len(fold_rows),
        "promoted": False,
        "fold_rows": fold_rows,
    }


def _resolve_metric_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigurationError(f"Publication-gate metric is missing: {path}")
        current = current[part]
    if isinstance(current, bool):
        return current
    return _finite(current, f"Publication-gate metric {path}")


def _evaluate_gate(observed: Any, operator: str, threshold: Any) -> bool:
    if operator == "==":
        return observed == threshold
    observed_number = _finite(observed, "Publication-gate observation")
    threshold_number = _finite(threshold, "Publication-gate threshold")
    if operator == ">=":
        return observed_number >= threshold_number
    if operator == "<=":
        return observed_number <= threshold_number
    raise ConfigurationError(f"Unsupported publication-gate operator: {operator}")


def _parse_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("Release challenger model parameters are invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ConfigurationError("Release challenger model parameters must be a mapping")
    return dict(parsed)


def _annual_public_metrics(
    metrics: Mapping[str, Any],
    yearly: Mapping[str, Any],
    execution: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trial_id": str(summary.get("trial_id")),
        "annualized_active_return": _finite(
            metrics.get("annualized_active_return"),
            "Annualized active return",
        ),
        "relative_active_total_return": _finite(
            metrics.get("relative_active_total_return"),
            "Relative active total return",
        ),
        "information_ratio": _finite(metrics.get("information_ratio"), "Information ratio"),
        "tracking_error": _finite(metrics.get("tracking_error"), "Tracking error"),
        "active_max_drawdown": _finite(
            metrics.get("active_max_drawdown"),
            "Active max drawdown",
        ),
        "capm_alpha_annualized": _finite(
            metrics.get("capm_alpha_annualized"),
            "CAPM alpha",
        ),
        "capm_beta": _finite(metrics.get("capm_beta"), "CAPM beta"),
        "average_turnover": _finite(metrics.get("average_turnover"), "Average turnover"),
        "transaction_cost": _finite(metrics.get("transaction_cost"), "Transaction cost"),
        "execution_cost_bps": _finite(
            execution.get("cost_bps_of_executed_notional"),
            "Execution cost bps",
        ),
        "notional_fill_ratio": _finite(
            execution.get("notional_fill_ratio"),
            "Notional fill ratio",
        ),
        "positive_active_years": int(yearly.get("positive_active_years", 0)),
        "year_count": int(yearly.get("year_count", 0)),
        "minimum_active_total_return": _finite(
            yearly.get("minimum_active_total_return"),
            "Minimum active total return",
        ),
        "optimizer_solve_rate": _finite(
            summary.get("optimizer_solve_rate"),
            "Optimizer solve rate",
        ),
    }


def _plot_backtest(
    pyplot: Any,
    mdates: Any,
    evidence: _AnnualEvidence,
    path: Path,
) -> None:
    daily = evidence.daily.copy()
    dates = pd.to_datetime(daily["trade_date"].astype(str), format="%Y%m%d")
    nav = pd.to_numeric(daily["nav"], errors="raise").astype(float)
    benchmark = pd.to_numeric(daily["benchmark_nav"], errors="raise").astype(float)
    active_nav = pd.to_numeric(daily["active_nav"], errors="raise").astype(float)
    active_return = active_nav - 1.0
    active_drawdown = active_nav / active_nav.cummax() - 1.0
    metrics = _as_mapping(evidence.summary.get("metrics"), "Baseline metrics")

    fig, axes = pyplot.subplots(
        3,
        1,
        figsize=(12, 7.7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    axes[0].plot(dates, nav, color=_BLUE, linewidth=2.0, label="组合")
    axes[0].plot(
        dates,
        benchmark,
        color=_INK,
        linewidth=1.35,
        linestyle="--",
        label="中证 500",
    )
    axes[0].set_ylabel("累计净值")
    axes[0].legend(loc="upper left", frameon=False, ncols=2)
    axes[0].grid(axis="y", color=_GRID, linewidth=0.8)

    axes[1].plot(dates, active_return, color=_BLUE, linewidth=1.45)
    axes[1].fill_between(
        dates,
        active_return,
        0.0,
        where=active_return.ge(0.0),
        color=_BLUE_OPEN,
        edgecolor="none",
    )
    axes[1].fill_between(
        dates,
        active_return,
        0.0,
        where=active_return.lt(0.0),
        color="#E5E8EA",
        edgecolor="none",
    )
    axes[1].axhline(0.0, color=_GRID, linewidth=0.9)
    axes[1].set_ylabel("累计主动收益")
    axes[1].yaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].grid(axis="y", color=_GRID, linewidth=0.8)

    axes[2].fill_between(dates, active_drawdown, 0.0, color=_BLUE_OPEN, edgecolor="none")
    axes[2].plot(dates, active_drawdown, color=_BLUE, linewidth=1.15)
    axes[2].set_ylabel("主动回撤")
    axes[2].yaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[2].grid(axis="y", color=_GRID, linewidth=0.8)
    locator = mdates.YearLocator()
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_GRID)
        axis.set_axisbelow(True)

    fig.suptitle(
        "图 1  冻结方案年度研究折净值与主动回撤",
        x=0.075,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.910,
        (
            f"{_human_date(metrics.get('start_date'))}—{_human_date(metrics.get('end_date'))}｜"
            "6 个滚动年度折｜已扣除模型化交易成本"
        ),
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.075,
        0.025,
        (
            "注：每个年度折在评价开始前冻结训练样本，日收益按时间顺序拼接；"
            "本区间属于研究样本，不是新的独立最终留出期。资料来源：Tushare Pro，本项目计算。"
        ),
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.12, hspace=0.12)
    _save_figure(fig, path, pyplot)


def _plot_factor_audit(pyplot: Any, evidence: _FactorEvidence, path: Path) -> None:
    factors = evidence.factors.copy()
    family = (
        factors.groupby("family", as_index=False)
        .agg(candidate_count=("factor", "size"), eligible_count=("eligible", "sum"))
        .sort_values(["eligible_count", "candidate_count", "family"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    family["label"] = family["family"].map(_FAMILY_LABELS).fillna(family["family"])
    eligible = factors.loc[factors["eligible"].astype(bool), ["factor"]]
    yearly = evidence.yearly.merge(
        eligible,
        on="factor",
        how="inner",
        validate="many_to_one",
    )
    yearly = (
        yearly.groupby("year", as_index=False)
        .agg(
            median_net_spread=("mean_q5_minus_q1_net", "median"),
            positive_count=("mean_q5_minus_q1_net", lambda values: int((values > 0).sum())),
            factor_count=("factor", "size"),
        )
        .sort_values("year")
    )
    yearly["median_bps"] = yearly["median_net_spread"] * 10_000

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    positions = list(range(len(family)))
    axes[0].barh(
        positions,
        family["candidate_count"],
        color=_BLUE_OPEN,
        height=0.62,
        label="候选",
    )
    axes[0].barh(
        positions,
        family["eligible_count"],
        color=_BLUE,
        height=0.62,
        label="通过审计",
    )
    for position, candidate, eligible_count in zip(
        positions,
        family["candidate_count"],
        family["eligible_count"],
        strict=True,
    ):
        axes[0].text(
            float(candidate) + 0.12,
            position,
            f"{int(eligible_count)}/{int(candidate)}",
            va="center",
            color=_INK,
            fontsize=8.5,
        )
    axes[0].set_yticks(positions, family["label"])
    axes[0].set_xlabel("因子数量")
    axes[0].set_title("因子家族覆盖")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_xlim(0, float(family["candidate_count"].max()) + 1.3)
    axes[0].grid(axis="x", color=_GRID, linewidth=0.8)

    colors = [_BLUE if value >= 0 else _MUTED for value in yearly["median_bps"]]
    year_positions = list(range(len(yearly)))
    bars = axes[1].bar(year_positions, yearly["median_bps"], color=colors, width=0.64)
    axes[1].axhline(0.0, color=_INK, linewidth=0.9)
    axes[1].set_title("入选因子的年度净多空价差中位数")
    axes[1].set_ylabel("基点 / 调仓")
    axes[1].set_xticks(year_positions, yearly["year"].astype(str))
    axes[1].grid(axis="y", color=_GRID, linewidth=0.8)
    for bar, row in zip(bars, yearly.itertuples(index=False), strict=True):
        value = float(row.median_bps)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35 if value >= 0 else value - 0.35,
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            color=_INK,
            fontsize=8.3,
        )
    last = yearly.iloc[-1]
    axes[1].annotate(
        f"{int(last['positive_count'])}/{int(last['factor_count'])} 个因子为正",
        xy=(len(yearly) - 1, float(last["median_bps"])),
        xytext=(-62, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": _MUTED, "linewidth": 0.9},
        color=_MUTED,
        fontsize=8.5,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_GRID)
        axis.set_axisbelow(True)

    fig.suptitle(
        "图 2  因子池审计与年度衰减",
        x=0.07,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.900,
        "2017—2025 年｜42 个候选、14 个通过审计｜覆盖率、时点一致性、IC 与扣费价差联合门槛",
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.025,
        (
            "注：右图为 14 个全样本审计入选因子的年度截面中位数；"
            "该汇总用于诊断，不替代每个年度折内的重新筛选。资料来源：Tushare Pro，本项目计算。"
        ),
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.15, wspace=0.28)
    _save_figure(fig, path, pyplot)


def _plot_freeze_rationale(
    pyplot: Any,
    baseline: _AnnualEvidence,
    expanded: _AnnualEvidence,
    path: Path,
) -> None:
    annual = baseline.yearly[["year", "annualized_active_return"]].merge(
        expanded.yearly[["year", "annualized_active_return"]],
        on="year",
        validate="one_to_one",
        suffixes=("_baseline", "_expanded"),
    )
    annual["increment"] = (
        annual["annualized_active_return_expanded"] - annual["annualized_active_return_baseline"]
    )
    annual["baseline_pct"] = annual["annualized_active_return_baseline"] * 100
    annual["increment_pp"] = annual["increment"] * 100
    positions = list(range(len(annual)))

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5.7))
    baseline_colors = [_BLUE if value >= 0 else _MUTED for value in annual["baseline_pct"]]
    bars = axes[0].bar(positions, annual["baseline_pct"], color=baseline_colors, width=0.64)
    axes[0].axhline(0.0, color=_INK, linewidth=0.9)
    axes[0].set_xticks(positions, annual["year"].astype(str))
    axes[0].set_ylabel("年化主动收益（%）")
    axes[0].set_title("冻结方案的年度表现")
    axes[0].grid(axis="y", color=_GRID, linewidth=0.8)
    _label_bars(axes[0], bars, annual["baseline_pct"])

    increment_colors = [_GOLD if value >= 0 else _MUTED for value in annual["increment_pp"]]
    delta_bars = axes[1].bar(positions, annual["increment_pp"], color=increment_colors, width=0.64)
    axes[1].axhline(0.0, color=_INK, linewidth=0.9)
    axes[1].set_xticks(positions, annual["year"].astype(str))
    axes[1].set_ylabel("相对冻结方案增量（百分点）")
    axes[1].set_title("扩展因子池的年度增量")
    axes[1].grid(axis="y", color=_GRID, linewidth=0.8)
    _label_bars(axes[1], delta_bars, annual["increment_pp"])
    median_increment = float(annual["increment_pp"].median())
    axes[1].text(
        0.98,
        0.04,
        f"年度增量中位数 {median_increment:+.2f} 个百分点",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=_MUTED,
        fontsize=8.8,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_GRID)
        axis.set_axisbelow(True)

    fig.suptitle(
        "图 3  备选因子扩展的年度稳定性检验",
        x=0.07,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.895,
        "2020—2025 年｜相同年度折、收益校准、风险约束与执行假设｜备选方案仅改变可选因子集合",
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.025,
        (
            "注：备选方案全期年化主动收益较冻结方案高 0.20 个百分点，"
            "但年度增量中位数为负，正主动收益年份由 5 个降至 4 个，因此未晋级。"
        ),
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.16, wspace=0.24)
    _save_figure(fig, path, pyplot)


def _label_bars(axis: Any, bars: Any, values: pd.Series) -> None:
    span = max(float(values.max() - values.min()), 1.0)
    offset = span * 0.025
    for bar, value in zip(bars, values, strict=True):
        numeric = float(value)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            numeric + offset if numeric >= 0 else numeric - offset,
            f"{numeric:+.2f}",
            ha="center",
            va="bottom" if numeric >= 0 else "top",
            color=_INK,
            fontsize=8.5,
        )


def _correlation_components(
    correlation: pd.DataFrame,
    selected: pd.Series,
    *,
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    names = tuple(selected)
    table = correlation.set_index("factor")
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            value = _finite(table.loc[left, right], "Factor correlation")
            if abs(value) >= threshold:
                union(left, right)
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(find(name), []).append(name)
    return tuple(
        sorted(
            (tuple(sorted(values)) for values in grouped.values()),
            key=lambda values: (-len(values), values),
        )
    )


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise ConfigurationError(
            'V2 README rendering requires the optional dependency: install ".[report]"'
        ) from exc
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "DejaVu Sans",
    )
    selected_font = next(
        (font for font in preferred_fonts if font in available_fonts),
        "DejaVu Sans",
    )
    pyplot.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font, "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "axes.titlecolor": _INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "figure.facecolor": _WHITE,
            "axes.facecolor": _WHITE,
            "savefig.facecolor": _WHITE,
            "axes.unicode_minus": False,
        }
    )
    return pyplot, mdates


def _save_figure(figure: Any, path: Path, pyplot: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        figure.savefig(
            temporary,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "csi500-alpha v2 README report"},
        )
        os.replace(temporary, path)
    finally:
        pyplot.close(figure)
        temporary.unlink(missing_ok=True)


def _validate_daily(frame: pd.DataFrame, label: str) -> None:
    required = {"trade_date", "nav", "benchmark_nav", "active_nav"}
    _require_columns(frame, required, f"{label} daily backtest")
    if len(frame) < 12:
        raise ConfigurationError(f"{label} daily backtest is too sparse")
    dates = frame["trade_date"].astype(str)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ConfigurationError(f"{label} daily dates must be unique and increasing")
    for column in ("nav", "benchmark_nav", "active_nav"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.map(math.isfinite)).any() or (values <= 0).any():
            raise ConfigurationError(f"{label} daily backtest has invalid {column}")


def _validate_yearly(frame: pd.DataFrame, label: str) -> None:
    required = {
        "year",
        "annualized_active_return",
        "active_total_return",
        "information_ratio",
    }
    _require_columns(frame, required, f"{label} yearly metrics")
    years = pd.to_numeric(frame["year"], errors="coerce")
    if years.isna().any() or years.duplicated().any() or not years.is_monotonic_increasing:
        raise ConfigurationError(f"{label} yearly metrics must have unique increasing years")
    for column in required.difference({"year"}):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.map(math.isfinite)).any():
            raise ConfigurationError(f"{label} yearly metrics has invalid {column}")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConfigurationError(f"{label} lacks columns: {missing}")


def _verify_fingerprint(path: Path, expected: Any, label: str) -> None:
    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path.name}")
    if not isinstance(expected, str) or not expected:
        raise ConfigurationError(f"{label} lacks a declared fingerprint")
    if sha256_file(path) != expected:
        raise ConfigurationError(f"{label} fingerprint does not match")


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


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    return dict(value)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{label} must be finite")
    return number


def _human_date(value: Any) -> str:
    try:
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Report date must be parseable: {value}") from exc
