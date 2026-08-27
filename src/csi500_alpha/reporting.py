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
from csi500_alpha.utils import canonical_json, sha256_file

_EVALUATION_ROLES = {
    "method_selection": "方法选择期（非最终留出期）",
    "extended_validation": "扩展验证期（非最终留出期）",
    "final_holdout": "最终留出期",
}

_ABLATION_LABELS = {
    "c0_ic_raw": "C0  原始 IC",
    "c1_ic_uncertainty": "C1  不确定性收缩",
    "c2_ic_correlation": "C2  相关性惩罚",
    "c3_ic_cost": "C3  换手惩罚",
    "c4_ic_full": "C4  权重稳定项",
}

_STRESS_LABELS = {
    "cost_0_5x": "成本 0.5 倍",
    "baseline": "基准情景",
    "cost_2_0x": "成本 2.0 倍",
    "aum_50m": "规模 0.5 亿元",
    "aum_300m": "规模 3 亿元",
    "participation_10pct": "ADV 上限 10%",
    "participation_20pct": "ADV 上限 20%",
}

_STRESS_ORDER = tuple(_STRESS_LABELS)

_BLUE = "#244A64"
_BLUE_OPEN = "#DCE7ED"
_RED = "#9B4A46"
_RED_OPEN = "#F1DFDC"
_INK = "#202A33"
_MUTED = "#626C75"
_GRID = "#D9DEE3"
_WHITE = "#FBFBFA"


@dataclass(frozen=True)
class PublicReportResult:
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
class _ReportEvidence:
    study_manifest: dict[str, Any]
    selected_trial: dict[str, Any]
    run_manifest: dict[str, Any]
    daily: pd.DataFrame
    ablation: pd.DataFrame
    stress: pd.DataFrame
    source_artifacts: Mapping[str, Path]


def build_public_report(
    *,
    study_root: str | Path,
    stress_root: str | Path,
    output_root: str | Path,
    ablation_study_root: str | Path | None = None,
    evaluation_role: str = "method_selection",
) -> PublicReportResult:
    """Export public-safe aggregate evidence without copying private row-level data."""

    if evaluation_role not in _EVALUATION_ROLES:
        raise ConfigurationError(
            f"evaluation_role must be one of {sorted(_EVALUATION_ROLES)}"
        )
    primary_root = Path(study_root).resolve()
    stress_path = Path(stress_root).resolve()
    ablation_root = (
        Path(ablation_study_root).resolve()
        if ablation_study_root is not None
        else primary_root
    )
    destination = Path(output_root).resolve()
    evidence = _load_evidence(
        primary_root,
        stress_path,
        ablation_root,
    )
    destination.mkdir(parents=True, exist_ok=True)

    summary = _public_summary(evidence, evaluation_role=evaluation_role)
    summary_path = destination / "public-summary.json"
    write_json_atomic(summary, summary_path)

    pyplot, mdates = _matplotlib()
    figures = (
        destination / "innovation-ablation.png",
        destination / "backtest-overview.png",
        destination / "stress-analysis.png",
    )
    _plot_innovation(pyplot, evidence.ablation, figures[0])
    _plot_backtest(
        pyplot,
        mdates,
        evidence.daily,
        evidence.selected_trial,
        evaluation_role,
        figures[1],
    )
    _plot_stress(pyplot, evidence.stress, figures[2])

    chart_map = {
        "innovation-ablation.png": {
            "title": "IC synthesis ablation",
            "scope": "C0-C4 method-selection ablation",
            "note": (
                "The stability term reduced factor-weight movement and also reduced "
                "information ratio in the same evaluation window."
            ),
            "type": "paired dot plots",
            "fields": ["trial_id", "information_ratio", "mean_factor_weight_l1_change"],
        },
        "backtest-overview.png": {
            "title": "Selected method NAV and drawdown",
            "scope": _EVALUATION_ROLES[evaluation_role],
            "note": "Portfolio and benchmark paths are shown net of modeled costs.",
            "type": "two-panel line and drawdown chart",
            "fields": ["trade_date", "nav", "benchmark_nav"],
        },
        "stress-analysis.png": {
            "title": "Cost and capacity stress",
            "scope": "selected signals fixed; risk, optimization and execution rerun",
            "note": (
                "Cost settings affect both optimizer turnover and simulated execution "
                "cost; information ratio is therefore not mechanically monotonic."
            ),
            "type": "paired dot plots with baseline references",
            "fields": [
                "scenario_id",
                "information_ratio",
                "cost_bps_of_executed_notional",
            ],
        },
    }
    outputs = {path.name: sha256_file(path) for path in (*figures, summary_path)}
    manifest = {
        "schema_version": 2,
        "public_boundary": (
            "Aggregate statistics and raster figures only. Signals, holdings, trades, "
            "index weights, vendor rows and local paths are excluded."
        ),
        "study_id": evidence.study_manifest["study_id"],
        "selected_trial_id": evidence.selected_trial["trial"]["id"],
        "evaluation_role": evaluation_role,
        "source_artifacts": {
            name: sha256_file(path)
            for name, path in sorted(evidence.source_artifacts.items())
        },
        "chart_map": chart_map,
        "outputs": dict(sorted(outputs.items())),
    }
    manifest_path = destination / "report-manifest.json"
    write_json_atomic(manifest, manifest_path)
    return PublicReportResult(
        output_root=destination,
        summary_path=summary_path,
        manifest_path=manifest_path,
        figure_paths=figures,
    )


def _load_evidence(
    study_root: Path,
    stress_root: Path,
    ablation_root: Path,
) -> _ReportEvidence:
    study_manifest_path = study_root / "study-manifest.json"
    selection_path = study_root / "selection.json"
    study_manifest = _read_object(study_manifest_path, "Study manifest")
    selection = _read_object(selection_path, "Study selection")
    if study_manifest.get("status") != "completed":
        raise ConfigurationError("Public report requires a completed Study")
    selected_id = selection.get("selected_trial_id")
    if not isinstance(selected_id, str) or not selected_id:
        raise ConfigurationError("Study selection does not identify a selected trial")
    if study_manifest.get("selected_trial_id") != selected_id:
        raise ConfigurationError("Study manifest and selection disagree on selected trial")

    selected_manifest_path = study_root / "trials" / selected_id / "trial-manifest.json"
    selected_trial = _read_object(selected_manifest_path, "Selected trial manifest")
    if selected_trial.get("status") != "completed":
        raise ConfigurationError("Selected trial is not completed")
    trial = _as_mapping(selected_trial.get("trial"), "Selected trial definition")
    if trial.get("id") != selected_id:
        raise ConfigurationError("Selected trial manifest has the wrong trial id")
    selected_root = _contained_path(
        study_root,
        selected_trial.get("artifact_root"),
        "selected trial artifact_root",
    )
    run_manifest_path = selected_root / "run-manifest.json"
    daily_path = selected_root / "daily.parquet"
    run_manifest = _read_object(run_manifest_path, "Selected run manifest")
    _assert_file(daily_path, "Selected daily backtest")
    fingerprints = _as_mapping(
        run_manifest.get("artifact_fingerprints"),
        "Selected run artifact_fingerprints",
    )
    expected_daily = fingerprints.get("daily")
    if not isinstance(expected_daily, str) or sha256_file(daily_path) != expected_daily:
        raise ConfigurationError("Selected daily backtest fingerprint does not match")
    selected_summary = _as_mapping(selected_trial.get("summary"), "Selected summary")
    run_summary = _as_mapping(run_manifest.get("summary"), "Selected run summary")
    if canonical_json(selected_summary) != canonical_json(run_summary):
        raise ConfigurationError("Selected trial summary differs from its run manifest")
    daily = pd.read_parquet(daily_path)
    _validate_daily(daily)

    ablation, ablation_sources = _load_ablation(ablation_root)
    stress, stress_sources = _load_stress(
        stress_root,
        study_id=str(study_manifest.get("study_id", "")),
        selected_trial_id=selected_id,
        run_manifest_path=run_manifest_path,
        data_snapshot_hash=str(study_manifest.get("data_snapshot_hash", "")),
    )
    source_artifacts = {
        "generator/reporting.py": Path(__file__).resolve(),
        "primary/study-manifest.json": study_manifest_path,
        "primary/selection.json": selection_path,
        "primary/selected-trial-manifest.json": selected_manifest_path,
        "primary/selected-run-manifest.json": run_manifest_path,
        "primary/selected-daily.parquet": daily_path,
        **ablation_sources,
        **stress_sources,
    }
    return _ReportEvidence(
        study_manifest=study_manifest,
        selected_trial=selected_trial,
        run_manifest=run_manifest,
        daily=daily,
        ablation=ablation,
        stress=stress,
        source_artifacts=source_artifacts,
    )


def _load_ablation(root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    manifest_path = root / "study-manifest.json"
    manifest = _read_object(manifest_path, "Ablation Study manifest")
    if manifest.get("status") != "completed":
        raise ConfigurationError("Ablation report requires a completed Study")
    rows: list[dict[str, Any]] = []
    sources = {"ablation/study-manifest.json": manifest_path}
    scopes: set[tuple[str, str]] = set()
    for trial_id, label in _ABLATION_LABELS.items():
        path = root / "trials" / trial_id / "trial-manifest.json"
        trial_manifest = _read_object(path, f"Ablation trial {trial_id}")
        if trial_manifest.get("status") != "completed":
            raise ConfigurationError(f"Ablation trial {trial_id} is not completed")
        summary = _as_mapping(trial_manifest.get("summary"), f"{trial_id} summary")
        metrics = _as_mapping(summary.get("metrics"), f"{trial_id} metrics")
        evaluation = _as_mapping(summary.get("evaluation"), f"{trial_id} evaluation")
        weights = _as_mapping(
            evaluation.get("model_weights"),
            f"{trial_id} model-weight evaluation",
        )
        ranges = _as_mapping(summary.get("artifact_date_ranges"), f"{trial_id} ranges")
        backtest_range = _as_mapping(ranges.get("backtest"), f"{trial_id} backtest range")
        scopes.add((str(backtest_range.get("start")), str(backtest_range.get("end"))))
        rows.append(
            {
                "trial_id": trial_id,
                "label": label,
                "information_ratio": _finite(metrics.get("information_ratio"), "IR"),
                "mean_factor_weight_l1_change": _finite(
                    weights.get("mean_factor_weight_l1_change"),
                    "mean factor-weight L1 change",
                ),
                "mean_effective_factor_count": _finite(
                    weights.get("mean_effective_factor_count"),
                    "mean effective factor count",
                ),
            }
        )
        sources[f"ablation/{trial_id}-manifest.json"] = path
    if len(scopes) != 1:
        raise ConfigurationError("C0-C4 ablation trials do not share one evaluation window")
    frame = pd.DataFrame(rows)
    frame.attrs["evaluation_start"], frame.attrs["evaluation_end"] = next(iter(scopes))
    return frame, sources


def _load_stress(
    root: Path,
    *,
    study_id: str,
    selected_trial_id: str,
    run_manifest_path: Path,
    data_snapshot_hash: str,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    manifest_path = root / "stress-manifest.json"
    summary_path = root / "stress-summary.json"
    manifest = _read_object(manifest_path, "Stress manifest")
    summary = _read_object(summary_path, "Stress summary")
    if manifest.get("status") != "completed":
        raise ConfigurationError("Public report requires a completed stress run")
    expected = {
        "source_study_id": study_id,
        "source_trial_id": selected_trial_id,
        "data_snapshot_hash": data_snapshot_hash,
    }
    changed = [key for key, value in expected.items() if manifest.get(key) != value]
    if changed:
        raise ConfigurationError(f"Stress run differs from selected Study: {changed}")
    if manifest.get("source_run_manifest_hash") != sha256_file(run_manifest_path):
        raise ConfigurationError("Stress run is not bound to the selected run manifest")
    if manifest.get("baseline_parity_passed") is not True:
        raise ConfigurationError("Stress baseline parity did not pass")
    if summary.get("baseline_parity_passed") is not True:
        raise ConfigurationError("Stress summary baseline parity did not pass")

    scenario_root = root / "scenarios"
    rows: list[dict[str, Any]] = []
    sources = {
        "stress/stress-manifest.json": manifest_path,
        "stress/stress-summary.json": summary_path,
    }
    for path in sorted(scenario_root.glob("*/scenario-manifest.json")):
        scenario_manifest = _read_object(path, f"Stress scenario {path.parent.name}")
        if scenario_manifest.get("status") != "completed":
            raise ConfigurationError(f"Stress scenario is not completed: {path.parent.name}")
        scenario = _as_mapping(scenario_manifest.get("scenario"), "Stress scenario")
        scenario_id = str(scenario.get("id", ""))
        scenario_summary = _as_mapping(
            scenario_manifest.get("summary"),
            f"Stress scenario {scenario_id} summary",
        )
        metrics = _as_mapping(scenario_summary.get("metrics"), f"{scenario_id} metrics")
        evaluation = _as_mapping(
            scenario_summary.get("evaluation"),
            f"{scenario_id} evaluation",
        )
        execution = _as_mapping(evaluation.get("execution"), f"{scenario_id} execution")
        yearly = _as_mapping(evaluation.get("yearly"), f"{scenario_id} yearly")
        rows.append(
            {
                "scenario_id": scenario_id,
                "label": _STRESS_LABELS.get(scenario_id, scenario_id),
                "information_ratio": _finite(metrics.get("information_ratio"), "IR"),
                "minimum_year_information_ratio": _finite(
                    yearly.get("minimum_information_ratio"),
                    "minimum-year IR",
                ),
                "annualized_active_return": _finite(
                    metrics.get("annualized_active_return"),
                    "annualized active return",
                ),
                "max_drawdown": _finite(metrics.get("max_drawdown"), "max drawdown"),
                "average_turnover": _finite(
                    metrics.get("average_turnover"),
                    "average turnover",
                ),
                "notional_fill_ratio": _finite(
                    execution.get("notional_fill_ratio"),
                    "notional fill ratio",
                ),
                "cost_bps_of_executed_notional": _finite(
                    execution.get("cost_bps_of_executed_notional"),
                    "execution cost bps",
                ),
                "optimizer_solve_rate": _finite(
                    scenario_summary.get("optimizer_solve_rate"),
                    "optimizer solve rate",
                ),
            }
        )
        sources[f"stress/{scenario_id}-manifest.json"] = path
    if not rows:
        raise ConfigurationError("Stress run has no completed scenario artifacts")
    frame = pd.DataFrame(rows)
    if "baseline" not in set(frame["scenario_id"]):
        raise ConfigurationError("Stress run does not contain a baseline scenario")
    expected_count = int(manifest.get("scenario_count", 0))
    if expected_count != len(frame):
        raise ConfigurationError(
            f"Stress scenario count differs: manifest={expected_count}, artifacts={len(frame)}"
        )
    order = {scenario_id: position for position, scenario_id in enumerate(_STRESS_ORDER)}
    frame["_order"] = frame["scenario_id"].map(order).fillna(len(order))
    frame = frame.sort_values(["_order", "scenario_id"]).drop(columns="_order")
    return frame.reset_index(drop=True), sources


def _public_summary(
    evidence: _ReportEvidence,
    *,
    evaluation_role: str,
) -> dict[str, Any]:
    summary = _as_mapping(evidence.selected_trial.get("summary"), "Selected summary")
    metrics = _as_mapping(summary.get("metrics"), "Selected metrics")
    evaluation = _as_mapping(summary.get("evaluation"), "Selected evaluation")
    calibration = _as_mapping(evaluation.get("calibration"), "Calibration evaluation")
    execution = _as_mapping(evaluation.get("execution"), "Execution evaluation")
    weights = _as_mapping(evaluation.get("model_weights"), "Model-weight evaluation")
    experiment = _as_mapping(evidence.run_manifest.get("experiment"), "Experiment")
    git = _as_mapping(evidence.study_manifest.get("git"), "Study git metadata")
    selected_definition = _as_mapping(
        evidence.selected_trial.get("trial"),
        "Selected trial definition",
    )
    ablation_rows = [
        {
            "trial_id": str(row.trial_id),
            "information_ratio": float(row.information_ratio),
            "mean_factor_weight_l1_change": float(row.mean_factor_weight_l1_change),
            "mean_effective_factor_count": float(row.mean_effective_factor_count),
        }
        for row in evidence.ablation.itertuples(index=False)
    ]
    stress_rows = [
        {
            "scenario_id": str(row.scenario_id),
            "information_ratio": float(row.information_ratio),
            "minimum_year_information_ratio": float(row.minimum_year_information_ratio),
            "annualized_active_return": float(row.annualized_active_return),
            "max_drawdown": float(row.max_drawdown),
            "average_turnover": float(row.average_turnover),
            "notional_fill_ratio": float(row.notional_fill_ratio),
            "cost_bps_of_executed_notional": float(
                row.cost_bps_of_executed_notional
            ),
            "optimizer_solve_rate": float(row.optimizer_solve_rate),
        }
        for row in evidence.stress.itertuples(index=False)
    ]
    return {
        "schema_version": 1,
        "data_source": "Tushare Pro",
        "study": {
            "id": evidence.study_manifest["study_id"],
            "selected_trial_id": selected_definition["id"],
            "completed_trials": int(evidence.study_manifest.get("completed_count", 0)),
            "declared_trials": int(evidence.study_manifest.get("trial_count", 0)),
            "source_commit": git.get("commit"),
        },
        "scope": {
            "training_start": experiment.get("train_start"),
            "training_end": experiment.get("train_end"),
            "evaluation_start": experiment.get("evaluation_start"),
            "evaluation_end": experiment.get("evaluation_end"),
            "evaluation_role": evaluation_role,
            "final_holdout": evaluation_role == "final_holdout",
        },
        "selected_method": {
            "selector": summary.get("selector"),
            "model": summary.get("model"),
            "calibrator": summary.get("calibrator"),
            "mean_effective_factor_count": _finite(
                weights.get("mean_effective_factor_count"),
                "mean effective factor count",
            ),
        },
        "headline_metrics": {
            "total_return": _finite(metrics.get("total_return"), "total return"),
            "benchmark_total_return": _finite(
                metrics.get("benchmark_total_return"),
                "benchmark total return",
            ),
            "annualized_return": _finite(
                metrics.get("annualized_return"),
                "annualized return",
            ),
            "annualized_active_return": _finite(
                metrics.get("annualized_active_return"),
                "annualized active return",
            ),
            "information_ratio": _finite(
                metrics.get("information_ratio"),
                "information ratio",
            ),
            "max_drawdown": _finite(metrics.get("max_drawdown"), "max drawdown"),
            "average_turnover": _finite(
                metrics.get("average_turnover"),
                "average turnover",
            ),
            "execution_cost_bps": _finite(
                execution.get("cost_bps_of_executed_notional"),
                "execution cost bps",
            ),
            "notional_fill_ratio": _finite(
                execution.get("notional_fill_ratio"),
                "notional fill ratio",
            ),
            "optimizer_solve_rate": _finite(
                summary.get("optimizer_solve_rate"),
                "optimizer solve rate",
            ),
            "daily_rank_ic": _finite(
                calibration.get("mean_daily_rank_ic"),
                "daily Rank IC",
            ),
            "calibration_slope": _finite(
                calibration.get("slope"),
                "calibration slope",
            ),
        },
        "innovation_ablation": {
            "evaluation_start": evidence.ablation.attrs["evaluation_start"],
            "evaluation_end": evidence.ablation.attrs["evaluation_end"],
            "rows": ablation_rows,
        },
        "stress": {
            "baseline_parity_passed": True,
            "rows": stress_rows,
        },
        "limitations": [
            "The current public backtest is not the final holdout."
            if evaluation_role != "final_holdout"
            else "The final holdout is reported without post-holdout retuning.",
            "C0-C4 is an ablation result, not evidence that the innovation dominates baselines.",
            "Costs and capacity are modeled; this is not live trading evidence.",
        ],
    }


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise ConfigurationError(
            'Public report rendering requires the optional dependency: install ".[report]"'
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


def _plot_innovation(pyplot: Any, frame: pd.DataFrame, path: Path) -> None:
    plot = frame.reset_index(drop=True)
    y_positions = list(range(len(plot)))
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    colors = [
        _RED
        if trial_id == "c4_ic_full"
        else _BLUE
        if trial_id == "c0_ic_raw"
        else _MUTED
        for trial_id in plot["trial_id"]
    ]
    values = (
        ("information_ratio", "信息比率（越高越好）"),
        ("mean_factor_weight_l1_change", "平均权重 L1 变化（越低越稳定）"),
    )
    for axis, (column, title) in zip(axes, values, strict=True):
        series = pd.to_numeric(plot[column], errors="raise").astype(float)
        reference = float(series.iloc[0])
        axis.axvline(reference, color=_GRID, linewidth=1.2, linestyle="--", zorder=0)
        axis.scatter(series, y_positions, s=52, color=colors, edgecolor=_WHITE, zorder=3)
        span = max(float(series.max() - series.min()), abs(reference) * 0.02, 0.01)
        for position, value in zip(y_positions, series, strict=True):
            axis.text(
                float(value) + span * 0.08,
                position,
                f"{float(value):.3f}",
                va="center",
                ha="left",
                color=_INK,
                fontsize=9,
            )
        axis.set_title(title, loc="left")
        axis.set_xlim(float(series.min()) - span * 0.35, float(series.max()) + span * 0.55)
        axis.set_yticks(y_positions, plot["label"])
        axis.grid(axis="x", color=_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(_GRID)
    axes[0].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(
        "图 1  IC 合成消融：稳定项降低权重漂移，同时降低样本期 IR",
        x=0.07,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.895,
        (
            f"方法选择期 {_human_date(frame.attrs['evaluation_start'])}—"
            f"{_human_date(frame.attrs['evaluation_end'])}｜"
            "筛选器、风险模型、组合优化与收益校准保持一致"
        ),
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.035,
        "注：虚线为 C0；横轴按样本范围展示。资料来源：Tushare Pro，本项目计算。",
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.79, bottom=0.19, wspace=0.20)
    _save_figure(fig, path, pyplot)


def _plot_backtest(
    pyplot: Any,
    mdates: Any,
    frame: pd.DataFrame,
    selected_trial: Mapping[str, Any],
    evaluation_role: str,
    path: Path,
) -> None:
    daily = frame.copy()
    dates = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    nav = pd.to_numeric(daily["nav"], errors="raise").astype(float)
    benchmark = pd.to_numeric(daily["benchmark_nav"], errors="raise").astype(float)
    portfolio_drawdown = nav / nav.cummax() - 1.0
    benchmark_drawdown = benchmark / benchmark.cummax() - 1.0
    summary = _as_mapping(selected_trial.get("summary"), "Selected summary")
    metrics = _as_mapping(summary.get("metrics"), "Selected metrics")

    fig, axes = pyplot.subplots(
        2,
        1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].plot(dates, nav, color=_BLUE, linewidth=2.0, label="组合")
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
        portfolio_drawdown,
        0.0,
        color=_BLUE_OPEN,
        edgecolor="none",
        label="组合回撤",
    )
    axes[1].plot(
        dates,
        benchmark_drawdown,
        color=_INK,
        linewidth=1.2,
        linestyle="--",
        label="基准回撤",
    )
    axes[1].set_ylabel("回撤")
    axes[1].yaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].legend(loc="lower left", frameon=False, ncols=2)
    axes[1].grid(axis="y", color=_GRID, linewidth=0.8)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_GRID)
        axis.set_axisbelow(True)

    role_title = {
        "method_selection": "方法选择期",
        "extended_validation": "扩展验证期",
        "final_holdout": "最终留出期",
    }[evaluation_role]
    fig.suptitle(
        f"图 2  {role_title}净值与回撤",
        x=0.075,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.900,
        (
            f"{_human_date(metrics.get('start_date'))}—"
            f"{_human_date(metrics.get('end_date'))}｜"
            f"{_EVALUATION_ROLES[evaluation_role]}｜已扣除模型化交易成本"
        ),
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.075,
        0.025,
        "注：累计净值以 1 为起点；虚线为中证 500。资料来源：Tushare Pro，本项目计算。",
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.13, hspace=0.12)
    _save_figure(fig, path, pyplot)


def _plot_stress(pyplot: Any, frame: pd.DataFrame, path: Path) -> None:
    plot = frame.reset_index(drop=True)
    baseline = plot.loc[plot["scenario_id"] == "baseline"].iloc[0]
    y_positions = list(range(len(plot)))
    colors = [
        _BLUE
        if scenario == "baseline"
        else _RED
        if str(scenario).startswith("cost_")
        else _MUTED
        for scenario in plot["scenario_id"]
    ]
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    columns = (
        ("information_ratio", "信息比率", float(baseline["information_ratio"]), ".3f"),
        (
            "cost_bps_of_executed_notional",
            "执行成本（bps / 成交金额）",
            float(baseline["cost_bps_of_executed_notional"]),
            ".2f",
        ),
    )
    for axis, (column, title, reference, format_spec) in zip(axes, columns, strict=True):
        series = pd.to_numeric(plot[column], errors="raise").astype(float)
        for position, value, color in zip(y_positions, series, colors, strict=True):
            axis.hlines(
                position,
                min(reference, float(value)),
                max(reference, float(value)),
                color=_GRID,
                linewidth=2.0,
                zorder=1,
            )
            axis.scatter(float(value), position, s=48, color=color, edgecolor=_WHITE, zorder=3)
            axis.text(
                float(value) + max(abs(float(series.max())) * 0.025, 0.01),
                position,
                format(float(value), format_spec),
                va="center",
                ha="left",
                color=_INK,
                fontsize=9,
            )
        axis.axvline(reference, color=_INK, linewidth=1.1, linestyle="--")
        axis.set_title(title, loc="left")
        axis.set_xlim(min(0.0, float(series.min()) * 1.10), float(series.max()) * 1.18)
        axis.set_yticks(y_positions, plot["label"])
        axis.grid(axis="x", color=_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(_GRID)
    axes[0].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(
        "图 3  成本与容量压力测试",
        x=0.07,
        ha="left",
        fontsize=14,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.900,
        "固定预测信号，仅重新估计风险、组合优化与模拟执行；虚线为基准情景",
        ha="left",
        color=_MUTED,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.025,
        (
            "注：成本倍数同时作用于优化惩罚与模拟成交成本；基准情景为 1 亿元规模、"
            "5% ADV 上限。资料来源：Tushare Pro，本项目计算。"
        ),
        ha="left",
        color=_MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.16, wspace=0.20)
    _save_figure(fig, path, pyplot)


def _save_figure(figure: Any, path: Path, pyplot: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        figure.savefig(
            temporary,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "csi500-alpha public report"},
        )
        os.replace(temporary, path)
    finally:
        pyplot.close(figure)
        temporary.unlink(missing_ok=True)


def _validate_daily(frame: pd.DataFrame) -> None:
    required = {"trade_date", "nav", "benchmark_nav"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConfigurationError(f"Selected daily backtest lacks columns: {missing}")
    if len(frame) < 12:
        raise ConfigurationError("Selected daily backtest is too sparse for a trend chart")
    dates = frame["trade_date"].astype(str)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ConfigurationError("Selected daily backtest dates must be unique and increasing")
    for column in ("nav", "benchmark_nav"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.map(math.isfinite)).any() or (values <= 0).any():
            raise ConfigurationError(f"Selected daily backtest has invalid {column}")


def _contained_path(root: Path, reference: Any, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ConfigurationError(f"{label} must be a non-empty relative path")
    candidate = Path(reference)
    if candidate.is_absolute():
        raise ConfigurationError(f"{label} must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigurationError(f"{label} escapes its Study root")
    if not resolved.is_dir():
        raise ConfigurationError(f"{label} does not exist: {reference}")
    return resolved


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _assert_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigurationError(f"{label} is not valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must contain a JSON object")
    return value


def _assert_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path}")


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
    text = str(value)
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Report date must be parseable: {text}") from exc
