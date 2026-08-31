from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.utils import sha256_file
from csi500_alpha.v2_reporting import build_v2_readme_report


def test_v2_readme_report_exports_fingerprinted_aggregate_evidence(tmp_path: Path) -> None:
    baseline_root, expanded_root, selection_root, factor_root = _v2_report_fixture(tmp_path)
    output = tmp_path / "public"

    result = build_v2_readme_report(
        baseline_root=baseline_root,
        expanded_root=expanded_root,
        selection_root=selection_root,
        factor_audit_root=factor_root,
        output_root=output,
    )

    assert result.summary_path.is_file()
    assert result.manifest_path.is_file()
    assert len(result.figure_paths) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in result.figure_paths)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["scope"]["evaluation_role"] == "rolling_annual_research"
    assert summary["released_method"]["positive_active_years"] == 2
    assert summary["release"]["version"] == "2.0.0"
    assert summary["release"]["status"] == "research_release"
    gate_assessment = summary["release"]["publication_gate_assessment"]
    assert gate_assessment["passed"] is False
    assert gate_assessment["failed_rule_count"] == 1
    signal_check = summary["release"]["candidate_signal_check"]
    assert signal_check["portfolio_path_identical"] is True
    assert signal_check["maximum_actual_candidate_share"] == 0.0
    assert signal_check["active_allocation_years"] == 0
    assert summary["expanded_pool_ablation"]["promoted"] is False
    assert summary["factor_audit"]["candidate_count"] == 4
    assert summary["factor_audit"]["eligible_count"] == 2
    assert summary["factor_audit"]["eligible_correlation_component_count"] == 2
    assert len(summary["annual_comparison"]) == 2
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert "trade_date" not in result.summary_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert set(manifest["outputs"]) == {
        "v2-backtest-overview.png",
        "v2-factor-audit.png",
        "v2-freeze-rationale.png",
        "v2-public-summary.json",
    }
    assert all(
        "comparison-baseline/" in key
        or "expanded/" in key
        or "selection/" in key
        or "factor-audit/" in key
        for key in manifest["source_artifacts"]
    )


def test_v2_readme_report_rejects_tampered_annual_evidence(tmp_path: Path) -> None:
    baseline_root, expanded_root, selection_root, factor_root = _v2_report_fixture(tmp_path)
    daily_path = baseline_root / "daily.parquet"
    daily = pd.read_parquet(daily_path)
    daily.loc[0, "nav"] = 99.0
    daily.to_parquet(daily_path, index=False)

    with pytest.raises(ConfigurationError, match="fingerprint does not match"):
        build_v2_readme_report(
            baseline_root=baseline_root,
            expanded_root=expanded_root,
            selection_root=selection_root,
            factor_audit_root=factor_root,
            output_root=tmp_path / "public",
        )


def test_repository_readme_matches_published_v2_evidence() -> None:
    repository = Path(__file__).resolve().parents[1]
    assets = repository / "docs" / "assets"
    readme = (repository / "README.md").read_text(encoding="utf-8")
    summary_text = (assets / "v2-public-summary.json").read_text(encoding="utf-8")
    manifest_text = (assets / "v2-report-manifest.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    manifest = json.loads(manifest_text)
    baseline = summary["released_method"]
    factor_audit = summary["factor_audit"]
    latest_factor_year = factor_audit["yearly_rows"][-1]

    expected_snippets = {
        f"{baseline['annualized_active_return']:.2%}",
        f"{baseline['relative_active_total_return']:.2%}",
        f"{baseline['information_ratio']:.3f}",
        f"{baseline['active_max_drawdown']:.2%}",
        f"{baseline['capm_alpha_annualized']:.2%}",
        f"{baseline['capm_beta']:.3f}",
        f"{baseline['positive_active_years']}/{baseline['year_count']}",
        f"{factor_audit['mean_coverage']:.1%}",
        f"{factor_audit['eligible_correlation_component_count']} 个连通分量",
        (f"{latest_factor_year['positive_net_spread_count']}/{latest_factor_year['factor_count']}"),
        f"{latest_factor_year['median_net_spread'] * 10_000:.2f} 个基点",
    }
    assert all(snippet in readme for snippet in expected_snippets)
    for name, expected in manifest["outputs"].items():
        assert sha256_file(assets / name) == expected
    assert "D:\\" not in summary_text
    assert "C:\\" not in summary_text
    assert "D:\\" not in manifest_text
    assert "C:\\" not in manifest_text
    assert "A0" not in readme
    assert "A3.2" not in readme


def _v2_report_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    baseline_root = tmp_path / "annual" / "baseline"
    expanded_root = tmp_path / "annual" / "expanded"
    dates = pd.DatetimeIndex(
        [*pd.bdate_range("2020-01-02", periods=24), *pd.bdate_range("2021-01-04", periods=24)]
    )
    benchmark_returns = pd.Series([0.0, *([0.001] * (len(dates) - 1))])
    benchmark_nav = (1.0 + benchmark_returns).cumprod()
    baseline_returns = benchmark_returns + pd.Series([0.0, *([0.0001] * (len(dates) - 1))])
    expanded_returns = benchmark_returns + pd.Series([0.0, *([0.00012] * (len(dates) - 1))])
    baseline_nav = (1.0 + baseline_returns).cumprod()
    expanded_nav = (1.0 + expanded_returns).cumprod()
    baseline_daily = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "nav": baseline_nav,
            "benchmark_nav": benchmark_nav,
            "active_nav": baseline_nav / benchmark_nav,
        }
    )
    expanded_daily = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "nav": expanded_nav,
            "benchmark_nav": benchmark_nav,
            "active_nav": expanded_nav / benchmark_nav,
        }
    )
    baseline_yearly = pd.DataFrame(
        {
            "year": [2020, 2021],
            "annualized_active_return": [0.02, 0.03],
            "active_total_return": [0.019, 0.029],
            "information_ratio": [0.4, 0.6],
        }
    )
    expanded_yearly = pd.DataFrame(
        {
            "year": [2020, 2021],
            "annualized_active_return": [0.01, 0.05],
            "active_total_return": [0.009, 0.048],
            "information_ratio": [0.2, 0.8],
        }
    )
    _write_annual_aggregate(
        baseline_root,
        trial_id="baseline",
        daily=baseline_daily,
        yearly=baseline_yearly,
        annualized_active_return=0.025,
        information_ratio=0.5,
        positive_years=2,
    )
    _write_annual_aggregate(
        expanded_root,
        trial_id="expanded",
        daily=expanded_daily,
        yearly=expanded_yearly,
        annualized_active_return=0.03,
        information_ratio=0.55,
        positive_years=2,
    )

    selection_root = tmp_path / "release-selection"
    released_root = selection_root / "aggregates" / "released"
    challenger_root = selection_root / "aggregates" / "challenger"
    _write_annual_aggregate(
        released_root,
        annual_id="fixture-release-annual",
        trial_id="released",
        daily=baseline_daily,
        yearly=baseline_yearly,
        annualized_active_return=0.025,
        information_ratio=0.5,
        positive_years=2,
    )
    challenger_fits = pd.DataFrame(
        {
            "fold_year": [2020, 2021],
            "fit_date": ["20200102", "20210104"],
            "status": ["selected", "selected"],
            "model_parameters": [
                json.dumps(
                    {
                        "candidate_share": 0.0,
                        "sleeve_evidence": {
                            "oof_t": -0.5,
                            "oof_evidence_passed": False,
                        },
                    }
                ),
                json.dumps(
                    {
                        "candidate_share": 0.0,
                        "sleeve_evidence": {
                            "oof_t": -0.25,
                            "oof_evidence_passed": False,
                        },
                    }
                ),
            ],
        }
    )
    _write_annual_aggregate(
        challenger_root,
        annual_id="fixture-release-annual",
        trial_id="challenger",
        daily=baseline_daily,
        yearly=baseline_yearly,
        annualized_active_return=0.025,
        information_ratio=0.5,
        positive_years=2,
        model_fits=challenger_fits,
    )
    release_status = "completed_without_publishable_candidate"
    _write_json(
        selection_root / "progress.json",
        {
            "status": release_status,
            "completed_tasks": 4,
            "pending_tasks": 0,
        },
    )
    _write_json(
        selection_root / "selection.json",
        {"selected_trial_id": "released"},
    )
    publication_rules = [
        {"path": "quality_passed", "operator": "==", "value": True},
        {"path": "metrics.annualized_active_return", "operator": ">=", "value": 0.03},
    ]
    _write_json(
        selection_root / "publication-gates.json",
        {
            "rules": publication_rules,
            "trials": [
                {"trial_id": "released", "passed": False},
                {"trial_id": "challenger", "passed": False},
            ],
        },
    )
    _write_json(
        selection_root / "annual-study-manifest.json",
        {
            "annual_id": "fixture-release-annual",
            "status": release_status,
            "task_count": 4,
            "completed_task_count": 4,
            "failed_task_count": 0,
            "pending_task_count": 0,
            "selected_trial_id": "released",
            "registry": {
                "current_candidates": [
                    {"id": "released", "included_in_selection": True},
                    {"id": "challenger", "included_in_selection": True},
                ]
            },
        },
    )

    factor_root = tmp_path / "factor-audit"
    factor_root.mkdir(parents=True)
    factors = pd.DataFrame(
        {
            "factor": ["value_a", "value_b", "risk_a", "size_a"],
            "family": ["value", "value", "risk", "size"],
            "eligible": [True, False, True, False],
            "mean_coverage": [0.95, 0.93, 1.0, 1.0],
            "mean_directed_rank_ic": [0.03, 0.01, 0.04, -0.01],
            "median_yearly_q5_minus_q1_net": [0.001, -0.0002, 0.0015, -0.0004],
        }
    )
    yearly_factor = pd.DataFrame(
        {
            "year": [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
            "factor": [
                "value_a",
                "value_b",
                "risk_a",
                "size_a",
                "value_a",
                "value_b",
                "risk_a",
                "size_a",
            ],
            "mean_q5_minus_q1_net": [
                0.001,
                -0.001,
                0.002,
                -0.002,
                0.0005,
                -0.001,
                -0.0002,
                -0.002,
            ],
        }
    )
    factor_summary_path = factor_root / "factor-summary.parquet"
    correlation_path = factor_root / "factor-correlation.parquet"
    yearly_factor_path = factor_root / "yearly-audit.parquet"
    factors.to_parquet(factor_summary_path, index=False)
    pd.DataFrame(
        {
            "factor": ["value_a", "value_b", "risk_a", "size_a"],
            "value_a": [1.0, 0.8, 0.1, 0.0],
            "value_b": [0.8, 1.0, 0.2, 0.0],
            "risk_a": [0.1, 0.2, 1.0, 0.1],
            "size_a": [0.0, 0.0, 0.1, 1.0],
        }
    ).to_parquet(correlation_path, index=False)
    yearly_factor.to_parquet(yearly_factor_path, index=False)
    factor_summary = {
        "status": "success",
        "audit_id": "fixture-audit",
        "run_id": "fixture-run",
        "start_date": "20200102",
        "end_date": "20211231",
        "decision_dates": 48,
        "panel_rows": 4_800,
        "factor_count": 4,
        "eligible_factor_count": 2,
        "point_in_time_violations": 0,
        "all_data_quality_checks_passed": True,
    }
    factor_summary_json = factor_root / "factor-audit-summary.json"
    _write_json(factor_summary_json, factor_summary)
    _write_json(
        factor_root / "factor-audit-manifest.json",
        {
            **factor_summary,
            "artifact_fingerprints": {
                "factor_audit_summary": sha256_file(factor_summary_json),
                "factor_correlation": sha256_file(correlation_path),
                "factor_summary": sha256_file(factor_summary_path),
                "yearly_audit": sha256_file(yearly_factor_path),
            },
        },
    )
    return baseline_root, expanded_root, selection_root, factor_root


def _write_annual_aggregate(
    root: Path,
    *,
    annual_id: str = "fixture-annual",
    trial_id: str,
    daily: pd.DataFrame,
    yearly: pd.DataFrame,
    annualized_active_return: float,
    information_ratio: float,
    positive_years: int,
    model_fits: pd.DataFrame | None = None,
) -> None:
    root.mkdir(parents=True)
    daily_path = root / "daily.parquet"
    yearly_path = root / "yearly-metrics.parquet"
    summary_path = root / "aggregate-summary.json"
    daily.to_parquet(daily_path, index=False)
    yearly.to_parquet(yearly_path, index=False)
    artifact_fingerprints = {
        "daily.parquet": sha256_file(daily_path),
        "yearly-metrics.parquet": sha256_file(yearly_path),
    }
    if model_fits is not None:
        model_fits_path = root / "model-fits.parquet"
        model_fits.to_parquet(model_fits_path, index=False)
        artifact_fingerprints["model-fits.parquet"] = sha256_file(model_fits_path)
    summary = {
        "trial_id": trial_id,
        "fold_count": 2,
        "quality_passed": True,
        "optimizer_solve_rate": 1.0,
        "metrics": {
            "start_date": str(daily["trade_date"].iloc[0]),
            "end_date": str(daily["trade_date"].iloc[-1]),
            "annualized_active_return": annualized_active_return,
            "relative_active_total_return": float(daily["active_nav"].iloc[-1] - 1.0),
            "information_ratio": information_ratio,
            "tracking_error": 0.05,
            "active_max_drawdown": -0.03,
            "capm_alpha_annualized": 0.03,
            "capm_beta": 0.95,
            "average_turnover": 0.02,
            "transaction_cost": 0.01,
        },
        "evaluation": {
            "yearly": {
                "positive_active_years": positive_years,
                "year_count": 2,
                "minimum_active_total_return": 0.01,
            },
            "execution": {
                "cost_bps_of_executed_notional": 10.0,
                "notional_fill_ratio": 0.99,
            },
        },
    }
    _write_json(summary_path, summary)
    _write_json(
        root / "aggregate-manifest.json",
        {
            "status": "completed",
            "error": None,
            "annual_id": annual_id,
            "data_snapshot_hash": "fixture-data-snapshot",
            "summary": summary,
            "artifact_fingerprints": {
                "aggregate-summary.json": sha256_file(summary_path),
                **artifact_fingerprints,
            },
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
