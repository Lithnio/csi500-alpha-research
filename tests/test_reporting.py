from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.reporting import build_public_report
from csi500_alpha.utils import sha256_file


def test_public_report_exports_only_aggregate_evidence(tmp_path: Path) -> None:
    study_root, stress_root = _report_fixture(tmp_path)
    output = tmp_path / "public"

    result = build_public_report(
        study_root=study_root,
        stress_root=stress_root,
        output_root=output,
    )

    assert result.summary_path.is_file()
    assert result.manifest_path.is_file()
    assert len(result.figure_paths) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in result.figure_paths)
    summary_text = result.summary_path.read_text(encoding="utf-8")
    assert "trade_date" not in summary_text
    assert "instrument" not in summary_text
    summary = json.loads(summary_text)
    assert summary["scope"]["evaluation_role"] == "method_selection"
    assert summary["headline_metrics"]["relative_active_total_return"] == pytest.approx(
        0.0714285714
    )
    assert summary["headline_metrics"]["active_max_drawdown"] == -0.08
    assert summary["headline_metrics"]["portfolio_max_drawdown"] == -0.20
    assert summary["headline_metrics"]["capm_alpha_annualized"] == 0.04
    assert summary["headline_metrics"]["capm_beta"] == 0.95
    assert summary["headline_metrics"][
        "maximum_post_trade_active_beta_deviation"
    ] == 0.04
    assert summary["headline_metrics"][
        "post_trade_policy_violation_fraction"
    ] == 0.0
    assert summary["headline_metrics"]["post_trade_audit_count"] == 23
    assert summary["headline_metrics"]["beta_audit_complete_fraction"] == 1.0
    assert len(summary["innovation_ablation"]["rows"]) == 5
    assert len(summary["stress"]["rows"]) == 4
    assert {row["execution_mode"] for row in summary["stress"]["rows"]} == {
        "reoptimized"
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {
        "backtest-overview.png",
        "innovation-ablation.png",
        "public-summary.json",
        "stress-analysis.png",
    }
    assert "not mechanically monotonic" in (
        manifest["chart_map"]["stress-analysis.png"]["note"]
    )
    assert all("D:\\" not in key for key in manifest["source_artifacts"])


def test_public_report_rejects_tampered_selected_daily(tmp_path: Path) -> None:
    study_root, stress_root = _report_fixture(tmp_path)
    daily_path = study_root / "trials" / "selected" / "attempt-001" / "daily.parquet"
    daily = pd.read_parquet(daily_path)
    daily.loc[0, "nav"] = 999.0
    daily.to_parquet(daily_path, index=False)

    with pytest.raises(ConfigurationError, match="fingerprint does not match"):
        build_public_report(
            study_root=study_root,
            stress_root=stress_root,
            output_root=tmp_path / "public",
        )


def _report_fixture(tmp_path: Path) -> tuple[Path, Path]:
    study_root = tmp_path / "studies" / "example-study"
    selected_root = study_root / "trials" / "selected" / "attempt-001"
    selected_root.mkdir(parents=True)
    dates = pd.bdate_range("2021-01-04", periods=24).strftime("%Y%m%d")
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "nav": 1.0 + pd.Series(range(len(dates))) * 0.002,
            "benchmark_nav": 1.0 - pd.Series(range(len(dates))) * 0.001,
        }
    )
    daily_path = selected_root / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    selected_summary = _trial_summary(
        information_ratio=1.4,
        l1_change=0.25,
        effective_factors=8.0,
    )
    run_manifest = {
        "artifact_fingerprints": {"daily": sha256_file(daily_path)},
        "experiment": {
            "train_start": "20170103",
            "train_end": "20201231",
            "evaluation_start": "20210104",
            "evaluation_end": "20211231",
        },
        "summary": selected_summary,
    }
    _write_json(selected_root / "run-manifest.json", run_manifest)
    selected_manifest = {
        "status": "completed",
        "artifact_root": "trials/selected/attempt-001",
        "trial": {"id": "selected", "purpose": "fixture", "overrides": {}},
        "summary": selected_summary,
    }
    _write_json(study_root / "trials" / "selected" / "trial-manifest.json", selected_manifest)
    _write_json(
        study_root / "study-manifest.json",
        {
            "status": "completed",
            "study_id": "example-study",
            "selected_trial_id": "selected",
            "completed_count": 6,
            "trial_count": 6,
            "data_snapshot_hash": "snapshot-hash",
            "git": {"commit": "abc123", "dirty": False},
        },
    )
    _write_json(
        study_root / "selection.json",
        {"study_id": "example-study", "selected_trial_id": "selected"},
    )
    for position, trial_id in enumerate(
        (
            "c0_ic_raw",
            "c1_ic_uncertainty",
            "c2_ic_correlation",
            "c3_ic_cost",
            "c4_ic_full",
        )
    ):
        summary = _trial_summary(
            information_ratio=0.8 - position * 0.02,
            l1_change=0.22 - position * 0.01,
            effective_factors=3.0 + position * 0.1,
        )
        _write_json(
            study_root / "trials" / trial_id / "trial-manifest.json",
            {
                "status": "completed",
                "trial": {"id": trial_id},
                "summary": summary,
            },
        )

    stress_root = study_root / "stress" / "example-stress"
    stress_root.mkdir(parents=True)
    _write_json(
        stress_root / "stress-manifest.json",
        {
            "status": "completed",
            "source_study_id": "example-study",
            "source_trial_id": "selected",
            "data_snapshot_hash": "snapshot-hash",
            "source_run_manifest_hash": sha256_file(selected_root / "run-manifest.json"),
            "baseline_parity_passed": True,
            "scenario_count": 4,
        },
    )
    _write_json(stress_root / "stress-summary.json", {"baseline_parity_passed": True})
    for position, scenario_id in enumerate(
        ("cost_0_5x", "baseline", "cost_2_0x", "aum_50m")
    ):
        _write_json(
            stress_root / "scenarios" / scenario_id / "scenario-manifest.json",
            {
                "status": "completed",
                "scenario": {"id": scenario_id},
                "summary": {
                    "metrics": {
                        "information_ratio": 1.5 - position * 0.1,
                        "annualized_active_return": 0.08 - position * 0.005,
                        "max_drawdown": -0.2 - position * 0.01,
                        "active_max_drawdown": -0.08 - position * 0.005,
                        "portfolio_max_drawdown": -0.2 - position * 0.01,
                        "post_trade_policy_violation_fraction": 0.0,
                        "maximum_post_trade_active_beta_deviation": 0.04,
                        "beta_audit_complete_fraction": 1.0,
                        "average_turnover": 0.04 + position * 0.002,
                    },
                    "evaluation": {
                        "yearly": {"minimum_information_ratio": 1.0 - position * 0.1},
                        "execution": {
                            "notional_fill_ratio": 0.99 - position * 0.001,
                            "cost_bps_of_executed_notional": 7.0 + position * 4.0,
                        },
                    },
                    "optimizer_solve_rate": 1.0,
                },
            },
        )
    return study_root, stress_root


def _trial_summary(
    *,
    information_ratio: float,
    l1_change: float,
    effective_factors: float,
) -> dict[str, object]:
    return {
        "selector": "stability_cost",
        "model": "direction_equal_weight",
        "calibrator": "rolling_ridge",
        "optimizer_solve_rate": 1.0,
        "artifact_date_ranges": {
            "backtest": {"start": "20210104", "end": "20211231"}
        },
        "metrics": {
            "start_date": "20210104",
            "end_date": "20211231",
            "total_return": 0.05,
            "benchmark_total_return": -0.02,
            "annualized_return": 0.05,
            "annualized_active_return": 0.07,
            "relative_active_total_return": 0.0714285714,
            "information_ratio": information_ratio,
            "max_drawdown": -0.20,
            "active_max_drawdown": -0.08,
            "portfolio_max_drawdown": -0.20,
            "capm_alpha_annualized": 0.04,
            "capm_beta": 0.95,
            "post_trade_audit_count": 23,
            "maximum_post_trade_active_beta_deviation": 0.04,
            "maximum_post_trade_industry_active_exposure": 0.015,
            "post_trade_policy_violation_fraction": 0.0,
            "beta_audit_complete_fraction": 1.0,
            "average_turnover": 0.04,
        },
        "evaluation": {
            "calibration": {"mean_daily_rank_ic": 0.04, "slope": 0.8},
            "execution": {
                "cost_bps_of_executed_notional": 12.0,
                "notional_fill_ratio": 0.99,
            },
            "model_weights": {
                "mean_factor_weight_l1_change": l1_change,
                "mean_effective_factor_count": effective_factors,
            },
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
