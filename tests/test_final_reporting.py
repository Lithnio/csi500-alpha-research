from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.final_reporting import (
    _recalculate_metrics,
    build_final_holdout_report,
)
from csi500_alpha.utils import canonical_json, sha256_file, sha256_text


def test_final_report_audits_and_exports_only_aggregate_evidence(
    tmp_path: Path,
) -> None:
    run_root = _final_run_fixture(tmp_path)
    output = tmp_path / "public"

    result = build_final_holdout_report(run_root=run_root, output_root=output)

    assert result.figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["evaluation_role"] == "final_holdout"
    assert summary["interpretation"]["status"] == "negative_active_holdout"
    assert summary["audit"]["maximum_metric_recalculation_difference"] == 0.0
    assert summary["audit"]["point_in_time_violations"] == 0
    assert summary["headline_metrics"]["return_gap_percentage_points"] == pytest.approx(
        (
            summary["headline_metrics"]["total_return"]
            - summary["headline_metrics"]["benchmark_total_return"]
        )
        * 100.0
    )
    assert len(summary["method"]["final_material_factor_weights"]) == 3
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["identity"]["lock_matches_run"] is True
    assert audit["quality_audit"]["artifact_hash_mismatches"] == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_role"] == "final_holdout"
    assert set(manifest["outputs"]) == {
        "final-holdout-audit.json",
        "final-holdout-summary.json",
        "final-holdout.png",
    }
    for path in (result.summary_path, result.audit_path, result.manifest_path):
        assert b"\r\n" not in path.read_bytes()
    for name, expected in manifest["outputs"].items():
        assert sha256_file(output / name) == expected
    public_text = result.summary_path.read_text(encoding="utf-8")
    public_text += result.audit_path.read_text(encoding="utf-8")
    public_text += result.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert '"instrument"' not in public_text


def test_final_report_rejects_tampered_daily_artifact(tmp_path: Path) -> None:
    run_root = _final_run_fixture(tmp_path)
    daily_path = run_root / "daily.parquet"
    daily = pd.read_parquet(daily_path)
    daily.loc[0, "nav"] = 9.0
    daily.to_parquet(daily_path, index=False)

    with pytest.raises(ConfigurationError, match="fingerprints differ"):
        build_final_holdout_report(
            run_root=run_root,
            output_root=tmp_path / "public",
        )


def _final_run_fixture(tmp_path: Path) -> Path:
    run_id = "20260827T000000000000Z-fixture"
    validation_id = "20260826T000000000000Z-fixture"
    protocol_id = "fixture-final-protocol"
    run_root = tmp_path / "runs" / run_id
    validation_root = tmp_path / "runs" / validation_id
    run_root.mkdir(parents=True)
    validation_root.mkdir(parents=True)
    dates = pd.bdate_range("2026-01-05", periods=24).strftime("%Y%m%d")
    nav = pd.Series(1.0 - pd.Series(range(len(dates))) * 0.001)
    benchmark_nav = pd.Series(1.0 + pd.Series(range(len(dates))) * 0.001)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "nav": nav,
            "cash": 0.0,
            "holdings": 1.0,
            "gross_trade_value": 0.01,
            "turnover": 0.01,
            "transaction_cost": 0.00001,
            "benchmark_nav": benchmark_nav,
            "portfolio_return": nav.pct_change().fillna(0.0),
            "benchmark_return": benchmark_nav.pct_change().fillna(0.0),
            "active_return": (
                nav.pct_change().fillna(0.0)
                - benchmark_nav.pct_change().fillna(0.0)
            ),
            "active_nav": 1.0,
        }
    )
    trades = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[5], dates[10]],
            "trade_date": [dates[1], dates[6], dates[11]],
            "instrument": ["A", "B", "C"],
            "status": ["filled", "partial", "blocked"],
            "linear_cost": [0.0001, 0.0001, 0.0],
            "stamp_duty": [0.0, 0.0001, 0.0],
            "impact_cost": [0.0001, 0.0001, 0.0],
        }
    )
    metrics = _recalculate_metrics(daily, trades)
    evaluation = {
        "calibration": {
            "mean_daily_rank_ic": 0.01,
            "slope": -0.5,
            "quintile_monotonicity": -0.5,
            "top_minus_bottom_realized_return": -0.01,
        },
        "execution": {
            "cost_bps_of_executed_notional": 8.0,
            "notional_fill_ratio": 0.99,
        },
        "model_weights": {"minimum_effective_factor_count": 3.0},
    }
    factor_names = ["factor_a", "factor_b", "factor_c"]
    summary = {
        "selector": "stability_cost",
        "model": "ic_shrinkage",
        "calibrator": "rolling_ridge",
        "factors": factor_names,
        "artifact_date_ranges": {
            "features": {"start": dates[0], "end": dates[-1]},
            "labels": {"start": dates[0], "end": dates[-1]},
            "label_availability": {"start": dates[1], "end": dates[-1]},
            "factor_diagnostics": {"start": dates[0], "end": dates[-1]},
            "signals": {"start": dates[0], "end": dates[-1]},
            "evaluation_signals": {"start": dates[0], "end": dates[-1]},
            "backtest": {"start": dates[0], "end": dates[-1]},
        },
        "metrics": metrics,
        "evaluation": evaluation,
    }

    frames = {
        "daily": daily,
        "trades": trades,
        "signals": pd.DataFrame(
            {
                "decision_date": dates,
                "instrument": ["A"] * len(dates),
                "model_fit_date": [dates[0]] * len(dates),
                "calibrator_fit_date": [dates[0]] * len(dates),
            }
        ),
        "evaluation_signals": pd.DataFrame(
            {
                "decision_date": dates,
                "instrument": ["A"] * len(dates),
            }
        ),
        "factor_ic": pd.DataFrame(
            {
                "decision_date": dates,
                "factor": ["factor_a"] * len(dates),
                "rank_ic": 0.01,
            }
        ),
        "targets": pd.DataFrame(
            {
                "execution_date": dates[1:],
                "instrument": ["A"] * (len(dates) - 1),
            }
        ),
        "model_fits": pd.DataFrame(
            {
                "fit_date": [dates[0]],
                "max_label_available_date": ["20251231"],
            }
        ),
        "calibration_fits": pd.DataFrame(
            {
                "fit_date": [dates[0]],
                "max_label_available_date": ["20251231"],
            }
        ),
        "factor_weight_history": pd.DataFrame(
            {
                "fit_date": [dates[0]] * 3,
                "factor": factor_names,
                "allocation_weight": [0.4, 0.35, 0.25],
            }
        ),
        "optimization": pd.DataFrame(
            {
                "decision_date": [dates[0]],
                "execution_date": [dates[1]],
                "status": ["optimal"],
                "maximum_violation": [0.0],
            }
        ),
        "calibration_bins": pd.DataFrame(
            {
                "bin": range(1, 6),
                "observations": [100] * 5,
                "dates": [20] * 5,
                "mean_expected_return": [-0.02, -0.01, 0.0, 0.01, 0.02],
                "mean_realized_active_return": [0.01, 0.0, -0.01, -0.02, -0.03],
            }
        ),
    }
    artifact_fingerprints: dict[str, str] = {}
    for name, frame in frames.items():
        path = run_root / f"{name.replace('_', '-')}.parquet"
        frame.to_parquet(path, index=False)
        artifact_fingerprints[name] = sha256_file(path)
    for name, value in (
        ("metrics", metrics),
        ("research_evaluation", evaluation),
        ("workflow_summary", summary),
    ):
        path = run_root / f"{name.replace('_', '-')}.json"
        _write_json(path, value)
        artifact_fingerprints[name] = sha256_file(path)

    gold_root = run_root / "gold"
    gold_root.mkdir()
    features = pd.DataFrame(
        {
            "decision_date": dates,
            "instrument": ["A"] * len(dates),
        }
    )
    labels = pd.DataFrame(
        {
            "decision_date": dates,
            "instrument": ["A"] * len(dates),
            "label_available_date": [*dates[1:], dates[-1]],
        }
    )
    features.to_parquet(gold_root / "features.parquet", index=False)
    labels.to_parquet(gold_root / "labels.parquet", index=False)
    gold_fingerprints = {
        "features": sha256_file(gold_root / "features.parquet"),
        "labels": sha256_file(gold_root / "labels.parquet"),
    }

    silver_root = tmp_path / "data" / "silver" / "final_2026"
    silver_root.mkdir(parents=True)
    pd.DataFrame({"trade_date": dates}).to_parquet(
        silver_root / "stock_bars.parquet",
        index=False,
    )
    data_fingerprints = {
        "stock_bars": sha256_file(silver_root / "stock_bars.parquet")
    }
    data_snapshot_hash = sha256_text(
        canonical_json(dict(sorted(data_fingerprints.items())))
    )
    experiment = {
        "stage": "frozen_test",
        "protocol_id": protocol_id,
        "train_start": "20170103",
        "train_end": "20221230",
        "validation_start": "20230103",
        "validation_end": "20251231",
        "test_start": dates[0],
        "test_end": dates[-1],
        "evaluation_start": dates[0],
        "evaluation_end": dates[-1],
        "research_spec_hash": "spec-hash",
        "research_source_hash": "source-hash",
        "data_snapshot_hash": data_snapshot_hash,
    }
    run_manifest = {
        "run_id": run_id,
        "dataset": "final_2026",
        "experiment": experiment,
        "artifact_fingerprints": artifact_fingerprints,
        "gold_fingerprints": gold_fingerprints,
        "data_fingerprints": data_fingerprints,
        "git": {"commit": "abc123", "dirty": False},
        "quality": {
            "checks": [
                {"name": "fixture", "passed": True, "severity": "error"}
            ]
        },
        "summary": summary,
    }
    _write_json(run_root / "run-manifest.json", run_manifest)
    _write_json(
        validation_root / "run-manifest.json",
        {"gold_fingerprints": {name: f"old-{value}" for name, value in gold_fingerprints.items()}},
    )

    protocol_root = tmp_path / "runs" / "_protocols" / protocol_id
    protocol_root.mkdir(parents=True)
    identity = {
        "protocol_id": protocol_id,
        "research_spec_hash": "spec-hash",
        "research_source_hash": "source-hash",
        "data_snapshot_hash": data_snapshot_hash,
        "data_fingerprints": data_fingerprints,
    }
    _write_json(
        protocol_root / "validation-lock.json",
        {**identity, "stage": "validation", "run_id": validation_id},
    )
    _write_json(
        protocol_root / "frozen-test.json",
        {
            **identity,
            "stage": "frozen_test",
            "run_id": run_id,
            "validation_run_id": validation_id,
            "completed_at": "2026-08-27T00:00:00+00:00",
        },
    )
    return run_root


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
