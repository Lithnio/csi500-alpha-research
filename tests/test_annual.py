from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.annual import (
    AnnualFold,
    AnnualFoldExecution,
    AnnualStudyRunner,
    AnnualStudySpec,
    AnnualTrialAggregate,
    annual_study_plan,
    build_annual_folds,
    resolve_annual_fold_config,
)
from csi500_alpha.config import AppConfig
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.pipeline import _aggregate_annual_trial, _audit_fold_fits
from csi500_alpha.study import StudyContext, StudyTrial
from csi500_alpha.utils import sha256_file


def test_repository_annual_plan_freezes_six_nonoverlapping_folds() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "annual" / "factor_family_v3_repair.yaml"
    spec = AnnualStudySpec.from_yaml(path)
    base = AppConfig.from_yaml(spec.method_study.base_config_path)
    open_dates = [
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2017-01-03", "2025-12-31")
    ]

    folds = build_annual_folds(spec, base, open_dates)
    plan = annual_study_plan(path, open_dates=open_dates)

    assert [fold.year for fold in folds] == list(range(2020, 2026))
    assert plan["task_count"] == 18
    assert plan["fold_plan_hash"]
    assert plan["registry"]["declared_current_candidate_count"] == 3
    assert plan["registry"]["declared_historical_experiment_count"] == 3
    assert plan["annual_id"] == "factor-family-v3-repair-annual-2020-2025"
    assert {
        item["id"] for item in plan["registry"]["historical_experiments"]
    } == {
        "v1-method-selection-2023-2025",
        "v1-revealed-2026-h1",
        "v2-annual-2020-constraint-diagnostic",
    }
    gate_paths = {item["path"] for item in plan["publication_gates"]}
    assert {
        "evaluation.constraints.post_trade_material_configured_breach_fraction",
        "evaluation.constraints.post_trade_material_execution_deterioration_fraction",
        "evaluation.constraints.p95_actual_active_risk_utilization",
    } <= gate_paths
    assert all(
        left.evaluation_end < right.evaluation_start
        for left, right in zip(folds, folds[1:], strict=False)
    )
    assert all(fold.last_mature_label_date < fold.embargo_start for fold in folds)
    assert all(fold.embargo_start < fold.evaluation_start for fold in folds)
    assert all(fold.first_decision_date == fold.evaluation_start for fold in folds)

    trial = spec.method_study.trials[0]
    method = AppConfig.from_yaml(spec.method_study.base_config_path)
    resolved = resolve_annual_fold_config(
        method,
        annual_id=spec.annual_id,
        trial_id=trial.trial_id,
        fold=folds[0],
        embargo_days=spec.embargo_days,
    )
    assert resolved.experiment.stage == "validation"
    assert resolved.experiment.train_end == folds[0].train_end
    assert resolved.experiment.validation_start == folds[0].evaluation_start
    assert resolved.experiment.validation_end == folds[0].evaluation_end


def test_a1_annual_plan_freezes_four_risk_trials_across_six_folds() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "annual" / "a1_risk_translation_v1.yaml"
    spec = AnnualStudySpec.from_yaml(path)
    base = AppConfig.from_yaml(spec.method_study.base_config_path)
    open_dates = [
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2017-01-03", "2025-12-31")
    ]

    plan = annual_study_plan(path, open_dates=open_dates)
    folds = build_annual_folds(spec, base, open_dates)

    assert plan["annual_id"] == "a1-risk-translation-v1-annual-2020-2025"
    assert plan["task_count"] == 24
    assert plan["registry"]["declared_current_candidate_count"] == 4
    assert [fold.year for fold in folds] == list(range(2020, 2026))


def test_a2_annual_plan_compares_only_frozen_and_discovery_pools() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "annual" / "a2_factor_expansion_v1.yaml"
    plan = annual_study_plan(path)

    assert plan["annual_id"] == "a2-factor-expansion-v1-annual-2020-2025"
    assert plan["task_count"] == 12
    assert plan["registry"]["declared_current_candidate_count"] == 2


def test_annual_runner_resumes_failed_tasks_and_rebuilds_aggregates(
    tmp_path: Path,
) -> None:
    spec = _annual_spec(tmp_path)
    folds = _folds()
    annual_root = tmp_path / "annual-studies" / spec.annual_id
    failures = {("b", 2021)}
    calls: list[tuple[str, int]] = []

    def executor(
        trial: StudyTrial,
        fold: AnnualFold,
        attempt_root: Path,
    ) -> AnnualFoldExecution:
        calls.append((trial.trial_id, fold.year))
        attempt_root.mkdir(parents=True)
        if (trial.trial_id, fold.year) in failures:
            raise RuntimeError("synthetic fold failure")
        artifact = attempt_root / "fold.txt"
        artifact.write_text(f"{trial.trial_id}-{fold.year}\n", encoding="utf-8")
        return AnnualFoldExecution(
            run_id=f"run-{trial.trial_id}-{fold.year}",
            config_hash=f"config-{trial.trial_id}-{fold.year}",
            method_hash=f"method-{trial.trial_id}",
            cost_hash="cost-v1",
            summary={"quality_passed": True, "score": 1.0},
            artifact_fingerprints={"fold.txt": sha256_file(artifact)},
        )

    def aggregator(
        trial: StudyTrial,
        task_manifests: Sequence[Mapping[str, object]],
        aggregate_root: Path,
    ) -> AnnualTrialAggregate:
        assert len(task_manifests) == 2
        aggregate_root.mkdir(parents=True, exist_ok=True)
        artifact = aggregate_root / "aggregate.txt"
        artifact.write_text(f"aggregate-{trial.trial_id}\n", encoding="utf-8")
        score = 1.0 if trial.trial_id == "a" else 0.25
        return AnnualTrialAggregate(
            summary={"quality_passed": True, "score": score},
            artifact_fingerprints={
                "aggregate.txt": sha256_file(artifact),
            },
        )

    first = AnnualStudyRunner(
        spec,
        folds=folds,
        annual_root=annual_root,
        context=_context(),
        executor=executor,
        aggregator=aggregator,
        max_workers=2,
        selected_years=[2020],
    ).run()
    assert first.status == "incomplete"
    assert first.completed_task_count == 2
    assert first.pending_task_count == 2

    calls.clear()
    second = AnnualStudyRunner(
        spec,
        folds=folds,
        annual_root=annual_root,
        context=_context(),
        executor=executor,
        aggregator=aggregator,
        max_workers=2,
    ).run()
    assert second.status == "incomplete_with_failures"
    assert second.completed_task_count == 3
    assert second.failed_task_count == 1
    assert len(calls) == 2
    assert set(calls) == {
        ("a", 2021),
        ("b", 2021),
    }

    failures.clear()
    calls.clear()
    third = AnnualStudyRunner(
        spec,
        folds=folds,
        annual_root=annual_root,
        context=_context(),
        executor=executor,
        aggregator=aggregator,
        max_workers=2,
    ).run()
    assert third.status == "completed"
    assert third.completed_task_count == 4
    assert third.failed_task_count == 0
    assert third.completed_trial_count == 2
    assert third.selected_trial_id == "a"
    assert calls == [("b", 2021)]

    retried = json.loads(
        (
            annual_root
            / "folds"
            / "2021"
            / "b"
            / "fold-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert retried["attempts"] == 2
    assert retried["artifact_root"].endswith("attempt-002")
    gates = json.loads(
        (annual_root / "publication-gates.json").read_text(encoding="utf-8")
    )
    assert gates["passed"] == ["a"]
    assert gates["failed"] == ["b"]
    selection = json.loads(
        (annual_root / "selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_trial_id"] == "a"
    tasks = pd.read_parquet(annual_root / "fold-tasks.parquet")
    assert len(tasks) == 4
    assert set(tasks["status"]) == {"completed"}
    assert tasks["data_snapshot_hash"].eq("data-hash").all()

    tampered = annual_root / retried["artifact_root"] / "fold.txt"
    tampered.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="artifact hash differs"):
        AnnualStudyRunner(
            spec,
            folds=folds,
            annual_root=annual_root,
            context=_context(),
            executor=executor,
            aggregator=aggregator,
        ).run()


def test_annual_aggregate_stitches_nav_and_audits_fold_boundaries(
    tmp_path: Path,
) -> None:
    trial = StudyTrial(trial_id="a", purpose="Candidate A.", overrides={})
    annual_root = tmp_path / "annual"
    task_manifests = [
        _write_fold_artifacts(annual_root, trial, _folds()[0], [0.0, 0.01, -0.002]),
        _write_fold_artifacts(annual_root, trial, _folds()[1], [0.0, 0.02, 0.003]),
    ]

    result = _aggregate_annual_trial(
        trial=trial,
        task_manifests=task_manifests,
        annual_root=annual_root,
        aggregate_root=annual_root / "aggregates" / "a",
        calibrator_name="robust_cross_section",
        calibrator_params={"target_scale": 0.01, "score_clip": 3.0},
    )

    assert result.summary["fold_years"] == [2020, 2021]
    assert result.summary["fold_count"] == 2
    combined = pd.read_parquet(annual_root / "aggregates" / "a" / "daily.parquet")
    assert combined["trade_date"].is_unique
    assert combined["nav"].iloc[-1] == pytest.approx(
        (1.01 * 0.998) * (1.02 * 1.003)
    )
    assert set(combined["fold_year"]) == {2020, 2021}

    invalid_fit = pd.DataFrame(
        {
            "fit_date": ["20200108"],
            "experiment_phase": ["validation"],
            "max_label_available_date": ["20200102"],
            "max_training_decision_date": ["20191231"],
        }
    )
    with pytest.raises(ConfigurationError, match="labels inside the embargo"):
        _audit_fold_fits(
            frame=invalid_fit,
            label="invalid fit",
            fold=_folds()[0].to_dict(),
        )


def test_annual_fit_audit_accepts_repeated_closed_model_checks() -> None:
    fold = _folds()[0].to_dict()
    closed = pd.DataFrame(
        {
            "fit_date": ["20200108", "20200115", "20200122"],
            "experiment_phase": ["validation"] * 3,
            "status": ["no_selected_factors"] * 3,
            "max_label_available_date": ["20191230"] * 3,
            "max_training_decision_date": ["20191231"] * 3,
        }
    )

    _audit_fold_fits(frame=closed, label="closed model", fold=fold)

    late_fit = closed.copy()
    late_fit.loc[late_fit.index[-1], "status"] = "fitted"
    with pytest.raises(ConfigurationError, match="not frozen"):
        _audit_fold_fits(frame=late_fit, label="late fit", fold=fold)


def _annual_spec(tmp_path: Path) -> AnnualStudySpec:
    (tmp_path / "base.yaml").write_text("project: {}\n", encoding="utf-8")
    (tmp_path / "methods.yaml").write_text(
        """
study:
  id: methods
  base_config: base.yaml
  max_trials: 2
selection:
  primary:
    path: score
    direction: maximize
  tie_breakers: []
  gates: []
trials:
  - id: a
    purpose: Candidate A.
    overrides: {}
  - id: b
    purpose: Candidate B.
    overrides: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "annual.yaml"
    path.write_text(
        """
annual_study:
  id: annual-test
  method_study: methods.yaml
  years: [2020, 2021]
  train_start: "20170103"
  embargo_days: 5
  max_workers: 2
publication_gates:
  - path: score
    operator: ">="
    value: 0.5
history:
  - id: old-run
    role: diagnostic
    status: completed
    included_in_selection: false
    note: Disclosed only.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return AnnualStudySpec.from_yaml(path)


def _folds() -> tuple[AnnualFold, ...]:
    return (
        AnnualFold(
            year=2020,
            train_start="20170103",
            train_end="20191231",
            embargo_start="20191231",
            last_mature_label_date="20191230",
            evaluation_start="20200108",
            evaluation_end="20201231",
            first_decision_date="20200108",
            last_decision_date="20201228",
            open_session_count=243,
            decision_count=48,
        ),
        AnnualFold(
            year=2021,
            train_start="20170103",
            train_end="20201228",
            embargo_start="20201228",
            last_mature_label_date="20201225",
            evaluation_start="20210105",
            evaluation_end="20211231",
            first_decision_date="20210105",
            last_decision_date="20211230",
            open_session_count=243,
            decision_count=49,
        ),
    )


def _context() -> StudyContext:
    return StudyContext(
        base_config_hash="base-hash",
        source_hash="source-hash",
        data_snapshot_hash="data-hash",
        data_fingerprints={"stock_bars": "bars-hash"},
        git={"commit": "abc", "dirty": False},
    )


def _write_fold_artifacts(
    annual_root: Path,
    trial: StudyTrial,
    fold: AnnualFold,
    portfolio_returns: list[float],
) -> dict[str, object]:
    attempt_root = annual_root / "folds" / fold.fold_id / trial.trial_id / "attempt-001"
    gold_root = attempt_root / "gold"
    gold_root.mkdir(parents=True)
    dates = pd.bdate_range(
        pd.Timestamp(f"{fold.year}-02-03"),
        periods=len(portfolio_returns),
    ).strftime("%Y%m%d")
    benchmark_returns = [0.0, 0.001, -0.001]
    nav = pd.Series(1.0 + pd.Series(portfolio_returns)).cumprod()
    benchmark_nav = pd.Series(1.0 + pd.Series(benchmark_returns)).cumprod()
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "nav": nav,
            "benchmark_nav": benchmark_nav,
            "active_nav": nav / benchmark_nav,
            "portfolio_return": portfolio_returns,
            "benchmark_return": benchmark_returns,
            "active_return": (
                (1.0 + pd.Series(portfolio_returns))
                / (1.0 + pd.Series(benchmark_returns))
                - 1.0
            ),
            "turnover": [0.0, 0.1, 0.0],
            "transaction_cost": [0.0, 0.0001, 0.0],
        }
    )
    signals = pd.DataFrame(
        {
            "decision_date": [fold.first_decision_date] * 2,
            "instrument": ["A", "B"],
            "expected_return": [-0.01, 0.01],
        }
    )
    labels = pd.DataFrame(
        {
            "decision_date": [fold.first_decision_date] * 2,
            "instrument": ["A", "B"],
            "forward_active_return": [-0.005, 0.005],
        }
    )
    model_fits = pd.DataFrame(
        {
            "fit_date": [fold.first_decision_date],
            "experiment_phase": ["validation"],
            "max_label_available_date": [fold.last_mature_label_date],
            "max_training_decision_date": [fold.train_end],
            "model": ["synthetic"],
            "status": ["fitted"],
            "model_parameters": ["{}"],
        }
    )
    calibration_fits = pd.DataFrame(
        {
            "fit_date": [fold.first_decision_date],
            "experiment_phase": ["validation"],
            "max_label_available_date": [fold.last_mature_label_date],
            "max_training_decision_date": [fold.train_end],
            "calibrator": ["synthetic"],
            "status": ["fitted"],
            "parameters": ["{}"],
        }
    )
    trades = pd.DataFrame(
        columns=[
            "trade_date",
            "status",
            "requested_value",
            "gross_value",
            "linear_cost",
            "stamp_duty",
            "impact_cost",
        ]
    )
    daily.to_parquet(attempt_root / "daily.parquet", index=False)
    trades.to_parquet(attempt_root / "trades.parquet", index=False)
    signals.to_parquet(attempt_root / "evaluation-signals.parquet", index=False)
    labels.to_parquet(gold_root / "labels.parquet", index=False)
    model_fits.to_parquet(attempt_root / "model-fits.parquet", index=False)
    calibration_fits.to_parquet(
        attempt_root / "calibration-fits.parquet",
        index=False,
    )
    pd.DataFrame().to_parquet(
        attempt_root / "constraint-audits.parquet",
        index=False,
    )
    summary = {
        "quality_passed": True,
        "valid_expected_returns": 2,
        "optimizer_attempts": 1,
        "optimizer_solved": 1,
    }
    (attempt_root / "workflow-summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (attempt_root / "run-manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"run-{trial.trial_id}-{fold.year}",
                "config_hash": f"config-{trial.trial_id}-{fold.year}",
                "annual_fold": {
                    "annual_id": "annual-test",
                    "fold_year": fold.year,
                },
                "experiment": {
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": fold.evaluation_start,
                    "validation_end": fold.evaluation_end,
                    "data_snapshot_hash": "data-hash",
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "annual_id": "annual-test",
        "trial": trial.to_dict(),
        "fold": fold.to_dict(),
        "artifact_root": attempt_root.relative_to(annual_root).as_posix(),
        "run_id": f"run-{trial.trial_id}-{fold.year}",
        "resolved_config_hash": f"config-{trial.trial_id}-{fold.year}",
        "data_snapshot_hash": "data-hash",
        "summary": summary,
    }
