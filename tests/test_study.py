from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.config import AppConfig
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.pipeline import plan_study as plan_repository_study
from csi500_alpha.study import (
    StudyContext,
    StudyRunner,
    StudySpec,
    StudyTrial,
    TrialExecution,
    resolve_trial_config,
    study_plan,
)


def test_repository_study_plan_and_trial_overrides_are_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    study_path = root / "configs" / "studies" / "core_baselines.yaml"
    spec = StudySpec.from_yaml(study_path)
    base = AppConfig.from_yaml(spec.base_config_path)

    assert study_plan(study_path)["trial_count"] == 9
    planned = plan_repository_study(study_path)
    assert len(planned["resolved_trials"]) == 9
    assert all(item["config_hash"] for item in planned["resolved_trials"])
    assert spec.base_config_path == root / "configs" / "full.yaml"
    ridge = next(
        trial for trial in spec.trials if trial.trial_id == "b2_stability_ridge"
    )
    resolved = resolve_trial_config(base, study_id=spec.study_id, trial=ridge)

    assert base.workflow.model.name == "direction_equal_weight"
    assert resolved.workflow.model.name == "ridge"
    assert resolved.workflow.selector.name == "stability_cost"
    assert resolved.paths == base.paths
    assert resolved.dates == base.dates
    assert resolved.experiment.train_start == base.experiment.train_start
    assert resolved.experiment.protocol_id.endswith("b2_stability_ridge")
    expected_ablation = {
        "c0_ic_raw": (False, 0.0, 0.0, 0.0),
        "c1_ic_uncertainty": (True, 0.0, 0.0, 0.0),
        "c2_ic_correlation": (True, 0.005, 0.0, 0.0),
        "c3_ic_cost": (True, 0.005, 0.01, 0.0),
        "c4_ic_full": (True, 0.005, 0.01, 0.01),
    }
    for trial_id, expected in expected_ablation.items():
        trial = next(trial for trial in spec.trials if trial.trial_id == trial_id)
        resolved_p3 = resolve_trial_config(base, study_id=spec.study_id, trial=trial)
        params = resolved_p3.workflow.model.params
        assert resolved_p3.workflow.model.name == "ic_shrinkage"
        assert (
            params["shrinkage_enabled"],
            params["correlation_penalty"],
            params["cost_penalty"],
            params["weight_turnover_penalty"],
        ) == expected

    invalid = StudyTrial(
        trial_id="invalid",
        purpose="Attempt to move a sample boundary.",
        overrides={"experiment": {"validation_end": "20251231"}},
    )
    with pytest.raises(ConfigurationError, match="Unknown trial invalid overrides keys"):
        resolve_trial_config(base, study_id=spec.study_id, trial=invalid)


def test_study_runner_records_failure_resumes_and_reselects(tmp_path: Path) -> None:
    spec = _study_spec(tmp_path)
    context = _context()
    calls: list[str] = []

    def first_executor(trial: StudyTrial, attempt_root: Path) -> TrialExecution:
        calls.append(trial.trial_id)
        attempt_root.mkdir(parents=True)
        if trial.trial_id == "b":
            raise RuntimeError("synthetic trial failure")
        return TrialExecution(
            run_id=f"run-{trial.trial_id}",
            config_hash=f"config-{trial.trial_id}",
            summary={
                "quality_passed": trial.trial_id != "c",
                "metrics": {
                    "information_ratio": 1.0 if trial.trial_id == "a" else 2.0,
                    "average_turnover": 0.2,
                },
            },
        )

    study_root = tmp_path / "studies" / spec.study_id
    first = StudyRunner(
        spec,
        study_root=study_root,
        context=context,
        executor=first_executor,
    ).run()

    assert calls == ["a", "b", "c"]
    assert first.status == "completed_with_failures"
    assert first.selected_trial_id == "a"
    assert first.completed_count == 2
    assert first.failed_count == 1
    failed = json.loads(
        (study_root / "trials" / "b" / "trial-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed["status"] == "failed"
    assert failed["attempts"] == 1
    assert failed["error"]["type"] == "RuntimeError"
    assert "synthetic trial failure" in failed["error"]["message"]
    selection = json.loads(
        (study_root / "selection.json").read_text(encoding="utf-8")
    )
    rejected = {
        item["trial_id"]: item["reasons"] for item in selection["rejected_trials"]
    }
    assert any("trial_status=failed" in reason for reason in rejected["b"])
    assert any("gate_failed=quality_passed" in reason for reason in rejected["c"])

    calls.clear()

    def retry_executor(trial: StudyTrial, attempt_root: Path) -> TrialExecution:
        calls.append(trial.trial_id)
        attempt_root.mkdir(parents=True)
        return TrialExecution(
            run_id=f"retry-{trial.trial_id}",
            config_hash=f"config-{trial.trial_id}",
            summary={
                "quality_passed": True,
                "metrics": {
                    "information_ratio": 3.0,
                    "average_turnover": 0.1,
                },
            },
        )

    second = StudyRunner(
        spec,
        study_root=study_root,
        context=context,
        executor=retry_executor,
    ).run()

    assert calls == ["b"]
    assert second.status == "completed"
    assert second.selected_trial_id == "b"
    assert second.skipped_count == 2
    retried = json.loads(
        (study_root / "trials" / "b" / "trial-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert retried["attempts"] == 2
    assert retried["artifact_root"].endswith("attempt-002")
    assert (study_root / "trials" / "b" / "attempt-001").is_dir()
    assert (study_root / "trials" / "b" / "attempt-002").is_dir()
    trials = pd.read_parquet(study_root / "trials.parquet")
    assert set(trials["trial_id"]) == {"a", "b", "c"}
    assert set(trials["source_hash"]) == {context.source_hash}
    assert trials["artifact_root"].nunique() == 3
    assert set(trials["resolved_config_hash"]) == {"config-a", "config-b", "config-c"}
    assert "metric__metrics__information_ratio" in trials
    reselection = json.loads(
        (study_root / "selection.json").read_text(encoding="utf-8")
    )
    assert reselection["ranking"][0]["trial_id"] == "b"
    assert reselection["ranking"][0]["metrics"]["metrics.information_ratio"] == 3.0


def test_study_runner_rejects_identity_drift(tmp_path: Path) -> None:
    spec = _study_spec(tmp_path)
    study_root = tmp_path / "studies" / spec.study_id

    def executor(trial: StudyTrial, attempt_root: Path) -> TrialExecution:
        attempt_root.mkdir(parents=True)
        return TrialExecution(
            run_id=trial.trial_id,
            config_hash=f"config-{trial.trial_id}",
            summary={
                "quality_passed": True,
                "metrics": {"information_ratio": 1.0, "average_turnover": 0.1},
            },
        )

    StudyRunner(
        spec,
        study_root=study_root,
        context=_context(),
        executor=executor,
    ).run()
    changed = replace(_context(), source_hash="changed-source")
    with pytest.raises(ConfigurationError, match="Existing study identity differs"):
        StudyRunner(
            spec,
            study_root=study_root,
            context=changed,
            executor=executor,
        ).run()


def test_primary_tolerance_prefers_robust_tie_breaker_inside_equivalence_band(
    tmp_path: Path,
) -> None:
    (tmp_path / "base.yaml").write_text("project: {}\n", encoding="utf-8")
    path = tmp_path / "tolerant-study.yaml"
    path.write_text(
        """
study:
  id: tolerant-study
  base_config: base.yaml
  max_trials: 2
selection:
  primary_tolerance: 0.10
  gates:
    - path: quality_passed
      operator: "=="
      value: true
  primary:
    path: metrics.information_ratio
    direction: maximize
  tie_breakers:
    - path: metrics.average_turnover
      direction: minimize
trials:
  - id: simpler
    purpose: Lower turnover candidate.
    overrides: {}
  - id: peak
    purpose: Slightly higher point estimate.
    overrides: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = StudySpec.from_yaml(path)

    def executor(trial: StudyTrial, attempt_root: Path) -> TrialExecution:
        attempt_root.mkdir(parents=True)
        is_peak = trial.trial_id == "peak"
        return TrialExecution(
            run_id=trial.trial_id,
            config_hash=f"config-{trial.trial_id}",
            summary={
                "quality_passed": True,
                "metrics": {
                    "information_ratio": 1.05 if is_peak else 1.00,
                    "average_turnover": 0.30 if is_peak else 0.05,
                },
            },
        )

    study_root = tmp_path / "studies" / spec.study_id
    result = StudyRunner(
        spec,
        study_root=study_root,
        context=_context(),
        executor=executor,
    ).run()

    assert result.selected_trial_id == "simpler"
    selection = json.loads(
        (study_root / "selection.json").read_text(encoding="utf-8")
    )
    assert selection["shortlisted_trial_ids"] == ["simpler", "peak"]
    assert selection["ranking"][0]["within_primary_equivalence_band"] is True
    assert selection["ranking"][1]["primary_gap_from_best"] == pytest.approx(0.0)


def _study_spec(tmp_path: Path) -> StudySpec:
    (tmp_path / "base.yaml").write_text("project: {}\n", encoding="utf-8")
    path = tmp_path / "study.yaml"
    path.write_text(
        """
study:
  id: synthetic-study
  base_config: base.yaml
  max_trials: 3
selection:
  gates:
    - path: quality_passed
      operator: "=="
      value: true
  primary:
    path: metrics.information_ratio
    direction: maximize
  tie_breakers:
    - path: metrics.average_turnover
      direction: minimize
trials:
  - id: a
    purpose: First candidate.
    overrides: {}
  - id: b
    purpose: Retry candidate.
    overrides: {}
  - id: c
    purpose: Gate failure candidate.
    overrides: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return StudySpec.from_yaml(path)


def _context() -> StudyContext:
    return StudyContext(
        base_config_hash="base-hash",
        source_hash="source-hash",
        data_snapshot_hash="data-hash",
        data_fingerprints={"stock_bars": "bars-hash"},
        git={"commit": "abc", "dirty": False},
    )
