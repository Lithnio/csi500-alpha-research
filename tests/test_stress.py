from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from csi500_alpha.config import AppConfig
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.pipeline import plan_portfolio_stress
from csi500_alpha.stress import (
    StressContext,
    StressExecution,
    StressRunner,
    StressScenario,
    StressSpec,
    resolve_stress_config,
)


def test_repository_stress_plan_is_bounded_and_resolves_one_way_changes() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "stress" / "core_cost_capacity.yaml"
    spec = StressSpec.from_yaml(path)
    plan = plan_portfolio_stress(path)
    base = AppConfig.from_yaml(root / "configs" / "full.yaml")

    assert spec.stress_id == "core-cost-capacity-v1"
    assert plan["scenario_count"] == 7
    assert plan["source_study_id"] == "core-baselines-v3"
    assert [scenario.scenario_id for scenario in spec.scenarios] == [
        "baseline",
        "cost_0_5x",
        "cost_2_0x",
        "aum_50m",
        "aum_300m",
        "participation_10pct",
        "participation_20pct",
    ]

    doubled = resolve_stress_config(
        base,
        next(scenario for scenario in spec.scenarios if scenario.scenario_id == "cost_2_0x"),
    )
    assert doubled.research.linear_cost_bps == base.research.linear_cost_bps * 2
    assert doubled.research.stamp_duty_after == base.research.stamp_duty_after * 2
    assert (
        doubled.optimizer.impact_bps_at_max_participation
        == base.optimizer.impact_bps_at_max_participation * 2
    )
    assert doubled.optimizer.portfolio_aum_cny == base.optimizer.portfolio_aum_cny

    larger_aum = resolve_stress_config(
        base,
        next(scenario for scenario in spec.scenarios if scenario.scenario_id == "aum_300m"),
    )
    assert larger_aum.optimizer.portfolio_aum_cny == 300_000_000
    assert (
        larger_aum.optimizer.max_adv_participation
        == base.optimizer.max_adv_participation
    )


def test_stress_runner_records_failures_resumes_and_checks_baseline(tmp_path: Path) -> None:
    spec = _stress_spec(tmp_path)
    context = _context()
    calls: list[str] = []

    def first_executor(
        scenario: StressScenario,
        attempt_root: Path,
    ) -> StressExecution:
        calls.append(scenario.scenario_id)
        attempt_root.mkdir(parents=True)
        if scenario.scenario_id == "cost_2x":
            raise RuntimeError("synthetic stress failure")
        return StressExecution(summary=_summary(scenario.scenario_id, parity=True))

    stress_root = tmp_path / "stress-output"
    first = StressRunner(
        spec,
        stress_root=stress_root,
        context=context,
        executor=first_executor,
    ).run()

    assert calls == ["baseline", "cost_2x", "aum_300m"]
    assert first.status == "completed_with_failures"
    assert first.completed_count == 2
    assert first.failed_count == 1
    assert first.baseline_parity_passed
    failed = json.loads(
        (stress_root / "scenarios" / "cost_2x" / "scenario-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed["attempts"] == 1
    assert failed["error"]["type"] == "RuntimeError"

    calls.clear()

    def retry_executor(
        scenario: StressScenario,
        attempt_root: Path,
    ) -> StressExecution:
        calls.append(scenario.scenario_id)
        attempt_root.mkdir(parents=True)
        return StressExecution(summary=_summary(scenario.scenario_id, parity=None))

    second = StressRunner(
        spec,
        stress_root=stress_root,
        context=context,
        executor=retry_executor,
    ).run()

    assert calls == ["cost_2x"]
    assert second.status == "completed"
    assert second.skipped_count == 2
    assert second.baseline_parity_passed
    results = pd.read_parquet(stress_root / "stress-results.parquet")
    assert set(results["scenario_id"]) == {"baseline", "cost_2x", "aum_300m"}
    assert "metric__metrics__information_ratio" in results
    comparison = json.loads(
        (stress_root / "stress-summary.json").read_text(encoding="utf-8")
    )
    assert comparison["completed_scenarios"] == ["aum_300m", "baseline", "cost_2x"]
    assert comparison["baseline_parity_passed"] is True

    with pytest.raises(ConfigurationError, match="Existing stress identity differs"):
        StressRunner(
            spec,
            stress_root=stress_root,
            context=replace(context, data_snapshot_hash="changed"),
            executor=retry_executor,
        ).run()


def test_stress_spec_requires_an_unmodified_baseline(tmp_path: Path) -> None:
    path = _stress_config_path(tmp_path, baseline_cost=2.0)

    with pytest.raises(ConfigurationError, match="unmodified scenario"):
        StressSpec.from_yaml(path)


def _stress_spec(tmp_path: Path) -> StressSpec:
    return StressSpec.from_yaml(_stress_config_path(tmp_path, baseline_cost=1.0))


def _stress_config_path(tmp_path: Path, *, baseline_cost: float) -> Path:
    (tmp_path / "source-study.yaml").write_text("study: {}\n", encoding="utf-8")
    path = tmp_path / "stress.yaml"
    path.write_text(
        f"""
stress:
  id: synthetic-stress
  source_study: source-study.yaml
  max_scenarios: 3
  baseline_parity_tolerance: 1.0e-8
scenarios:
  - id: baseline
    purpose: Reproduce source.
    cost_multiplier: {baseline_cost}
  - id: cost_2x
    purpose: Double costs.
    cost_multiplier: 2.0
  - id: aum_300m
    purpose: Increase capacity demand.
    portfolio_aum_cny: 300000000
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _context() -> StressContext:
    return StressContext(
        source_study_id="source-study",
        source_trial_id="selected",
        source_resolved_config_hash="config-hash",
        source_run_manifest_hash="manifest-hash",
        source_hash="source-hash",
        data_snapshot_hash="data-hash",
    )


def _summary(scenario_id: str, *, parity: bool | None) -> dict[str, object]:
    information_ratio = 1.0 if scenario_id == "baseline" else 0.8
    return {
        "metrics": {
            "information_ratio": information_ratio,
            "max_drawdown": -0.1,
            "average_turnover": 0.2,
            "annualized_active_return": 0.03,
            "transaction_cost": 0.001,
        },
        "evaluation": {
            "yearly": {
                "minimum_information_ratio": 0.2,
                "positive_active_year_fraction": 1.0,
            },
            "execution": {
                "notional_fill_ratio": 0.9,
                "cost_bps_of_executed_notional": 8.0,
            },
        },
        "optimizer_solve_rate": 1.0,
        "source_metric_max_abs_difference": 0.0,
        "source_metric_parity_passed": parity,
    }
