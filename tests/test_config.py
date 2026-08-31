from dataclasses import asdict, replace
from pathlib import Path

import pytest

from csi500_alpha.config import AppConfig
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.study import StudySpec, resolve_trial_config


def test_smoke_config_loads_from_project_root() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "smoke.yaml")
    assert config.paths.root == root
    assert config.dates.raw_start < config.dates.backtest_start < config.dates.end
    assert config.source.index_code == "000905.SH"
    assert config.source.total_return_index_code == "H00905.CSI"
    assert config.research.top_n == 30


def test_framework_config_keeps_feature_and_portfolio_periods_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "framework_ridge_smoke.yaml")
    assert config.workflow.feature_start < config.workflow.portfolio_start
    assert config.workflow.selector.name == "coverage_correlation"
    assert config.workflow.model.name == "ridge"
    assert config.workflow.calibrator.name == "robust_cross_section"


def test_full_config_declares_frozen_research_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "full.yaml")

    assert config.experiment.stage == "validation"
    assert config.experiment.train_end < config.experiment.validation_start
    assert config.experiment.validation_end < config.experiment.test_start
    assert config.experiment.embargo_days == config.features.label_horizon
    assert config.workflow.feature_provider.name == "builtin_daily"
    assert config.source.calls_per_minute_limit == 200
    assert config.source.effective_min_request_interval_seconds >= 0.31
    assert config.features.industry_coverage_threshold == 0.90
    assert config.experiment.allow_frozen_test is False


def test_extended_validation_can_reserve_future_holdout() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "extended_validation_2023_2025.yaml")

    assert config.experiment.stage == "validation"
    assert config.experiment.validation_end == config.dates.end
    assert config.experiment.test_start > config.dates.end
    assert config.experiment.allow_frozen_test is False
    assert config.optimizer.tracking_error_cap == 0.05
    assert config.optimizer.constraint_materiality_tolerance == 0.0001
    assert config.risk.model == "ledoit_wolf"
    assert config.risk.beta_model == "feature_60"


def test_a1_study_is_a_fixed_signal_two_by_two_risk_design() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = StudySpec.from_yaml(
        root / "configs" / "studies" / "a1_risk_translation_v1.yaml"
    )
    base = AppConfig.from_yaml(spec.base_config_path)
    resolved = [
        resolve_trial_config(base, study_id=spec.study_id, trial=trial)
        for trial in spec.trials
    ]

    assert [trial.trial_id for trial in spec.trials] == [
        "a0_current",
        "a1_beta_ewma",
        "a1_factor_covariance",
        "a1_joint",
    ]
    assert {(config.risk.model, config.risk.beta_model) for config in resolved} == {
        ("ledoit_wolf", "feature_60"),
        ("ledoit_wolf", "ewma_shrunk"),
        ("factor_ewma", "feature_60"),
        ("factor_ewma", "ewma_shrunk"),
    }
    reference = resolved[0].workflow
    assert all(config.workflow == reference for config in resolved[1:])


def test_final_holdout_config_exactly_freezes_selected_d1_method() -> None:
    root = Path(__file__).resolve().parents[1]
    final = AppConfig.from_yaml(root / "configs" / "final_holdout_2026.yaml")
    study = StudySpec.from_yaml(root / "configs" / "studies" / "adaptive_bridge.yaml")
    base = AppConfig.from_yaml(study.base_config_path)
    selected_trial = next(
        trial for trial in study.trials if trial.trial_id == "d1_ic_full_calibrated"
    )
    selected = resolve_trial_config(
        base,
        study_id=study.study_id,
        trial=selected_trial,
    )

    assert asdict(final.research) == asdict(selected.research)
    assert asdict(final.risk) == asdict(selected.risk)
    assert asdict(final.optimizer) == asdict(selected.optimizer)
    assert asdict(final.features) == asdict(selected.features)
    assert asdict(final.workflow) == asdict(selected.workflow)
    final_download = asdict(final.download)
    selected_download = asdict(selected.download)
    assert final_download.pop("reference_cache_tag") == "final-20260630"
    assert final_download.pop("eligibility_refresh_start") == "20260101"
    selected_download.pop("reference_cache_tag")
    selected_download.pop("eligibility_refresh_start")
    assert final_download == selected_download
    assert final.paths.dataset == "final_2026"
    assert final.dates.end == final.experiment.test_end == "20260630"
    assert final.experiment.stage == "validation"
    assert final.experiment.allow_frozen_test is True
    assert final.experiment.protocol_id == "csi500-final-holdout-2026-v1"


def test_download_pilot_spans_industry_transition() -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(root / "configs" / "pilot_2020_2022.yaml")

    assert config.dates.raw_start < config.features.industry_transition_date
    assert config.features.industry_transition_date < config.dates.end
    assert config.download.industry_taxonomies == ("SW2014", "SW2021")
    assert config.download.supplement_industry_by_instrument is True
    assert config.experiment.stage == "walk_forward"


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    candidate = tmp_path / "unknown.yaml"
    candidate.write_text(f"{text}\nunknown_section: {{}}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unknown configuration keys"):
        AppConfig.from_yaml(candidate)


def test_snapshot_cache_configuration_is_validated() -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "smoke.yaml")

    with pytest.raises(ConfigurationError, match="reference_cache_tag"):
        replace(
            base,
            download=replace(base.download, reference_cache_tag="unsafe/tag"),
        ).validate()
    with pytest.raises(ConfigurationError, match="eligibility_refresh_start"):
        replace(
            base,
            download=replace(base.download, eligibility_refresh_start="19990101"),
        ).validate()
    with pytest.raises(ConfigurationError, match="must differ"):
        replace(
            base,
            source=replace(
                base.source,
                total_return_index_code=base.source.index_code,
            ),
        ).validate()
