from dataclasses import replace
from pathlib import Path

import pytest

from csi500_alpha.config import AppConfig
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.pipeline import (
    _bind_experiment_data_snapshot,
    _complete_experiment_run,
    _prepare_experiment_run,
    _require_final_protocol_authorization,
)


def test_validation_lock_gates_and_seals_frozen_test(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "full.yaml")
    config = replace(
        base,
        paths=replace(base.paths, run_root=tmp_path / "runs"),
        experiment=replace(base.experiment, allow_frozen_test=True),
    )

    fingerprints = {"calendar": "calendar-hash", "stock_bars": "bars-hash"}
    validation = _bind_experiment_data_snapshot(
        config,
        _prepare_experiment_run(config),
        fingerprints,
    )
    _complete_experiment_run(config, validation, "validation-run")

    frozen = replace(
        config,
        experiment=replace(config.experiment, stage="frozen_test"),
    )
    frozen_context = _bind_experiment_data_snapshot(
        frozen,
        _prepare_experiment_run(frozen),
        fingerprints,
    )
    assert frozen_context["validation_run_id"] == "validation-run"

    with pytest.raises(ConfigurationError, match="differs from the validated snapshot"):
        _bind_experiment_data_snapshot(
            frozen,
            _prepare_experiment_run(frozen),
            {**fingerprints, "stock_bars": "changed-bars-hash"},
        )

    changed = replace(
        frozen,
        optimizer=replace(frozen.optimizer, risk_aversion=99.0),
    )
    with pytest.raises(ConfigurationError, match="differ from the validated protocol"):
        _prepare_experiment_run(changed)

    _complete_experiment_run(frozen, frozen_context, "test-run")
    with pytest.raises(ConfigurationError, match="already has a completed frozen test"):
        _prepare_experiment_run(frozen)


def test_experiment_lock_cannot_complete_without_data_snapshot(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "full.yaml")
    config = replace(
        base,
        paths=replace(base.paths, run_root=tmp_path / "runs"),
    )

    with pytest.raises(ConfigurationError, match="requires a bound Silver data snapshot"):
        _complete_experiment_run(
            config,
            _prepare_experiment_run(config),
            "invalid-validation-run",
        )


def test_config_can_explicitly_disable_frozen_test(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "full.yaml")
    config = replace(
        base,
        paths=replace(base.paths, run_root=tmp_path / "runs"),
        experiment=replace(base.experiment, stage="frozen_test"),
    )

    with pytest.raises(ConfigurationError, match="frozen_test is disabled"):
        _prepare_experiment_run(config)


def test_frozen_test_requires_explicit_confirmation_and_clean_git() -> None:
    root = Path(__file__).resolve().parents[1]
    base = AppConfig.from_yaml(root / "configs" / "final_holdout_2026.yaml")
    frozen = replace(
        base,
        experiment=replace(base.experiment, stage="frozen_test"),
    )
    clean = {"inside_work_tree": True, "dirty": False, "commit": "abc123"}

    with pytest.raises(ConfigurationError, match="clean Git worktree"):
        _require_final_protocol_authorization(
            base,
            confirmed=False,
            git_state={"inside_work_tree": True, "dirty": True, "commit": "abc123"},
        )
    _require_final_protocol_authorization(
        base,
        confirmed=False,
        git_state=clean,
    )

    with pytest.raises(ConfigurationError, match="confirm-final-holdout"):
        _require_final_protocol_authorization(
            frozen,
            confirmed=False,
            git_state=clean,
        )
    with pytest.raises(ConfigurationError, match="clean Git worktree"):
        _require_final_protocol_authorization(
            frozen,
            confirmed=True,
            git_state={"inside_work_tree": True, "dirty": True, "commit": "abc123"},
        )
    with pytest.raises(ConfigurationError, match="recorded commit"):
        _require_final_protocol_authorization(
            frozen,
            confirmed=True,
            git_state={"inside_work_tree": True, "dirty": False, "commit": None},
        )

    _require_final_protocol_authorization(
        frozen,
        confirmed=True,
        git_state=clean,
    )
