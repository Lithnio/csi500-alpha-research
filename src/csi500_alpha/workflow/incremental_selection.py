from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.workflow.contracts import SelectionResult
from csi500_alpha.workflow.selection import StabilityCostSelector, StabilityCostSettings


@dataclass(frozen=True)
class IncrementalAdmissionSettings:
    """Separate a frozen core selection domain from incremental candidates."""

    candidate_factors: tuple[str, ...]
    core: StabilityCostSettings
    additions: StabilityCostSettings
    min_core_fraction: float = 0.50
    min_residual_variance_ratio: float = 0.05

    def validate(self) -> None:
        self.core.validate()
        self.additions.validate()
        if not self.candidate_factors:
            raise ConfigurationError(
                "Incremental admission candidate_factors cannot be empty"
            )
        if len(set(self.candidate_factors)) != len(self.candidate_factors):
            raise ConfigurationError(
                "Incremental admission candidate_factors cannot contain duplicates"
            )
        if not 0 < self.min_core_fraction <= 1:
            raise ConfigurationError(
                "Incremental admission min_core_fraction must be in (0, 1]"
            )
        if not 0 <= self.min_residual_variance_ratio <= 1:
            raise ConfigurationError(
                "Incremental admission min_residual_variance_ratio must be in [0, 1]"
            )


class IncrementalStabilityCostSelector:
    """Admit candidates without changing the core factor testing universe.

    The core and candidate pools are evaluated by independent stability/cost
    selectors. Candidate scores are direction-aligned and residualized against
    the selected core composite within each historical cross-section before the
    candidate-only multiple-testing correction is applied.
    """

    name = "incremental_stability_cost"

    def __init__(
        self,
        *,
        directions: Mapping[str, int],
        families: Mapping[str, str],
        settings: IncrementalAdmissionSettings,
        label_column: str = "forward_active_return",
    ) -> None:
        settings.validate()
        invalid_directions = sorted(
            name for name, value in directions.items() if value not in {-1, 1}
        )
        if invalid_directions:
            raise ConfigurationError(
                "Incremental admission directions must be -1 or 1: "
                f"{invalid_directions}"
            )
        self.directions = {
            str(name): int(value) for name, value in directions.items()
        }
        self.families = {str(name): str(value) for name, value in families.items()}
        self.settings = settings
        self.label_column = label_column
        self._core_selector = StabilityCostSelector(
            directions=self.directions,
            families=self.families,
            settings=settings.core,
            label_column=label_column,
        )

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        provided = tuple(dict.fromkeys(str(name) for name in candidates))
        additions = self.settings.candidate_factors
        missing_additions = sorted(set(additions).difference(provided))
        if missing_additions:
            raise ConfigurationError(
                "Incremental admission candidates are absent from the workflow pool: "
                f"{missing_additions}"
            )
        core = tuple(name for name in provided if name not in set(additions))
        if not core:
            raise ConfigurationError(
                "Incremental admission requires at least one core factor"
            )
        missing_directions = sorted(set(provided).difference(self.directions))
        missing_families = sorted(set(provided).difference(self.families))
        if missing_directions:
            raise ConfigurationError(
                f"Incremental admission lacks directions: {missing_directions}"
            )
        if missing_families:
            raise ConfigurationError(
                f"Incremental admission lacks families: {missing_families}"
            )

        core_result = self._core_selector.select(
            training,
            core,
            as_of_date=as_of_date,
        )
        core_diagnostics = dict(core_result.diagnostics)
        if not core_result.factor_names:
            factor_diagnostics = {
                factor: {
                    **dict(details),
                    "admission_role": "core",
                }
                for factor, details in core_diagnostics.get(
                    "factor_diagnostics",
                    {},
                ).items()
            }
            factor_diagnostics.update(
                {
                    factor: {
                        "admission_role": "candidate",
                        "selected": False,
                        "status": "not_evaluated",
                        "reasons": ["core_selection_closed"],
                    }
                    for factor in additions
                }
            )
            return SelectionResult(
                factor_names=(),
                diagnostics=self._diagnostics(
                    provided=provided,
                    core=core,
                    core_result=core_result,
                    addition_result=None,
                    residualization={},
                    selected=(),
                    factor_diagnostics=factor_diagnostics,
                    status="core_selection_closed",
                    as_of_date=as_of_date,
                ),
            )

        residual_training, residualization = self._residualize_candidates(
            training,
            additions,
            core_result.factor_names,
            as_of_date=as_of_date,
        )
        # Residuals depend on the currently selected core. Rebuilding this small
        # selector avoids reusing daily statistics computed under an older core.
        addition_selector = StabilityCostSelector(
            directions={factor: 1 for factor in additions},
            families=self.families,
            settings=self.settings.additions,
            label_column=self.label_column,
        )
        addition_result = addition_selector.select(
            residual_training,
            additions,
            as_of_date=as_of_date,
        )
        admitted = tuple(
            factor
            for factor in addition_result.factor_names
            if residualization[factor]["passed_variance_gate"]
        )
        selected = (*core_result.factor_names, *admitted)
        factor_diagnostics = {
            factor: {
                **dict(details),
                "admission_role": "core",
            }
            for factor, details in core_diagnostics.get(
                "factor_diagnostics",
                {},
            ).items()
        }
        for factor, details in addition_result.diagnostics.get(
            "factor_diagnostics",
            {},
        ).items():
            reasons = list(details.get("reasons", ()))
            if not residualization[factor]["passed_variance_gate"]:
                reasons.append("residual_variance_below_minimum")
            is_selected = factor in admitted
            factor_diagnostics[factor] = {
                **dict(details),
                "admission_role": "candidate",
                "selected": is_selected,
                "status": "selected" if is_selected else "rejected",
                "reasons": list(dict.fromkeys(reasons)),
                "residualization": residualization[factor],
            }
        return SelectionResult(
            factor_names=selected,
            diagnostics=self._diagnostics(
                provided=provided,
                core=core,
                core_result=core_result,
                addition_result=addition_result,
                residualization=residualization,
                selected=selected,
                factor_diagnostics=factor_diagnostics,
                status="selected",
                as_of_date=as_of_date,
            ),
        )

    def _residualize_candidates(
        self,
        training: pd.DataFrame,
        additions: Sequence[str],
        selected_core: Sequence[str],
        *,
        as_of_date: str,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        required = {
            "decision_date",
            "label_available_date",
            self.label_column,
            *(f"{factor}__z" for factor in (*selected_core, *additions)),
        }
        missing = sorted(required.difference(training.columns))
        if missing:
            raise ConfigurationError(
                f"Incremental admission training panel lacks columns: {missing}"
            )
        residualized = training.copy()
        decision_date = training["decision_date"].astype(str)
        label_available = training["label_available_date"]
        visible = decision_date.lt(str(as_of_date))
        visible &= label_available.notna()
        visible &= label_available.fillna("").astype(str).lt(str(as_of_date))
        visible &= pd.to_numeric(
            training[self.label_column],
            errors="coerce",
        ).notna()
        visible_dates = sorted(decision_date.loc[visible].unique())
        recent_dates = set(
            visible_dates[-self.settings.additions.lookback_dates :]
        )
        visible &= decision_date.isin(recent_dates)
        visible_training = training.loc[visible]
        core_matrix = pd.DataFrame(index=visible_training.index)
        for factor in selected_core:
            core_matrix[factor] = (
                pd.to_numeric(
                    visible_training[f"{factor}__z"],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                * self.directions[factor]
            )
        minimum_core = max(
            1,
            int(np.ceil(len(selected_core) * self.settings.min_core_fraction)),
        )
        core_composite = core_matrix.mean(axis=1, skipna=True).where(
            core_matrix.notna().sum(axis=1).ge(minimum_core)
        )
        diagnostics: dict[str, dict[str, Any]] = {}
        grouped_indices = visible_training.groupby(
            "decision_date",
            sort=True,
        ).groups
        for factor in additions:
            output = pd.Series(np.nan, index=training.index, dtype=float)
            directed_candidate = (
                pd.to_numeric(
                    visible_training[f"{factor}__z"],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                * self.directions[factor]
            )
            raw_sum_squares = 0.0
            residual_sum_squares = 0.0
            evaluated_dates = 0
            evaluated_rows = 0
            for indices in grouped_indices.values():
                cross_section = pd.DataFrame(
                    {
                        "core": core_composite.loc[indices],
                        "candidate": directed_candidate.loc[indices],
                    }
                ).dropna()
                if (
                    len(cross_section) < self.settings.additions.min_cross_section
                    or cross_section["candidate"].nunique() < 2
                ):
                    continue
                target = cross_section["candidate"].to_numpy(dtype=float)
                centered = target - float(np.mean(target))
                raw_variance = float(centered @ centered)
                if raw_variance <= 1e-16:
                    continue
                design = np.column_stack(
                    [
                        np.ones(len(cross_section), dtype=float),
                        cross_section["core"].to_numpy(dtype=float),
                    ]
                )
                coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
                values = target - design @ coefficients
                output.loc[cross_section.index] = values
                raw_sum_squares += raw_variance
                residual_sum_squares += float(values @ values)
                evaluated_dates += 1
                evaluated_rows += len(cross_section)
            ratio = (
                residual_sum_squares / raw_sum_squares
                if raw_sum_squares > 0
                else np.nan
            )
            passed = bool(
                np.isfinite(ratio)
                and ratio >= self.settings.min_residual_variance_ratio
            )
            if not passed:
                output[:] = np.nan
            residualized[f"{factor}__z"] = output
            diagnostics[factor] = {
                "selected_core_factors": list(selected_core),
                "evaluated_dates": evaluated_dates,
                "evaluated_rows": evaluated_rows,
                "residual_variance_ratio": (
                    float(ratio) if np.isfinite(ratio) else None
                ),
                "minimum_residual_variance_ratio": (
                    self.settings.min_residual_variance_ratio
                ),
                "passed_variance_gate": passed,
            }
        return residualized, diagnostics

    def _diagnostics(
        self,
        *,
        provided: Sequence[str],
        core: Sequence[str],
        core_result: SelectionResult,
        addition_result: SelectionResult | None,
        residualization: Mapping[str, Mapping[str, Any]],
        selected: Sequence[str],
        factor_diagnostics: Mapping[str, Mapping[str, Any]],
        status: str,
        as_of_date: str,
    ) -> dict[str, Any]:
        admitted = [
            factor for factor in selected if factor in set(self.settings.candidate_factors)
        ]
        return {
            "status": status,
            "as_of_date": str(as_of_date),
            "provided_candidate_count": len(provided),
            "core_candidate_count": len(core),
            "incremental_candidate_count": len(self.settings.candidate_factors),
            "core_selected_count": len(core_result.factor_names),
            "incremental_selected_count": len(admitted),
            "selected_count": len(selected),
            "selected_factors": list(selected),
            "core_selected_factors": list(core_result.factor_names),
            "incremental_selected_factors": admitted,
            "multiple_testing_domains": {
                "core": list(core),
                "additions": list(self.settings.candidate_factors),
            },
            "settings": asdict(self.settings),
            "residualization": {
                factor: dict(values) for factor, values in residualization.items()
            },
            "core_selection": dict(core_result.diagnostics),
            "addition_selection": (
                dict(addition_result.diagnostics)
                if addition_result is not None
                else None
            ),
            "factor_diagnostics": {
                factor: dict(values) for factor, values in factor_diagnostics.items()
            },
        }
