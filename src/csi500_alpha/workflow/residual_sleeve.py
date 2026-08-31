from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.research.diagnostics import newey_west_mean_standard_error
from csi500_alpha.workflow.contracts import AlphaModel, ModelFitSummary
from csi500_alpha.workflow.family_models import FamilyAlphaModel, FamilyModelSettings


@dataclass(frozen=True)
class ResidualSleeveSettings:
    """Settings for a frozen core sleeve plus an incremental residual sleeve."""

    candidate_factors: tuple[str, ...]
    core: FamilyModelSettings
    min_core_fraction: float = 0.50
    min_candidate_fraction: float = 0.50
    min_cross_section: int = 100
    min_sleeve_dates: int = 52
    lookback_dates: int = 156
    oof_segments: int = 4
    min_oof_train_dates: int = 36
    min_oof_blocks: int = 2
    min_oof_positive_block_fraction: float = 0.50
    min_oof_t: float = 0.0
    oof_target_t: float = 1.0
    candidate_weighting_method: str = "equal"
    candidate_recent_segments: int = 2
    min_candidate_recent_positive_fraction: float = 0.0
    min_candidate_weight_t: float = 0.0
    candidate_weight_full_confidence_t: float = 0.0
    candidate_churn_floor: float = 0.02
    max_blend_churn_ratio: float = 0.0
    turnover_budget_grid_size: int = 41
    hac_max_lags: int = 4
    risk_aversion: float = 3.0
    candidate_anchor_penalty: float = 0.001
    candidate_change_penalty: float = 0.001
    linear_cost_bps: float = 5.0
    stamp_duty_change_date: str = "20230828"
    stamp_duty_before: float = 0.001
    stamp_duty_after: float = 0.0005
    feasibility_tolerance: float = 1e-8

    def validate(self) -> None:
        self.core.validate()
        if not self.candidate_factors:
            raise ConfigurationError(
                "Residual sleeve candidate_factors cannot be empty"
            )
        if len(set(self.candidate_factors)) != len(self.candidate_factors):
            raise ConfigurationError(
                "Residual sleeve candidate_factors cannot contain duplicates"
            )
        fractions = {
            "min_core_fraction": self.min_core_fraction,
            "min_candidate_fraction": self.min_candidate_fraction,
            "min_oof_positive_block_fraction": (
                self.min_oof_positive_block_fraction
            ),
        }
        if any(not 0 < value <= 1 for value in fractions.values()):
            raise ConfigurationError(
                "Residual sleeve coverage and OOF fractions must be in (0, 1]"
            )
        positive_counts = {
            "min_cross_section": self.min_cross_section,
            "min_sleeve_dates": self.min_sleeve_dates,
            "lookback_dates": self.lookback_dates,
            "oof_segments": self.oof_segments,
            "min_oof_train_dates": self.min_oof_train_dates,
            "min_oof_blocks": self.min_oof_blocks,
            "candidate_recent_segments": self.candidate_recent_segments,
            "turnover_budget_grid_size": self.turnover_budget_grid_size,
        }
        if any(value < 1 for value in positive_counts.values()):
            raise ConfigurationError(
                "Residual sleeve sample counts must be positive"
            )
        if self.lookback_dates < self.min_sleeve_dates:
            raise ConfigurationError(
                "Residual sleeve lookback cannot be shorter than min_sleeve_dates"
            )
        if self.oof_segments < 2:
            raise ConfigurationError(
                "Residual sleeve oof_segments must be at least two"
            )
        if self.min_oof_blocks >= self.oof_segments:
            raise ConfigurationError(
                "Residual sleeve min_oof_blocks must be smaller than oof_segments"
            )
        if self.turnover_budget_grid_size < 2:
            raise ConfigurationError(
                "Residual sleeve turnover_budget_grid_size must be at least two"
            )
        if self.candidate_weighting_method not in {"equal", "net_stability"}:
            raise ConfigurationError(
                "Residual sleeve candidate_weighting_method must be equal or "
                "net_stability"
            )
        if not 0 <= self.min_candidate_recent_positive_fraction <= 1:
            raise ConfigurationError(
                "Residual sleeve min_candidate_recent_positive_fraction must be "
                "in [0, 1]"
            )
        nonnegative = {
            "min_oof_t": self.min_oof_t,
            "oof_target_t": self.oof_target_t,
            "min_candidate_weight_t": self.min_candidate_weight_t,
            "candidate_weight_full_confidence_t": (
                self.candidate_weight_full_confidence_t
            ),
            "candidate_churn_floor": self.candidate_churn_floor,
            "max_blend_churn_ratio": self.max_blend_churn_ratio,
            "risk_aversion": self.risk_aversion,
            "candidate_anchor_penalty": self.candidate_anchor_penalty,
            "candidate_change_penalty": self.candidate_change_penalty,
            "linear_cost_bps": self.linear_cost_bps,
            "stamp_duty_before": self.stamp_duty_before,
            "stamp_duty_after": self.stamp_duty_after,
            "feasibility_tolerance": self.feasibility_tolerance,
        }
        if any(not np.isfinite(value) or value < 0 for value in nonnegative.values()):
            raise ConfigurationError(
                "Residual sleeve penalties and cost settings must be finite and "
                "nonnegative"
            )
        if self.oof_target_t < self.min_oof_t:
            raise ConfigurationError(
                "Residual sleeve oof_target_t cannot be below min_oof_t"
            )
        if (
            self.candidate_weight_full_confidence_t
            < self.min_candidate_weight_t
        ):
            raise ConfigurationError(
                "Residual sleeve candidate_weight_full_confidence_t cannot be "
                "below min_candidate_weight_t"
            )
        if self.candidate_churn_floor <= 0:
            raise ConfigurationError(
                "Residual sleeve candidate_churn_floor must be positive"
            )
        if 0 < self.max_blend_churn_ratio < 1:
            raise ConfigurationError(
                "Residual sleeve max_blend_churn_ratio must be zero to disable "
                "the budget or at least one"
            )
        if self.hac_max_lags < 0:
            raise ConfigurationError(
                "Residual sleeve hac_max_lags cannot be negative"
            )
        if (
            len(self.stamp_duty_change_date) != 8
            or not self.stamp_duty_change_date.isdigit()
        ):
            raise ConfigurationError(
                "Residual sleeve stamp_duty_change_date must be YYYYMMDD"
            )


class ResidualSleeveBlendAlphaModel:
    """Blend the frozen core with a separately residualized candidate sleeve.

    The core keeps the A0 family-equal construction. Candidate scores are
    direction-aligned and residualized against the same core composite used by
    incremental admission, one contemporaneous cross-section at a time. The
    candidate share is a continuous solution based on net sleeve returns and is
    shrunk by chronological out-of-fold evidence.
    """

    name = "residual_sleeve_blend"

    def __init__(
        self,
        *,
        directions: Mapping[str, int],
        families: Mapping[str, str],
        settings: ResidualSleeveSettings,
        name: str = "residual_sleeve_blend",
        method: str = "oof_net_residual_sleeve_blend",
    ) -> None:
        settings.validate()
        invalid = sorted(
            factor for factor, direction in directions.items() if direction not in {-1, 1}
        )
        if invalid:
            raise ConfigurationError(
                f"Residual sleeve directions must be -1 or 1: {invalid}"
            )
        missing_candidates = sorted(
            set(settings.candidate_factors).difference(directions)
        )
        if missing_candidates:
            raise ConfigurationError(
                "Residual sleeve candidates lack directions: "
                f"{missing_candidates}"
            )
        self.directions = {str(name): int(value) for name, value in directions.items()}
        self.families = {str(name): str(value) for name, value in families.items()}
        self.settings = settings
        self.name = str(name)
        self.method = str(method)
        self.core_model: FamilyAlphaModel | None = None
        self.core_factors: tuple[str, ...] = ()
        self.candidate_factors: tuple[str, ...] = ()
        self.candidate_multiplier = 0.0
        self.candidate_share = 0.0
        self.candidate_inner_weights: dict[str, float] = {}
        self.factor_weights: dict[str, float] = {}
        self._previous_candidate_multiplier = 0.0
        self._previous_candidate_share = 0.0
        self._previous_factor_weights: dict[str, float] = {}
        self._has_inherited_state = False

    def inherit_refit_state(self, previous_model: AlphaModel | None) -> None:
        self._previous_candidate_multiplier = 0.0
        self._previous_candidate_share = 0.0
        self._previous_factor_weights = {}
        self._has_inherited_state = False
        if not isinstance(previous_model, ResidualSleeveBlendAlphaModel):
            return
        if previous_model.name != self.name:
            return
        if previous_model.settings.candidate_factors != self.settings.candidate_factors:
            return
        self._previous_candidate_multiplier = float(
            previous_model.candidate_multiplier
        )
        self._previous_candidate_share = float(previous_model.candidate_share)
        self._previous_factor_weights = dict(previous_model.factor_weights)
        self._has_inherited_state = bool(previous_model.core_factors)

    def fit(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> ModelFitSummary:
        names = tuple(str(name) for name in factor_names)
        if not names:
            raise InsufficientTrainingData("No factors were selected")
        if len(set(names)) != len(names):
            raise ConfigurationError(
                "Residual sleeve factor names must be unique"
            )
        candidate_set = set(self.settings.candidate_factors)
        core_factors = tuple(name for name in names if name not in candidate_set)
        candidate_factors = tuple(name for name in names if name in candidate_set)
        if not core_factors:
            raise InsufficientTrainingData(
                "Residual sleeve requires at least one selected core factor"
            )
        missing_directions = sorted(set(names).difference(self.directions))
        missing_families = sorted(set(names).difference(self.families))
        if missing_directions:
            raise ConfigurationError(
                f"Residual sleeve lacks directions: {missing_directions}"
            )
        if missing_families:
            raise ConfigurationError(
                f"Residual sleeve lacks families: {missing_families}"
            )

        core_model = FamilyAlphaModel(
            method="direction_equal",
            directions=self.directions,
            families=self.families,
            settings=self.settings.core,
        )
        core_fit = core_model.fit(
            training,
            core_factors,
            label_column=label_column,
            as_of_date=as_of_date,
        )
        sample = self._mature_training(
            training,
            names,
            label_column=label_column,
            as_of_date=as_of_date,
        )
        decision_dates = int(sample["decision_date"].nunique())
        if decision_dates < self.settings.min_sleeve_dates:
            raise InsufficientTrainingData(
                "Residual sleeve requires at least "
                f"{self.settings.min_sleeve_dates} mature dates; received "
                f"{decision_dates}"
            )

        core_score = core_model.predict(sample)
        residuals, residualization = self._candidate_residuals(
            sample,
            core_factors,
            candidate_factors,
        )
        candidate_inner_weights, candidate_weighting = (
            self._candidate_inner_weighting(
                sample,
                residuals,
                label_column=label_column,
            )
        )
        candidate_score = self._candidate_composite(
            residuals,
            weights=candidate_inner_weights,
        )
        evidence = self._sleeve_evidence(
            sample,
            core_score,
            candidate_score,
            residuals,
            candidate_inner_weights,
            label_column=label_column,
        )
        candidate_multiplier = float(evidence["candidate_multiplier"])
        candidate_share = float(evidence["candidate_share"])
        core_share = 1.0 - candidate_share

        factor_weights = {
            factor: core_share * float(core_model.factor_weights[factor])
            for factor in core_factors
        }
        factor_weights.update(
            {
                factor: candidate_share * candidate_inner_weights[factor]
                for factor in candidate_factors
            }
        )
        previous_weights = (
            self._previous_factor_weights if self._has_inherited_state else factor_weights
        )
        union = set(factor_weights) | set(previous_weights)
        realized_change = float(
            sum(
                abs(
                    factor_weights.get(factor, 0.0)
                    - previous_weights.get(factor, 0.0)
                )
                for factor in union
            )
        )
        family_factors: dict[str, list[str]] = {}
        family_weights: dict[str, float] = {}
        for factor, weight in factor_weights.items():
            family = self.families[factor]
            family_factors.setdefault(family, []).append(factor)
            family_weights[family] = family_weights.get(family, 0.0) + weight

        self.core_model = core_model
        self.core_factors = core_factors
        self.candidate_factors = candidate_factors
        self.candidate_multiplier = candidate_multiplier
        self.candidate_share = candidate_share
        self.candidate_inner_weights = candidate_inner_weights
        self.factor_weights = factor_weights
        concentration = float(sum(weight * weight for weight in factor_weights.values()))
        return ModelFitSummary(
            observations=len(sample),
            decision_dates=decision_dates,
            parameters={
                "method": self.method,
                "as_of_date": str(as_of_date),
                "settings": asdict(self.settings),
                "core_factors": list(core_factors),
                "candidate_factors": list(candidate_factors),
                "core_share": core_share,
                "candidate_multiplier": candidate_multiplier,
                "candidate_share": candidate_share,
                "previous_candidate_multiplier": (
                    self._previous_candidate_multiplier
                ),
                "previous_candidate_share": self._previous_candidate_share,
                "candidate_inner_weights": candidate_inner_weights,
                "candidate_weighting": candidate_weighting,
                "core_model": dict(core_fit.parameters),
                "residualization": residualization,
                "sleeve_evidence": evidence,
                "factor_weights": factor_weights,
                "family_factors": family_factors,
                "family_weights": family_weights,
                "realized_factor_weight_l1_change": realized_change,
                "weight_concentration": concentration,
                "effective_factor_count": (
                    1.0 / concentration if concentration > 0 else 0.0
                ),
                "effective_sleeve_count": (
                    1.0 / (core_share * core_share + candidate_share * candidate_share)
                ),
            },
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.core_model is None or not self.core_factors:
            raise RuntimeError(
                "Residual sleeve model must be fitted before prediction"
            )
        core_score = self.core_model.predict(frame)
        if (
            self.candidate_share <= self.settings.feasibility_tolerance
            or not self.candidate_factors
        ):
            return core_score.rename("score")
        residuals, _ = self._candidate_residuals(
            frame,
            self.core_factors,
            self.candidate_factors,
        )
        candidate_score = self._candidate_composite(
            residuals,
            weights=self.candidate_inner_weights,
        )
        scaled_candidate = self._match_cross_sectional_scale(
            frame,
            reference=core_score,
            candidate=candidate_score,
        )
        core_share = 1.0 - self.candidate_share
        numerator = core_score.fillna(0.0) * core_share
        numerator += scaled_candidate.fillna(0.0) * self.candidate_share
        available = core_score.notna().astype(float) * core_share
        available += scaled_candidate.notna().astype(float) * self.candidate_share
        valid = core_score.notna() & available.gt(0.0)
        return numerator.div(available).where(valid).rename("score")

    def _mature_training(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> pd.DataFrame:
        required = {
            "decision_date",
            "instrument",
            "label_available_date",
            label_column,
            *(f"{factor}__z" for factor in factor_names),
        }
        missing = sorted(required.difference(training.columns))
        if missing:
            raise ConfigurationError(
                f"Residual sleeve training data lacks columns: {missing}"
            )
        if training.duplicated(["decision_date", "instrument"]).any():
            raise ConfigurationError(
                "Residual sleeve decision_date/instrument key is not unique"
            )
        decision_date = training["decision_date"].astype(str)
        available = training["label_available_date"]
        visible = decision_date.lt(str(as_of_date))
        visible &= available.notna()
        visible &= available.fillna("").astype(str).lt(str(as_of_date))
        visible &= pd.to_numeric(training[label_column], errors="coerce").notna()
        sample = training.loc[visible].copy()
        dates = sorted(sample["decision_date"].astype(str).unique())
        recent = set(dates[-self.settings.lookback_dates :])
        return sample.loc[
            sample["decision_date"].astype(str).isin(recent)
        ].copy()

    def _candidate_residuals(
        self,
        frame: pd.DataFrame,
        core_factors: Sequence[str],
        candidate_factors: Sequence[str],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        residuals = pd.DataFrame(index=frame.index)
        diagnostics: dict[str, dict[str, Any]] = {}
        if not candidate_factors:
            return residuals, diagnostics
        core_matrix = pd.DataFrame(index=frame.index)
        for factor in core_factors:
            column = f"{factor}__z"
            if column not in frame:
                raise ConfigurationError(
                    f"Residual sleeve input lacks column: {column}"
                )
            core_matrix[factor] = (
                pd.to_numeric(frame[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                * self.directions[factor]
            )
        minimum_core = max(
            1,
            int(np.ceil(len(core_factors) * self.settings.min_core_fraction)),
        )
        core_composite = core_matrix.mean(axis=1, skipna=True).where(
            core_matrix.notna().sum(axis=1).ge(minimum_core)
        )
        grouped = frame.groupby("decision_date", sort=True).groups
        for factor in candidate_factors:
            column = f"{factor}__z"
            if column not in frame:
                raise ConfigurationError(
                    f"Residual sleeve input lacks column: {column}"
                )
            directed = (
                pd.to_numeric(frame[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                * self.directions[factor]
            )
            output = pd.Series(np.nan, index=frame.index, dtype=float)
            slopes: list[float] = []
            raw_sum_squares = 0.0
            residual_sum_squares = 0.0
            evaluated_dates = 0
            for indices in grouped.values():
                cross_section = pd.DataFrame(
                    {
                        "core": core_composite.loc[indices],
                        "candidate": directed.loc[indices],
                    }
                ).dropna()
                if (
                    len(cross_section) < self.settings.min_cross_section
                    or cross_section["candidate"].nunique() < 2
                    or cross_section["core"].nunique() < 2
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
                slopes.append(float(coefficients[1]))
                raw_sum_squares += raw_variance
                residual_sum_squares += float(values @ values)
                evaluated_dates += 1
            residuals[factor] = output
            variance_ratio = (
                residual_sum_squares / raw_sum_squares
                if raw_sum_squares > 0
                else np.nan
            )
            diagnostics[factor] = {
                "evaluated_dates": evaluated_dates,
                "mean_cross_sectional_slope": (
                    float(np.mean(slopes)) if slopes else None
                ),
                "median_cross_sectional_slope": (
                    float(np.median(slopes)) if slopes else None
                ),
                "residual_variance_ratio": (
                    float(variance_ratio) if np.isfinite(variance_ratio) else None
                ),
            }
        return residuals, diagnostics

    def _candidate_inner_weighting(
        self,
        sample: pd.DataFrame,
        residuals: pd.DataFrame,
        *,
        label_column: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        factors = tuple(str(factor) for factor in residuals.columns)
        if not factors:
            return {}, {
                "method": self.settings.candidate_weighting_method,
                "status": "no_admitted_candidates",
                "eligible_factors": [],
                "factor_statistics": {},
            }
        if self.settings.candidate_weighting_method == "equal":
            weights = {factor: 1.0 / len(factors) for factor in factors}
            return weights, {
                "method": "equal",
                "status": "weighted",
                "eligible_factors": list(factors),
                "factor_statistics": {},
                "normalized_weights": weights,
            }

        statistics: dict[str, dict[str, Any]] = {}
        raw_strength: dict[str, float] = {}
        for factor in factors:
            score = pd.to_numeric(residuals[factor], errors="coerce")
            net_spread = self._daily_net_spread(
                sample,
                score,
                label_column=label_column,
            )
            mean_spread, standard_error = newey_west_mean_standard_error(
                net_spread,
                max_lags=self.settings.hac_max_lags,
            )
            if np.isfinite(mean_spread) and np.isfinite(standard_error):
                if standard_error > 1e-16:
                    net_t = float(mean_spread / standard_error)
                elif mean_spread > 0:
                    net_t = float("inf")
                else:
                    net_t = 0.0
            else:
                net_t = np.nan
            segment_means = self._contiguous_segment_means(
                net_spread,
                segments=self.settings.oof_segments,
            )
            recent_means = segment_means[
                -min(
                    self.settings.candidate_recent_segments,
                    len(segment_means),
                ) :
            ]
            recent_positive_fraction = (
                float(np.mean([value > 0 for value in recent_means]))
                if recent_means
                else 0.0
            )
            score_churn = self._score_churn(sample, score)
            t_passed = bool(
                net_t == float("inf")
                or (
                    np.isfinite(net_t)
                    and net_t >= self.settings.min_candidate_weight_t
                )
            )
            eligible = bool(
                np.isfinite(mean_spread)
                and mean_spread > 0
                and t_passed
                and recent_positive_fraction
                >= self.settings.min_candidate_recent_positive_fraction
            )
            confidence = self._threshold_confidence(
                net_t,
                passed=eligible,
                minimum=self.settings.min_candidate_weight_t,
                full=self.settings.candidate_weight_full_confidence_t,
            )
            churn_denominator = (
                max(score_churn, self.settings.candidate_churn_floor)
                if np.isfinite(score_churn)
                else 1.0
            )
            strength = (
                float(
                    mean_spread
                    * confidence
                    * recent_positive_fraction
                    / churn_denominator
                )
                if eligible
                else 0.0
            )
            if not np.isfinite(strength) or strength <= 0:
                strength = 0.0
            raw_strength[factor] = strength
            statistics[factor] = {
                "sleeve_dates": int(len(net_spread)),
                "mean_net_spread": (
                    float(mean_spread) if np.isfinite(mean_spread) else None
                ),
                "hac_standard_error": (
                    float(standard_error)
                    if np.isfinite(standard_error)
                    else None
                ),
                "net_spread_t": float(net_t) if np.isfinite(net_t) else None,
                "infinite_positive_t": net_t == float("inf"),
                "segment_mean_net_spreads": segment_means,
                "recent_segment_count": len(recent_means),
                "recent_positive_fraction": recent_positive_fraction,
                "mean_score_churn": (
                    float(score_churn) if np.isfinite(score_churn) else None
                ),
                "evidence_passed": eligible,
                "confidence": confidence,
                "raw_strength": strength,
            }
        total_strength = float(sum(raw_strength.values()))
        if total_strength <= self.settings.feasibility_tolerance:
            weights = {factor: 0.0 for factor in factors}
            status = "no_candidate_passed_weight_gate"
        else:
            weights = {
                factor: raw_strength[factor] / total_strength
                for factor in factors
            }
            status = "weighted"
        return weights, {
            "method": "net_stability",
            "status": status,
            "eligible_factors": [
                factor for factor in factors if weights[factor] > 0
            ],
            "factor_statistics": statistics,
            "normalized_weights": weights,
        }

    def _candidate_composite(
        self,
        residuals: pd.DataFrame,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> pd.Series:
        if residuals.empty or not len(residuals.columns):
            return pd.Series(np.nan, index=residuals.index, dtype=float)
        if (
            weights is not None
            and self.settings.candidate_weighting_method != "equal"
        ):
            active_weights = pd.Series(
                {
                    factor: float(weights.get(str(factor), 0.0))
                    for factor in residuals.columns
                },
                dtype=float,
            )
            active_weights = active_weights.where(active_weights.gt(0.0), 0.0)
            total = float(active_weights.sum())
            if total <= self.settings.feasibility_tolerance:
                return pd.Series(np.nan, index=residuals.index, dtype=float)
            active_weights /= total
            available_weight = residuals.notna().mul(active_weights, axis=1).sum(axis=1)
            numerator = residuals.fillna(0.0).mul(active_weights, axis=1).sum(axis=1)
            return numerator.div(available_weight).where(
                available_weight.ge(self.settings.min_candidate_fraction)
            )
        minimum = max(
            1,
            int(
                np.ceil(
                    len(residuals.columns)
                    * self.settings.min_candidate_fraction
                )
            ),
        )
        return residuals.mean(axis=1, skipna=True).where(
            residuals.notna().sum(axis=1).ge(minimum)
        )

    @staticmethod
    def _contiguous_segment_means(
        values: pd.Series,
        *,
        segments: int,
    ) -> list[float]:
        clean = pd.to_numeric(values, errors="coerce").dropna().sort_index()
        if clean.empty:
            return []
        output: list[float] = []
        for positions in np.array_split(np.arange(len(clean)), segments):
            if len(positions):
                output.append(float(clean.iloc[positions].mean()))
        return output

    @staticmethod
    def _threshold_confidence(
        statistic: float,
        *,
        passed: bool,
        minimum: float,
        full: float,
    ) -> float:
        if not passed:
            return 0.0
        if np.isfinite(statistic) and statistic < minimum:
            return 0.0
        if statistic == float("inf") or full <= 0:
            return 1.0
        if not np.isfinite(statistic):
            return 0.0
        return float(np.clip(statistic / full, 0.0, 1.0))

    def _sleeve_evidence(
        self,
        sample: pd.DataFrame,
        core_score: pd.Series,
        candidate_score: pd.Series,
        residuals: pd.DataFrame,
        candidate_inner_weights: Mapping[str, float],
        *,
        label_column: str,
    ) -> dict[str, Any]:
        oof_scope = (
            "conditional_on_admitted_candidates"
            if self.settings.candidate_weighting_method == "equal"
            else "conditional_on_admitted_candidates_with_nested_candidate_weighting"
        )
        core_returns = self._daily_net_spread(
            sample,
            core_score,
            label_column=label_column,
        )
        candidate_returns = self._daily_net_spread(
            sample,
            candidate_score,
            label_column=label_column,
        )
        scaled_candidate = self._match_cross_sectional_scale(
            sample,
            reference=core_score,
            candidate=candidate_score,
        )
        returns = pd.concat(
            [core_returns.rename("core"), candidate_returns.rename("candidate")],
            axis=1,
            join="inner",
        ).dropna()
        if len(returns) < self.settings.min_sleeve_dates:
            return {
                "status": "insufficient_sleeve_dates",
                "oof_evidence_scope": oof_scope,
                "sleeve_dates": int(len(returns)),
                "candidate_multiplier": 0.0,
                "candidate_share": 0.0,
                "raw_candidate_multiplier": 0.0,
                "raw_candidate_share": 0.0,
                "oof_confidence": 0.0,
            }

        raw_multiplier, full_statistics = self._optimal_candidate_multiplier(
            returns,
            previous_multiplier=(
                self._previous_candidate_multiplier
                if self._has_inherited_state
                else 0.0
            ),
            apply_change_penalty=True,
        )
        oof_dates = (
            returns.index
            if self.settings.candidate_weighting_method == "equal"
            else core_returns.index
        )
        positions = np.arange(len(oof_dates))
        blocks = np.array_split(positions, self.settings.oof_segments)
        block_rows: list[dict[str, Any]] = []
        oof_increment: list[pd.Series] = []
        for block_number in range(1, len(blocks)):
            train_positions = np.concatenate(blocks[:block_number])
            test_positions = blocks[block_number]
            if (
                len(train_positions) < self.settings.min_oof_train_dates
                or not len(test_positions)
            ):
                continue
            block_inner_weights = dict(candidate_inner_weights)
            holdout_dates = {
                str(value) for value in oof_dates[test_positions]
            }
            holdout_mask = sample["decision_date"].astype(str).isin(holdout_dates)
            holdout_sample = sample.loc[holdout_mask]
            holdout_core_score = core_score.loc[holdout_mask]
            holdout_candidate_score = scaled_candidate.loc[holdout_mask]
            if self.settings.candidate_weighting_method == "equal":
                block_multiplier, _ = self._optimal_candidate_multiplier(
                    returns.iloc[train_positions],
                    previous_multiplier=0.0,
                    apply_change_penalty=False,
                )
            else:
                train_dates = {
                    str(value) for value in oof_dates[train_positions]
                }
                train_mask = sample["decision_date"].astype(str).isin(train_dates)
                train_sample = sample.loc[train_mask]
                train_residuals = residuals.loc[train_mask]
                block_inner_weights, _ = self._candidate_inner_weighting(
                    train_sample,
                    train_residuals,
                    label_column=label_column,
                )
                train_candidate_score = self._candidate_composite(
                    train_residuals,
                    weights=block_inner_weights,
                )
                train_core_score = core_score.loc[train_mask]
                train_scaled_candidate = self._match_cross_sectional_scale(
                    train_sample,
                    reference=train_core_score,
                    candidate=train_candidate_score,
                )
                train_core_returns = self._daily_net_spread(
                    train_sample,
                    train_core_score,
                    label_column=label_column,
                )
                train_candidate_returns = self._daily_net_spread(
                    train_sample,
                    train_scaled_candidate,
                    label_column=label_column,
                )
                block_returns = pd.concat(
                    [
                        train_core_returns.rename("core"),
                        train_candidate_returns.rename("candidate"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(block_returns) >= self.settings.min_oof_train_dates:
                    block_multiplier, _ = self._optimal_candidate_multiplier(
                        block_returns,
                        previous_multiplier=0.0,
                        apply_change_penalty=False,
                    )
                else:
                    block_multiplier = 0.0
                holdout_candidate = self._candidate_composite(
                    residuals.loc[holdout_mask],
                    weights=block_inner_weights,
                )
                holdout_candidate_score = self._match_cross_sectional_scale(
                    holdout_sample,
                    reference=holdout_core_score,
                    candidate=holdout_candidate,
                )
            holdout_core_returns = self._daily_net_spread(
                holdout_sample,
                holdout_core_score,
                label_column=label_column,
            )
            if block_multiplier <= self.settings.feasibility_tolerance:
                holdout_combined_returns = holdout_core_returns.copy()
            else:
                holdout_combined_score = (
                    holdout_core_score
                    + block_multiplier * holdout_candidate_score
                ) / (1.0 + block_multiplier)
                holdout_combined_returns = self._daily_net_spread(
                    holdout_sample,
                    holdout_combined_score,
                    label_column=label_column,
                )
            increment = holdout_combined_returns.sub(
                holdout_core_returns,
            ).dropna()
            if increment.empty:
                continue
            oof_increment.append(increment)
            block_share = block_multiplier / (1.0 + block_multiplier)
            block_rows.append(
                {
                    "block": block_number + 1,
                    "train_dates": int(len(train_positions)),
                    "test_dates": int(len(test_positions)),
                    "candidate_multiplier": block_multiplier,
                    "candidate_share": block_share,
                    "candidate_inner_weights": block_inner_weights,
                    "mean_net_increment": float(increment.mean()),
                }
            )
        joined_increment = (
            pd.concat(oof_increment).sort_index()
            if oof_increment
            else pd.Series(dtype=float)
        )
        mean_increment, standard_error = newey_west_mean_standard_error(
            joined_increment,
            max_lags=self.settings.hac_max_lags,
        )
        if np.isfinite(mean_increment) and np.isfinite(standard_error):
            if standard_error > 1e-16:
                oof_t = float(mean_increment / standard_error)
            elif mean_increment > 0:
                oof_t = float("inf")
            else:
                oof_t = 0.0
        else:
            oof_t = np.nan
        positive_fraction = (
            float(np.mean([row["mean_net_increment"] > 0 for row in block_rows]))
            if block_rows
            else 0.0
        )
        enough_blocks = len(block_rows) >= self.settings.min_oof_blocks
        evidence_passed = bool(
            enough_blocks
            and np.isfinite(mean_increment)
            and mean_increment > 0
            and positive_fraction
            >= self.settings.min_oof_positive_block_fraction
            and (
                oof_t == float("inf")
                or (
                    np.isfinite(oof_t)
                    and oof_t >= self.settings.min_oof_t
                )
            )
        )
        confidence = self._threshold_confidence(
            oof_t,
            passed=evidence_passed,
            minimum=self.settings.min_oof_t,
            full=self.settings.oof_target_t,
        )
        unconstrained_multiplier = float(max(raw_multiplier * confidence, 0.0))
        unconstrained_share = unconstrained_multiplier / (
            1.0 + unconstrained_multiplier
        )
        turnover_budget = self._apply_score_churn_budget(
            sample,
            core_score,
            scaled_candidate,
            unconstrained_share=unconstrained_share,
        )
        candidate_share = float(turnover_budget["candidate_share"])
        candidate_multiplier = (
            candidate_share / (1.0 - candidate_share)
            if candidate_share < 1.0 - self.settings.feasibility_tolerance
            else unconstrained_multiplier
        )
        raw_share = raw_multiplier / (1.0 + raw_multiplier)
        return {
            "status": "admitted" if candidate_share > 0 else "core_only",
            "oof_evidence_scope": oof_scope,
            "sleeve_dates": int(len(returns)),
            "core_mean_net_spread": float(returns["core"].mean()),
            "candidate_mean_net_spread": float(returns["candidate"].mean()),
            "raw_candidate_multiplier": raw_multiplier,
            "raw_candidate_share": raw_share,
            "candidate_multiplier": candidate_multiplier,
            "candidate_share": candidate_share,
            "oof_candidate_multiplier": unconstrained_multiplier,
            "oof_candidate_share": unconstrained_share,
            "previous_candidate_multiplier": (
                self._previous_candidate_multiplier
            ),
            "previous_candidate_share": self._previous_candidate_share,
            "oof_blocks": block_rows,
            "oof_block_count": len(block_rows),
            "oof_mean_net_increment": (
                float(mean_increment) if np.isfinite(mean_increment) else None
            ),
            "oof_hac_standard_error": (
                float(standard_error) if np.isfinite(standard_error) else None
            ),
            "oof_t": float(oof_t) if np.isfinite(oof_t) else None,
            "oof_positive_block_fraction": positive_fraction,
            "minimum_oof_t": self.settings.min_oof_t,
            "full_confidence_oof_t": self.settings.oof_target_t,
            "oof_evidence_passed": evidence_passed,
            "oof_confidence": confidence,
            "turnover_budget": turnover_budget,
            "full_sample_objective": full_statistics,
        }

    def _apply_score_churn_budget(
        self,
        sample: pd.DataFrame,
        core_score: pd.Series,
        scaled_candidate: pd.Series,
        *,
        unconstrained_share: float,
    ) -> dict[str, Any]:
        enabled = self.settings.max_blend_churn_ratio > 0
        if not enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "maximum_blend_to_core_churn_ratio": None,
                "unconstrained_candidate_share": unconstrained_share,
                "candidate_share": unconstrained_share,
                "binding": False,
            }
        core_churn = self._score_churn(sample, core_score)
        if unconstrained_share <= self.settings.feasibility_tolerance:
            return {
                "enabled": True,
                "status": "no_candidate_evidence",
                "maximum_blend_to_core_churn_ratio": (
                    self.settings.max_blend_churn_ratio
                ),
                "core_score_churn": (
                    float(core_churn) if np.isfinite(core_churn) else None
                ),
                "allowed_blend_score_churn": (
                    float(core_churn * self.settings.max_blend_churn_ratio)
                    if np.isfinite(core_churn)
                    else None
                ),
                "unconstrained_blend_score_churn": None,
                "budgeted_blend_score_churn": (
                    float(core_churn) if np.isfinite(core_churn) else None
                ),
                "unconstrained_candidate_share": unconstrained_share,
                "candidate_share": 0.0,
                "binding": False,
            }
        if not np.isfinite(core_churn):
            return {
                "enabled": True,
                "status": "insufficient_core_churn_history",
                "maximum_blend_to_core_churn_ratio": (
                    self.settings.max_blend_churn_ratio
                ),
                "core_score_churn": None,
                "allowed_blend_score_churn": None,
                "unconstrained_blend_score_churn": None,
                "budgeted_blend_score_churn": None,
                "unconstrained_candidate_share": unconstrained_share,
                "candidate_share": 0.0,
                "binding": True,
            }

        allowed_churn = core_churn * self.settings.max_blend_churn_ratio
        grid = np.linspace(
            0.0,
            unconstrained_share,
            self.settings.turnover_budget_grid_size,
        )
        grid_rows: list[dict[str, float]] = []
        for share in grid:
            blend_score = self._blend_scores(
                core_score,
                scaled_candidate,
                candidate_share=float(share),
            )
            churn = self._score_churn(sample, blend_score)
            grid_rows.append(
                {
                    "candidate_share": float(share),
                    "blend_score_churn": (
                        float(churn) if np.isfinite(churn) else float("inf")
                    ),
                }
            )
        feasible_rows = [
            row
            for row in grid_rows
            if row["blend_score_churn"]
            <= allowed_churn + self.settings.feasibility_tolerance
        ]
        selected = (
            max(feasible_rows, key=lambda row: row["candidate_share"])
            if feasible_rows
            else {
                "candidate_share": 0.0,
                "blend_score_churn": float(core_churn),
            }
        )
        selected_share = float(selected["candidate_share"])
        selected_churn = float(selected["blend_score_churn"])
        unconstrained_churn = float(grid_rows[-1]["blend_score_churn"])
        return {
            "enabled": True,
            "status": "budget_applied",
            "maximum_blend_to_core_churn_ratio": (
                self.settings.max_blend_churn_ratio
            ),
            "core_score_churn": float(core_churn),
            "allowed_blend_score_churn": float(allowed_churn),
            "unconstrained_blend_score_churn": (
                unconstrained_churn
                if np.isfinite(unconstrained_churn)
                else None
            ),
            "budgeted_blend_score_churn": selected_churn,
            "unconstrained_candidate_share": unconstrained_share,
            "candidate_share": selected_share,
            "binding": bool(
                selected_share
                < unconstrained_share - self.settings.feasibility_tolerance
            ),
            "grid_size": len(grid_rows),
        }

    @staticmethod
    def _blend_scores(
        core_score: pd.Series,
        candidate_score: pd.Series,
        *,
        candidate_share: float,
    ) -> pd.Series:
        core_share = 1.0 - candidate_share
        numerator = core_score.fillna(0.0) * core_share
        numerator += candidate_score.fillna(0.0) * candidate_share
        available = core_score.notna().astype(float) * core_share
        available += candidate_score.notna().astype(float) * candidate_share
        valid = core_score.notna() & available.gt(0.0)
        return numerator.div(available).where(valid).rename("score")

    def _score_churn(
        self,
        sample: pd.DataFrame,
        score: pd.Series,
    ) -> float:
        working = sample[["decision_date", "instrument"]].copy()
        working["score"] = pd.to_numeric(score, errors="coerce")
        churn_values: list[float] = []
        previous_rank: pd.Series | None = None
        for _, frame in working.groupby("decision_date", sort=True):
            current_rank = (
                frame.dropna(subset=["score"])
                .set_index("instrument")["score"]
                .rank(method="average", pct=True)
            )
            if previous_rank is not None:
                common = previous_rank.index.intersection(current_rank.index)
                if len(common) >= self.settings.min_cross_section:
                    churn_values.append(
                        float(
                            (
                                current_rank.loc[common]
                                - previous_rank.loc[common]
                            )
                            .abs()
                            .median()
                        )
                    )
            previous_rank = current_rank
        return float(np.mean(churn_values)) if churn_values else np.nan

    def _optimal_candidate_multiplier(
        self,
        returns: pd.DataFrame,
        *,
        previous_multiplier: float,
        apply_change_penalty: bool,
    ) -> tuple[float, dict[str, float]]:
        core = returns["core"].to_numpy(dtype=float)
        candidate = returns["candidate"].to_numpy(dtype=float)
        mean_candidate = float(np.mean(candidate))
        variance_candidate = (
            float(np.var(candidate, ddof=1)) if len(candidate) > 1 else 0.0
        )
        covariance = (
            float(np.cov(core, candidate, ddof=1)[0, 1])
            if len(candidate) > 1
            else 0.0
        )
        change_penalty = (
            self.settings.candidate_change_penalty
            if apply_change_penalty
            else 0.0
        )
        numerator = (
            mean_candidate
            - 2.0 * self.settings.risk_aversion * covariance
            + 2.0 * change_penalty * previous_multiplier
        )
        denominator = 2.0 * (
            self.settings.risk_aversion * variance_candidate
            + self.settings.candidate_anchor_penalty
            + change_penalty
        )
        multiplier = (
            0.0
            if denominator <= 1e-16
            else float(max(numerator / denominator, 0.0))
        )
        if not np.isfinite(multiplier):
            multiplier = 0.0
        return multiplier, {
            "mean_candidate_net_spread": mean_candidate,
            "variance_candidate_net_spread": variance_candidate,
            "core_candidate_covariance": covariance,
            "objective_numerator": numerator,
            "objective_denominator": denominator,
            "candidate_multiplier": multiplier,
            "candidate_share": multiplier / (1.0 + multiplier),
        }

    def _daily_net_spread(
        self,
        sample: pd.DataFrame,
        score: pd.Series,
        *,
        label_column: str,
    ) -> pd.Series:
        working = sample[["decision_date", "instrument", label_column]].copy()
        working["score"] = pd.to_numeric(score, errors="coerce")
        working["label"] = pd.to_numeric(
            sample[label_column], errors="coerce"
        )
        if "label_entry_date" in sample:
            working["trade_date"] = sample["label_entry_date"].astype(str)
        else:
            working["trade_date"] = sample["decision_date"].astype(str)
        previous_members: dict[int, frozenset[str]] = {}
        output: dict[str, float] = {}
        linear_rate = self.settings.linear_cost_bps / 10_000.0
        for decision_date, frame in working.groupby("decision_date", sort=True):
            valid = frame.dropna(subset=["score", "label"])
            if (
                len(valid) < self.settings.min_cross_section
                or valid["score"].nunique() < 5
            ):
                continue
            quintiles = np.ceil(
                valid["score"].rank(method="first", pct=True) * 5.0
            ).clip(1, 5).astype(int)
            trade_date = str(valid["trade_date"].iloc[0])
            stamp_rate = (
                self.settings.stamp_duty_before
                if trade_date < self.settings.stamp_duty_change_date
                else self.settings.stamp_duty_after
            )
            leg_return: dict[int, float] = {}
            leg_cost: dict[int, float] = {}
            for quintile in (1, 5):
                members = frozenset(
                    valid.loc[quintiles.eq(quintile), "instrument"].astype(str)
                )
                turnover, cost = self._leg_turnover_and_cost(
                    members,
                    previous_members.get(quintile),
                    linear_rate=linear_rate,
                    stamp_rate=stamp_rate,
                )
                del turnover
                previous_members[quintile] = members
                leg_return[quintile] = float(
                    valid.loc[quintiles.eq(quintile), "label"].mean()
                )
                leg_cost[quintile] = cost
            output[str(decision_date)] = (
                leg_return[5]
                - leg_return[1]
                - leg_cost[1]
                - leg_cost[5]
            )
        return pd.Series(output, dtype=float).sort_index()

    @staticmethod
    def _leg_turnover_and_cost(
        current: frozenset[str],
        previous: frozenset[str] | None,
        *,
        linear_rate: float,
        stamp_rate: float,
    ) -> tuple[float, float]:
        if not current:
            raise ValueError("Equal-weight sleeve leg cannot be empty")
        prior = previous or frozenset()
        current_weight = 1.0 / len(current)
        prior_weight = 1.0 / len(prior) if prior else 0.0
        instruments = sorted(current.union(prior))
        delta = np.fromiter(
            (
                (current_weight if instrument in current else 0.0)
                - (prior_weight if instrument in prior else 0.0)
                for instrument in instruments
            ),
            dtype=float,
            count=len(instruments),
        )
        buys = float(np.clip(delta, 0.0, None).sum())
        sells = float((-np.clip(delta, None, 0.0)).sum())
        return max(buys, sells), linear_rate * (buys + sells) + stamp_rate * sells

    @staticmethod
    def _match_cross_sectional_scale(
        frame: pd.DataFrame,
        *,
        reference: pd.Series,
        candidate: pd.Series,
    ) -> pd.Series:
        output = pd.Series(np.nan, index=frame.index, dtype=float)
        for indices in frame.groupby("decision_date", sort=False).groups.values():
            values = pd.DataFrame(
                {
                    "reference": reference.loc[indices],
                    "candidate": candidate.loc[indices],
                }
            ).dropna()
            if len(values) < 2:
                continue
            reference_std = float(values["reference"].std(ddof=0))
            candidate_std = float(values["candidate"].std(ddof=0))
            if reference_std <= 1e-16 or candidate_std <= 1e-16:
                continue
            centered = values["candidate"] - float(values["candidate"].mean())
            output.loc[values.index] = centered * reference_std / candidate_std
        return output
