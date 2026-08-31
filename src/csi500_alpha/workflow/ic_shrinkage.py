from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.research.diagnostics import newey_west_mean_standard_error
from csi500_alpha.workflow.contracts import AlphaModel, ModelFitSummary


@dataclass(frozen=True)
class ICShrinkageSettings:
    """Configuration for uncertainty-aware, cost-conscious factor synthesis."""

    min_cross_section: int = 100
    min_ic_dates: int = 52
    min_churn_dates: int = 12
    lookback_dates: int = 156
    hac_max_lags: int = 4
    min_mean_directed_ic: float = 0.0
    shrinkage_enabled: bool = True
    prior_variance_floor: float = 1e-8
    correlation_penalty: float = 0.005
    cost_penalty: float = 0.01
    weight_turnover_penalty: float = 0.01
    weight_change_norm: str = "l2"
    core_factors: tuple[str, ...] = ()
    core_anchor_mode: str = "equal_family"
    core_anchor_penalty: float = 0.0
    max_factor_weight: float = 0.35
    min_active_factors: int = 3
    min_factor_fraction: float = 0.50
    solvers: tuple[str, ...] = ("CLARABEL", "OSQP")
    feasibility_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.min_cross_section < 5:
            raise ConfigurationError("IC shrinkage min_cross_section must be at least 5")
        if self.min_ic_dates < 6:
            raise ConfigurationError("IC shrinkage min_ic_dates must be at least 6")
        if self.min_churn_dates < 1:
            raise ConfigurationError("IC shrinkage min_churn_dates must be positive")
        if self.lookback_dates < self.min_ic_dates:
            raise ConfigurationError(
                "IC shrinkage lookback_dates cannot be shorter than min_ic_dates"
            )
        if self.hac_max_lags < 0:
            raise ConfigurationError("IC shrinkage hac_max_lags cannot be negative")
        numeric_nonnegative = {
            "prior_variance_floor": self.prior_variance_floor,
            "correlation_penalty": self.correlation_penalty,
            "cost_penalty": self.cost_penalty,
            "weight_turnover_penalty": self.weight_turnover_penalty,
            "core_anchor_penalty": self.core_anchor_penalty,
            "feasibility_tolerance": self.feasibility_tolerance,
        }
        for name, value in numeric_nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ConfigurationError(f"IC shrinkage {name} must be finite and nonnegative")
        if not np.isfinite(self.min_mean_directed_ic):
            raise ConfigurationError(
                "IC shrinkage min_mean_directed_ic must be finite"
            )
        if self.weight_change_norm not in {"l1", "l2"}:
            raise ConfigurationError(
                "IC shrinkage weight_change_norm must be 'l1' or 'l2'"
            )
        if self.core_anchor_mode not in {"equal_factor", "equal_family"}:
            raise ConfigurationError(
                "IC shrinkage core_anchor_mode must be equal_factor or equal_family"
            )
        if len(set(self.core_factors)) != len(self.core_factors):
            raise ConfigurationError(
                "IC shrinkage core_factors cannot contain duplicates"
            )
        if self.core_anchor_penalty > 0 and not self.core_factors:
            raise ConfigurationError(
                "IC shrinkage core_factors are required when the core anchor is active"
            )
        if not 0 < self.max_factor_weight <= 1:
            raise ConfigurationError(
                "IC shrinkage max_factor_weight must be in (0, 1]"
            )
        if self.min_active_factors < 1:
            raise ConfigurationError("IC shrinkage min_active_factors must be positive")
        if self.min_active_factors * self.max_factor_weight < 1 - 1e-12:
            raise ConfigurationError(
                "IC shrinkage min_active_factors and max_factor_weight cannot provide "
                "unit capacity"
            )
        if not 0 < self.min_factor_fraction <= 1:
            raise ConfigurationError(
                "IC shrinkage min_factor_fraction must be in (0, 1]"
            )
        if not self.solvers or any(not solver for solver in self.solvers):
            raise ConfigurationError("IC shrinkage solvers cannot be empty")


class ICShrinkageAlphaModel:
    """Combine directed factors with IC shrinkage and convex operational penalties."""

    name = "ic_shrinkage"

    def __init__(
        self,
        *,
        directions: Mapping[str, int],
        families: Mapping[str, str] | None = None,
        settings: ICShrinkageSettings | None = None,
        name: str = "ic_shrinkage",
    ) -> None:
        invalid = sorted(
            factor for factor, direction in directions.items() if direction not in {-1, 1}
        )
        if invalid:
            raise ConfigurationError(
                f"IC shrinkage directions must be -1 or 1: {invalid}"
            )
        self.settings = settings or ICShrinkageSettings()
        missing_core_directions = sorted(
            set(self.settings.core_factors).difference(directions)
        )
        if missing_core_directions:
            raise ConfigurationError(
                "IC shrinkage core factors lack directions: "
                f"{missing_core_directions}"
            )
        self.directions = dict(directions)
        self.families = dict(families or {})
        if (
            self.settings.core_anchor_penalty > 0
            and self.settings.core_anchor_mode == "equal_family"
        ):
            missing_core_families = sorted(
                set(self.settings.core_factors).difference(self.families)
            )
            if missing_core_families:
                raise ConfigurationError(
                    "IC shrinkage core factors lack families: "
                    f"{missing_core_families}"
                )
        self.name = str(name)
        self.factor_names: tuple[str, ...] = ()
        self.factor_weights: dict[str, float] = {}
        self._previous_weights: dict[str, float] = {}
        self._has_inherited_state = False

    def inherit_refit_state(self, previous_model: AlphaModel | None) -> None:
        self._previous_weights = {}
        self._has_inherited_state = False
        if not isinstance(previous_model, ICShrinkageAlphaModel):
            return
        self._previous_weights = dict(previous_model.factor_weights)
        self._has_inherited_state = bool(previous_model.factor_names)

    def fit(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> ModelFitSummary:
        names = tuple(factor_names)
        if not names:
            raise InsufficientTrainingData("No factors were selected")
        if len(set(names)) != len(names):
            raise ConfigurationError("IC shrinkage factor names must be unique")
        missing_directions = sorted(set(names).difference(self.directions))
        if missing_directions:
            raise ConfigurationError(
                f"IC shrinkage model lacks directions: {missing_directions}"
            )
        sample = self._mature_training(
            training,
            names,
            label_column=label_column,
            as_of_date=as_of_date,
        )
        decision_dates = int(sample["decision_date"].nunique())
        if decision_dates < self.settings.min_ic_dates:
            raise InsufficientTrainingData(
                "IC shrinkage requires at least "
                f"{self.settings.min_ic_dates} mature dates; received {decision_dates}"
            )

        directed = pd.DataFrame(index=sample.index)
        for factor in names:
            directed[factor] = (
                pd.to_numeric(sample[f"{factor}__z"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                * self.directions[factor]
            )
        labels = pd.to_numeric(sample[label_column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        factor_statistics: dict[str, dict[str, Any]] = {}
        prior_candidates: list[str] = []
        for factor in names:
            ic_values = self._daily_ic(sample, directed[factor], labels)
            mean_ic, standard_error = newey_west_mean_standard_error(
                ic_values,
                max_lags=self.settings.hac_max_lags,
            )
            churn_values = self._daily_churn(sample, directed[factor])
            churn = float(churn_values.mean()) if not churn_values.empty else np.nan
            reasons: list[str] = []
            if len(ic_values) < self.settings.min_ic_dates:
                reasons.append("insufficient_ic_dates")
            if not np.isfinite(mean_ic) or not np.isfinite(standard_error):
                reasons.append("invalid_ic_uncertainty")
            elif mean_ic <= self.settings.min_mean_directed_ic:
                reasons.append("nonpositive_directed_ic")
            if len(churn_values) < self.settings.min_churn_dates:
                reasons.append("insufficient_churn_dates")
            if not np.isfinite(churn):
                reasons.append("invalid_churn")
            if not reasons:
                prior_candidates.append(factor)
            factor_statistics[factor] = {
                "direction": self.directions[factor],
                "ic_dates": int(len(ic_values)),
                "mean_directed_ic": _finite_or_none(mean_ic),
                "hac_standard_error": _finite_or_none(standard_error),
                "newey_west_t": _safe_ratio(mean_ic, standard_error),
                "churn_dates": int(len(churn_values)),
                "score_churn": _finite_or_none(churn),
                "exclusion_reasons": reasons,
            }
        if len(prior_candidates) < self.settings.min_active_factors:
            raise InsufficientTrainingData(
                "IC shrinkage has only "
                f"{len(prior_candidates)} statistically usable factors; requires "
                f"{self.settings.min_active_factors}"
            )

        prior_variance_raw = float(
            np.mean(
                [
                    float(factor_statistics[factor]["mean_directed_ic"]) ** 2
                    - float(factor_statistics[factor]["hac_standard_error"]) ** 2
                    for factor in prior_candidates
                ]
            )
        )
        prior_variance = max(prior_variance_raw, self.settings.prior_variance_floor)
        eligible: list[str] = []
        for factor in names:
            statistics = factor_statistics[factor]
            mean_ic = statistics["mean_directed_ic"]
            standard_error = statistics["hac_standard_error"]
            if mean_ic is None or standard_error is None:
                coefficient = None
                posterior = None
            else:
                coefficient = self._shrinkage_coefficient(
                    prior_variance,
                    float(standard_error),
                )
                posterior = max(float(mean_ic), 0.0) * coefficient
            statistics["shrinkage_coefficient"] = _finite_or_none(coefficient)
            statistics["posterior_directed_ic"] = _finite_or_none(posterior)
            if not statistics["exclusion_reasons"] and posterior is not None:
                if posterior > 0:
                    eligible.append(factor)
                else:
                    statistics["exclusion_reasons"].append("nonpositive_posterior_ic")

        if len(eligible) < self.settings.min_active_factors:
            raise InsufficientTrainingData(
                f"IC shrinkage has only {len(eligible)} eligible factors; requires "
                f"{self.settings.min_active_factors}"
            )
        if len(eligible) * self.settings.max_factor_weight < 1 - 1e-12:
            raise InsufficientTrainingData(
                "Eligible factors cannot satisfy the maximum factor-weight constraint"
            )

        raw_correlation = self._average_daily_correlation(sample, directed, eligible)
        psd_correlation = _nearest_correlation(raw_correlation)
        redundancy = psd_correlation * psd_correlation
        posterior_vector = np.asarray(
            [factor_statistics[factor]["posterior_directed_ic"] for factor in eligible],
            dtype=float,
        )
        churn_vector = np.asarray(
            [factor_statistics[factor]["score_churn"] for factor in eligible],
            dtype=float,
        )
        core_anchor, core_anchor_source, active_core = self._core_anchor_vector(
            eligible
        )
        previous, previous_source, dropped_mass, dropped_squared = (
            self._previous_weight_vector(
                eligible,
                initial=(
                    core_anchor
                    if self.settings.core_anchor_penalty > 0
                    else None
                ),
            )
        )
        optimized, optimization = self._solve(
            posterior=posterior_vector,
            redundancy=redundancy,
            churn=churn_vector,
            previous=previous,
            dropped_previous_weight=dropped_mass,
            dropped_previous_squared=dropped_squared,
            core_anchor=core_anchor,
        )

        weights = {factor: 0.0 for factor in names}
        for factor, weight in zip(eligible, optimized, strict=True):
            weights[factor] = float(weight)
        effective_previous = {
            factor: float(weight)
            for factor, weight in zip(eligible, previous, strict=True)
        }
        for factor in names:
            factor_statistics[factor]["eligible"] = factor in eligible
            factor_statistics[factor]["previous_weight"] = effective_previous.get(
                factor,
                float(self._previous_weights.get(factor, 0.0)),
            )
            factor_statistics[factor]["optimized_weight"] = weights[factor]

        self.factor_names = names
        self.factor_weights = weights
        comparison_weights = (
            self._previous_weights if self._has_inherited_state else effective_previous
        )
        union = set(weights) | set(comparison_weights)
        realized_turnover = float(
            sum(
                abs(weights.get(factor, 0.0) - comparison_weights.get(factor, 0.0))
                for factor in union
            )
        )
        core_set = set(self.settings.core_factors)
        core_weight = float(
            sum(weight for factor, weight in weights.items() if factor in core_set)
        )
        candidate_weight = float(sum(weights.values()) - core_weight)
        risk_contributions = self._risk_contributions(
            eligible,
            optimized,
            redundancy,
        )
        family_factors: dict[str, list[str]] = {}
        family_weights: dict[str, float] = {}
        for factor, weight in weights.items():
            family = self.families.get(factor)
            if family is None:
                continue
            family_factors.setdefault(family, []).append(factor)
            family_weights[family] = family_weights.get(family, 0.0) + weight
        parameters: dict[str, Any] = {
            "method": (
                "core_anchored_empirical_bayes_ic_shrinkage"
                if self.settings.core_anchor_penalty > 0
                else "empirical_bayes_ic_shrinkage_convex_synthesis"
            ),
            "as_of_date": str(as_of_date),
            "settings": asdict(self.settings),
            "factor_statistics": factor_statistics,
            "prior_variance_raw": prior_variance_raw,
            "prior_variance": prior_variance,
            "eligible_factors": eligible,
            "raw_correlation": _matrix_payload(eligible, raw_correlation),
            "psd_correlation": _matrix_payload(eligible, psd_correlation),
            "redundancy_penalty_matrix": _matrix_payload(eligible, redundancy),
            "previous_weight_source": previous_source,
            "dropped_previous_weight": dropped_mass,
            "dropped_previous_squared": dropped_squared,
            "realized_factor_weight_l1_change": realized_turnover,
            "core_anchor_source": core_anchor_source,
            "active_core_factors": list(active_core),
            "core_anchor_weights": {
                factor: float(weight)
                for factor, weight in zip(eligible, core_anchor, strict=True)
            },
            "core_weight": core_weight,
            "candidate_weight": candidate_weight,
            "family_factors": family_factors,
            "family_weights": family_weights,
            "weights": weights,
            "signed_weights": {
                factor: weight * self.directions[factor]
                for factor, weight in weights.items()
            },
            "weight_concentration": float(np.square(optimized).sum()),
            "effective_factor_count": float(1.0 / np.square(optimized).sum()),
            "factor_risk_contributions": risk_contributions,
            "maximum_factor_risk_contribution": max(
                risk_contributions.values(),
                default=0.0,
            ),
            "optimization": optimization,
        }
        return ModelFitSummary(
            observations=len(sample),
            decision_dates=decision_dates,
            parameters=parameters,
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self.factor_names or not self.factor_weights:
            raise RuntimeError("IC shrinkage model must be fitted before prediction")
        active = tuple(
            factor
            for factor in self.factor_names
            if self.factor_weights[factor] > self.settings.feasibility_tolerance
        )
        if not active:
            raise RuntimeError("IC shrinkage model has no active factors")
        columns = [f"{factor}__z" for factor in active]
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        directed = matrix.mul([self.directions[factor] for factor in active], axis=1)
        weight_vector = pd.Series(
            [self.factor_weights[factor] for factor in active],
            index=columns,
            dtype=float,
        )
        available = directed.notna()
        numerator = directed.fillna(0.0).mul(weight_vector, axis=1).sum(axis=1)
        denominator = available.mul(weight_vector, axis=1).sum(axis=1)
        minimum = max(1, int(np.ceil(len(active) * self.settings.min_factor_fraction)))
        valid = available.sum(axis=1).ge(minimum) & denominator.gt(0)
        score = numerator.div(denominator).where(valid)
        return score.rename("score")

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
            raise ConfigurationError(f"IC shrinkage training data lacks columns: {missing}")
        if training.duplicated(["decision_date", "instrument"]).any():
            raise ConfigurationError(
                "IC shrinkage training decision_date/instrument key is not unique"
            )
        decision_date = training["decision_date"].astype(str)
        label_available = training["label_available_date"]
        visible = decision_date.lt(str(as_of_date))
        visible &= label_available.notna()
        visible &= label_available.fillna("").astype(str).lt(str(as_of_date))
        numeric_label = pd.to_numeric(training[label_column], errors="coerce")
        visible &= np.isfinite(numeric_label)
        sample = training.loc[visible].copy()
        dates = sorted(sample["decision_date"].astype(str).unique())
        recent_dates = set(dates[-self.settings.lookback_dates :])
        return sample.loc[sample["decision_date"].astype(str).isin(recent_dates)].copy()

    def _daily_ic(
        self,
        sample: pd.DataFrame,
        score: pd.Series,
        labels: pd.Series,
    ) -> pd.Series:
        values: list[float] = []
        for _, indices in sample.groupby("decision_date", sort=True).groups.items():
            cross_section = pd.DataFrame(
                {"score": score.loc[indices], "label": labels.loc[indices]}
            ).dropna()
            if len(cross_section) < self.settings.min_cross_section:
                continue
            if cross_section["score"].nunique() < 2 or cross_section["label"].nunique() < 2:
                continue
            rank_ic = float(cross_section["score"].corr(cross_section["label"], method="spearman"))
            if np.isfinite(rank_ic):
                values.append(rank_ic)
        return pd.Series(values, dtype=float)

    def _daily_churn(self, sample: pd.DataFrame, score: pd.Series) -> pd.Series:
        values: list[float] = []
        previous: pd.Series | None = None
        for _, indices in sample.groupby("decision_date", sort=True).groups.items():
            current = pd.Series(
                score.loc[indices].to_numpy(dtype=float),
                index=sample.loc[indices, "instrument"].astype(str),
                dtype=float,
            ).dropna()
            current = current.rank(method="average", pct=True)
            if previous is not None:
                common = previous.index.intersection(current.index)
                if len(common) >= self.settings.min_cross_section:
                    values.append(
                        float((current.loc[common] - previous.loc[common]).abs().median())
                    )
            previous = current
        return pd.Series(values, dtype=float)

    def _average_daily_correlation(
        self,
        sample: pd.DataFrame,
        directed: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> np.ndarray:
        names = tuple(factor_names)
        total = np.zeros((len(names), len(names)), dtype=float)
        counts = np.zeros_like(total)
        for _, indices in sample.groupby("decision_date", sort=True).groups.items():
            matrix = directed.loc[indices, list(names)].corr(
                method="spearman",
                min_periods=self.settings.min_cross_section,
            ).to_numpy(dtype=float)
            finite = np.isfinite(matrix)
            total[finite] += matrix[finite]
            counts[finite] += 1.0
        average = np.divide(
            total,
            counts,
            out=np.zeros_like(total),
            where=counts > 0,
        )
        average = (average + average.T) / 2.0
        np.fill_diagonal(average, 1.0)
        return np.clip(average, -1.0, 1.0)

    def _shrinkage_coefficient(self, prior_variance: float, standard_error: float) -> float:
        if not self.settings.shrinkage_enabled:
            return 1.0
        denominator = prior_variance + standard_error**2
        return 1.0 if denominator <= 0 else float(prior_variance / denominator)

    def _previous_weight_vector(
        self,
        eligible: Sequence[str],
        *,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, float, float]:
        if not self._has_inherited_state:
            if initial is not None:
                return (
                    np.asarray(initial, dtype=float).copy(),
                    "core_anchor_initialization",
                    0.0,
                    0.0,
                )
            return (
                np.full(len(eligible), 1.0 / len(eligible), dtype=float),
                "equal_weight_initialization",
                0.0,
                0.0,
            )
        eligible_set = set(eligible)
        dropped = np.asarray(
            [
                weight
                for factor, weight in self._previous_weights.items()
                if factor not in eligible_set
            ],
            dtype=float,
        )
        return (
            np.asarray(
                [self._previous_weights.get(factor, 0.0) for factor in eligible],
                dtype=float,
            ),
            "prior_model",
            float(dropped.sum()) if len(dropped) else 0.0,
            float(np.square(dropped).sum()) if len(dropped) else 0.0,
        )

    def _core_anchor_vector(
        self,
        eligible: Sequence[str],
    ) -> tuple[np.ndarray, str, tuple[str, ...]]:
        anchor = np.zeros(len(eligible), dtype=float)
        if self.settings.core_anchor_penalty <= 0:
            return anchor, "disabled", ()
        core_set = set(self.settings.core_factors)
        active_core = tuple(factor for factor in eligible if factor in core_set)
        if not active_core:
            raise InsufficientTrainingData(
                "IC shrinkage has no eligible core factor for the active soft anchor"
            )
        positions = {factor: position for position, factor in enumerate(eligible)}
        if self.settings.core_anchor_mode == "equal_factor":
            for factor in active_core:
                anchor[positions[factor]] = 1.0 / len(active_core)
            return anchor, "equal_factor_core", active_core

        grouped: dict[str, list[str]] = {}
        for factor in active_core:
            grouped.setdefault(self.families[factor], []).append(factor)
        family_weight = 1.0 / len(grouped)
        for factors in grouped.values():
            factor_weight = family_weight / len(factors)
            for factor in factors:
                anchor[positions[factor]] = factor_weight
        return anchor, "equal_family_core", active_core

    @staticmethod
    def _risk_contributions(
        eligible: Sequence[str],
        weights: np.ndarray,
        risk_matrix: np.ndarray,
    ) -> dict[str, float]:
        total = float(weights @ risk_matrix @ weights)
        if total <= 1e-16:
            return {factor: 0.0 for factor in eligible}
        marginal = risk_matrix @ weights
        contributions = weights * marginal / total
        return {
            factor: float(value)
            for factor, value in zip(eligible, contributions, strict=True)
        }

    def _solve(
        self,
        *,
        posterior: np.ndarray,
        redundancy: np.ndarray,
        churn: np.ndarray,
        previous: np.ndarray,
        dropped_previous_weight: float,
        dropped_previous_squared: float,
        core_anchor: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        weights = cp.Variable(len(posterior))
        return_term = posterior @ weights
        correlation_term = cp.quad_form(weights, cp.psd_wrap(redundancy))
        cost_term = churn @ weights
        if self.settings.weight_change_norm == "l1":
            stability_term = (
                cp.norm1(weights - previous) + dropped_previous_weight
            )
        else:
            stability_term = (
                cp.sum_squares(weights - previous) + dropped_previous_squared
            )
        core_anchor_term = cp.sum_squares(weights - core_anchor)
        objective = cp.Maximize(
            return_term
            - self.settings.correlation_penalty * correlation_term
            - self.settings.cost_penalty * cost_term
            - self.settings.weight_turnover_penalty * stability_term
            - self.settings.core_anchor_penalty * core_anchor_term
        )
        problem = cp.Problem(
            objective,
            [
                weights >= 0,
                cp.sum(weights) == 1,
                weights <= self.settings.max_factor_weight,
            ],
        )
        attempts: list[dict[str, str]] = []
        for solver in self.settings.solvers:
            try:
                problem.solve(solver=solver, warm_start=False, verbose=False)
                attempts.append({"solver": solver, "status": str(problem.status)})
            except cp.error.SolverError as exc:
                attempts.append(
                    {
                        "solver": solver,
                        "status": "solver_error",
                        "error": str(exc),
                    }
                )
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise InsufficientTrainingData(
                f"IC shrinkage optimization failed: {attempts}"
            )
        if weights.value is None:
            raise InsufficientTrainingData("IC shrinkage optimizer returned no weights")
        optimized = np.asarray(weights.value, dtype=float).reshape(-1)
        if len(optimized) != len(posterior):
            raise InsufficientTrainingData(
                "IC shrinkage optimizer returned the wrong number of weights"
            )
        if not np.isfinite(optimized).all():
            raise InsufficientTrainingData("IC shrinkage optimizer returned non-finite weights")
        tolerance = self.settings.feasibility_tolerance
        if (
            abs(float(optimized.sum()) - 1.0) > tolerance
            or float(optimized.min()) < -tolerance
            or float(optimized.max()) > self.settings.max_factor_weight + tolerance
        ):
            raise InsufficientTrainingData(
                "IC shrinkage optimizer returned constraint-violating weights"
            )
        optimized = np.maximum(optimized, 0.0)
        optimized /= optimized.sum()
        if float(optimized.max()) > self.settings.max_factor_weight + tolerance:
            raise InsufficientTrainingData(
                "IC shrinkage normalized weights violate the maximum-weight constraint"
            )
        correlation_value = float(optimized @ redundancy @ optimized)
        cost_value = float(churn @ optimized)
        stability_value = (
            float(np.abs(optimized - previous).sum() + dropped_previous_weight)
            if self.settings.weight_change_norm == "l1"
            else float(
                np.square(optimized - previous).sum()
                + dropped_previous_squared
            )
        )
        core_anchor_value = float(np.square(optimized - core_anchor).sum())
        diagnostics: dict[str, Any] = {
            "solver": str(problem.solver_stats.solver_name),
            "status": str(problem.status),
            "attempts": attempts,
            "objective_value": _finite_or_none(problem.value),
            "components": {
                "posterior_ic_contribution": float(posterior @ optimized),
                "correlation_redundancy": correlation_value,
                "correlation_penalty": self.settings.correlation_penalty
                * correlation_value,
                "score_churn": cost_value,
                "cost_penalty": self.settings.cost_penalty * cost_value,
                "weight_change_norm": self.settings.weight_change_norm,
                "weight_change": stability_value,
                "weight_turnover_penalty": self.settings.weight_turnover_penalty
                * stability_value,
                "core_anchor_deviation_squared": core_anchor_value,
                "core_anchor_penalty": self.settings.core_anchor_penalty
                * core_anchor_value,
            },
            "constraint_residuals": {
                "sum_to_one": abs(float(optimized.sum()) - 1.0),
                "nonnegative_weight_violation": max(-float(optimized.min()), 0.0),
                "maximum_weight_excess": max(
                    float(optimized.max()) - self.settings.max_factor_weight,
                    0.0,
                ),
            },
        }
        return optimized, diagnostics


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), 1e-12))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) / 2.0
    np.fill_diagonal(projected, 1.0)
    return np.clip(projected, -1.0, 1.0)


def _matrix_payload(names: Sequence[str], matrix: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        left: {
            right: float(matrix[row, column])
            for column, right in enumerate(names)
        }
        for row, left in enumerate(names)
    }


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)
