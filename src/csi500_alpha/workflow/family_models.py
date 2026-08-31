from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.workflow.contracts import ModelFitSummary

FamilyMethod = Literal["direction_equal", "ridge", "robust_ic"]


@dataclass(frozen=True)
class FamilyModelSettings:
    min_cross_section: int = 100
    min_training_rows: int = 1_000
    min_training_dates: int = 52
    min_ic_dates: int = 52
    lookback_dates: int = 156
    ridge_alpha: float = 10.0
    min_factor_fraction: float = 0.50
    min_family_fraction: float = 0.50
    min_active_factors: int = 5
    min_active_families: int = 3
    max_factor_weight: float = 0.20
    max_family_weight: float = 0.35

    def validate(self) -> None:
        counts = {
            "min_cross_section": self.min_cross_section,
            "min_training_rows": self.min_training_rows,
            "min_training_dates": self.min_training_dates,
            "min_ic_dates": self.min_ic_dates,
            "lookback_dates": self.lookback_dates,
            "min_active_factors": self.min_active_factors,
            "min_active_families": self.min_active_families,
        }
        if any(value < 1 for value in counts.values()):
            raise ConfigurationError("Family-model counts must be positive")
        if self.lookback_dates < max(self.min_training_dates, self.min_ic_dates):
            raise ConfigurationError(
                "Family-model lookback cannot be shorter than required dates"
            )
        proportions = {
            "min_factor_fraction": self.min_factor_fraction,
            "min_family_fraction": self.min_family_fraction,
            "max_factor_weight": self.max_factor_weight,
            "max_family_weight": self.max_family_weight,
        }
        if any(not 0 < value <= 1 for value in proportions.values()):
            raise ConfigurationError(
                "Family-model fractions and weight caps must be in (0, 1]"
            )
        if self.ridge_alpha < 0:
            raise ConfigurationError("Family-model ridge_alpha cannot be negative")
        if self.min_active_factors * self.max_factor_weight < 1.0 - 1e-12:
            raise ConfigurationError(
                "Family-model minimum factor count and factor cap cannot provide "
                "unit weight"
            )
        if self.min_active_families * self.max_family_weight < 1.0 - 1e-12:
            raise ConfigurationError(
                "Family-model minimum family count and family cap cannot provide "
                "unit weight"
            )


class FamilyAlphaModel:
    """Aggregate factors within economic families before assigning family weights."""

    def __init__(
        self,
        *,
        method: FamilyMethod,
        directions: Mapping[str, int],
        families: Mapping[str, str],
        settings: FamilyModelSettings,
    ) -> None:
        if method not in {"direction_equal", "ridge", "robust_ic"}:
            raise ConfigurationError(f"Unknown family-model method: {method}")
        settings.validate()
        invalid = sorted(
            factor for factor, direction in directions.items() if direction not in {-1, 1}
        )
        if invalid:
            raise ConfigurationError(
                f"Family-model directions must be -1 or 1: {invalid}"
            )
        self.method = method
        self.name = f"family_{method}"
        self.directions = {str(key): int(value) for key, value in directions.items()}
        self.families = {str(key): str(value) for key, value in families.items()}
        self.settings = settings
        self.factor_names: tuple[str, ...] = ()
        self.family_factors: dict[str, tuple[str, ...]] = {}
        self.family_weights: dict[str, float] = {}
        self.factor_weights: dict[str, float] = {}

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
            raise ConfigurationError("Family-model factor names must be unique")
        missing_directions = sorted(set(names).difference(self.directions))
        missing_families = sorted(set(names).difference(self.families))
        if missing_directions:
            raise ConfigurationError(
                f"Family model lacks directions: {missing_directions}"
            )
        if missing_families:
            raise ConfigurationError(f"Family model lacks families: {missing_families}")
        if len(names) < self.settings.min_active_factors:
            raise InsufficientTrainingData(
                "Family model requires at least "
                f"{self.settings.min_active_factors} factors; received {len(names)}"
            )

        family_factors = _group_factors(names, self.families)
        if len(family_factors) < self.settings.min_active_families:
            raise InsufficientTrainingData(
                "Family model requires at least "
                f"{self.settings.min_active_families} families; received "
                f"{len(family_factors)}"
            )
        sample = self._mature_training(
            training,
            names,
            label_column=label_column,
            as_of_date=as_of_date,
        )
        decision_dates = int(sample["decision_date"].nunique())
        if (
            len(sample) < self.settings.min_training_rows
            or decision_dates < self.settings.min_training_dates
        ):
            raise InsufficientTrainingData(
                "Family model requires at least "
                f"{self.settings.min_training_rows} rows and "
                f"{self.settings.min_training_dates} dates; received "
                f"{len(sample)} rows and {decision_dates} dates"
            )

        family_scores = self._family_scores(sample, family_factors)
        labels = pd.to_numeric(sample[label_column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        strengths, method_statistics, observations = self._strengths(
            sample,
            family_scores,
            labels,
        )
        eligible = tuple(
            family for family in family_factors if strengths.get(family, 0.0) > 0.0
        )
        if len(eligible) < self.settings.min_active_families:
            raise InsufficientTrainingData(
                "Family model has only "
                f"{len(eligible)} usable families; requires "
                f"{self.settings.min_active_families}"
            )
        capacities = {
            family: min(
                self.settings.max_family_weight,
                len(family_factors[family]) * self.settings.max_factor_weight,
            )
            for family in eligible
        }
        family_weights = _allocate_capped_weights(strengths, capacities, eligible)
        factor_weights = {
            factor: family_weights[family] / len(factors)
            for family, factors in family_factors.items()
            for factor in factors
            if family in family_weights
        }
        if max(factor_weights.values(), default=0.0) > (
            self.settings.max_factor_weight + 1e-12
        ):
            raise RuntimeError("Family-model factor cap projection failed")
        if max(family_weights.values(), default=0.0) > (
            self.settings.max_family_weight + 1e-12
        ):
            raise RuntimeError("Family-model family cap projection failed")

        self.factor_names = names
        self.family_factors = family_factors
        self.family_weights = family_weights
        self.factor_weights = factor_weights
        family_concentration = float(
            sum(weight * weight for weight in family_weights.values())
        )
        factor_concentration = float(
            sum(weight * weight for weight in factor_weights.values())
        )
        return ModelFitSummary(
            observations=observations,
            decision_dates=decision_dates,
            parameters={
                "method": self.method,
                "as_of_date": str(as_of_date),
                "settings": asdict(self.settings),
                "family_factors": {
                    family: list(factors)
                    for family, factors in family_factors.items()
                },
                "method_statistics": method_statistics,
                "eligible_families": list(eligible),
                "family_capacities": capacities,
                "family_weights": family_weights,
                "factor_weights": factor_weights,
                "signed_factor_weights": {
                    factor: weight * self.directions[factor]
                    for factor, weight in factor_weights.items()
                },
                "effective_family_count": (
                    1.0 / family_concentration if family_concentration > 0 else 0.0
                ),
                "effective_factor_count": (
                    1.0 / factor_concentration if factor_concentration > 0 else 0.0
                ),
            },
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self.family_weights or not self.family_factors:
            raise RuntimeError("Family model must be fitted before prediction")
        family_scores = self._family_scores(frame, self.family_factors)
        active = tuple(self.family_weights)
        available = family_scores[list(active)].notna()
        weight_vector = pd.Series(self.family_weights, dtype=float).reindex(active)
        numerator = (
            family_scores[list(active)]
            .fillna(0.0)
            .mul(weight_vector, axis="columns")
            .sum(axis=1)
        )
        available_weight = available.mul(weight_vector, axis="columns").sum(axis=1)
        required_families = max(
            1,
            int(np.ceil(len(active) * self.settings.min_family_fraction)),
        )
        valid = available.sum(axis=1).ge(required_families)
        valid &= available_weight.gt(0.0)
        return numerator.div(available_weight).where(valid).rename("score")

    def _strengths(
        self,
        sample: pd.DataFrame,
        family_scores: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        families = tuple(family_scores.columns)
        if self.method == "direction_equal":
            return (
                {family: 1.0 for family in families},
                {family: {"preference": 1.0} for family in families},
                len(sample),
            )
        if self.method == "ridge":
            return self._ridge_strengths(sample, family_scores, labels)
        return self._robust_ic_strengths(sample, family_scores, labels)

    def _ridge_strengths(
        self,
        sample: pd.DataFrame,
        family_scores: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        minimum = max(
            1,
            int(
                np.ceil(
                    len(family_scores.columns) * self.settings.min_family_fraction
                )
            ),
        )
        valid = labels.notna() & family_scores.notna().sum(axis=1).ge(minimum)
        fit_scores = family_scores.loc[valid].fillna(0.0)
        fit_labels = labels.loc[valid]
        dates = sample.loc[valid, "decision_date"].astype(str)
        if (
            len(fit_scores) < self.settings.min_training_rows
            or dates.nunique() < self.settings.min_training_dates
        ):
            raise InsufficientTrainingData(
                "Family ridge has insufficient jointly usable training data"
            )
        date_counts = dates.groupby(dates).transform("size")
        sample_weight = 1.0 / date_counts.to_numpy(dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        estimator = Ridge(
            alpha=self.settings.ridge_alpha,
            fit_intercept=True,
            positive=True,
        )
        estimator.fit(
            fit_scores.to_numpy(dtype=float),
            fit_labels.to_numpy(dtype=float),
            sample_weight=sample_weight,
        )
        coefficients = {
            family: max(float(value), 0.0)
            for family, value in zip(
                family_scores.columns,
                estimator.coef_,
                strict=True,
            )
        }
        statistics = {
            family: {"positive_ridge_coefficient": coefficients[family]}
            for family in family_scores.columns
        }
        statistics["_fit"] = {
            "intercept": float(estimator.intercept_),
            "observations": int(len(fit_scores)),
            "decision_dates": int(dates.nunique()),
        }
        return coefficients, statistics, len(fit_scores)

    def _robust_ic_strengths(
        self,
        sample: pd.DataFrame,
        family_scores: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[dict[str, float], dict[str, Any], int]:
        strengths: dict[str, float] = {}
        statistics: dict[str, Any] = {}
        for family in family_scores:
            ic_values: list[float] = []
            for _, indices in sample.groupby("decision_date", sort=True).groups.items():
                frame = pd.DataFrame(
                    {
                        "score": family_scores.loc[indices, family],
                        "label": labels.loc[indices],
                    }
                ).dropna()
                if (
                    len(frame) >= self.settings.min_cross_section
                    and frame["score"].nunique() > 1
                    and frame["label"].nunique() > 1
                ):
                    value = frame["score"].corr(frame["label"], method="spearman")
                    if np.isfinite(value):
                        ic_values.append(float(value))
            series = pd.Series(ic_values, dtype=float)
            median_ic = float(series.median()) if not series.empty else np.nan
            positive_frequency = (
                float(series.gt(0.0).mean()) if not series.empty else np.nan
            )
            strength = (
                max(median_ic, 0.0) * positive_frequency
                if len(series) >= self.settings.min_ic_dates
                and np.isfinite(median_ic)
                and np.isfinite(positive_frequency)
                else 0.0
            )
            strengths[family] = float(strength)
            statistics[family] = {
                "ic_dates": int(len(series)),
                "median_rank_ic": _finite_or_none(median_ic),
                "positive_ic_frequency": _finite_or_none(positive_frequency),
                "robust_ic_preference": float(strength),
            }
        return strengths, statistics, len(sample)

    def _family_scores(
        self,
        frame: pd.DataFrame,
        family_factors: Mapping[str, Sequence[str]],
    ) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for family, factors in family_factors.items():
            columns = [f"{factor}__z" for factor in factors]
            missing = sorted(set(columns).difference(frame.columns))
            if missing:
                raise ConfigurationError(
                    f"Family-model input lacks columns: {missing}"
                )
            scores = frame[columns].apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            directed = scores.mul(
                [self.directions[factor] for factor in factors],
                axis="columns",
            )
            minimum = max(
                1,
                int(np.ceil(len(factors) * self.settings.min_factor_fraction)),
            )
            output[family] = directed.mean(axis=1, skipna=True).where(
                directed.notna().sum(axis=1).ge(minimum)
            )
        return output

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
                f"Family-model training data lacks columns: {missing}"
            )
        if training.duplicated(["decision_date", "instrument"]).any():
            raise ConfigurationError(
                "Family-model decision_date/instrument key is not unique"
            )
        decision_date = training["decision_date"].astype(str)
        label_available = training["label_available_date"]
        visible = decision_date.lt(str(as_of_date))
        visible &= label_available.notna()
        visible &= label_available.fillna("").astype(str).lt(str(as_of_date))
        visible &= pd.to_numeric(training[label_column], errors="coerce").notna()
        sample = training.loc[visible].copy()
        dates = sorted(sample["decision_date"].astype(str).unique())
        recent = set(dates[-self.settings.lookback_dates :])
        return sample.loc[sample["decision_date"].astype(str).isin(recent)].copy()


def _group_factors(
    factor_names: Sequence[str],
    families: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for factor in factor_names:
        grouped.setdefault(families[factor], []).append(factor)
    return {
        family: tuple(factors)
        for family, factors in sorted(grouped.items())
    }


def _allocate_capped_weights(
    preferences: Mapping[str, float],
    capacities: Mapping[str, float],
    eligible: Sequence[str],
) -> dict[str, float]:
    names = tuple(eligible)
    positive = {
        name: max(float(preferences.get(name, 0.0)), 0.0) for name in names
    }
    active = [name for name in names if positive[name] > 0.0]
    if not active:
        raise InsufficientTrainingData("Family model has no positive family preference")
    if sum(float(capacities[name]) for name in active) < 1.0 - 1e-12:
        raise InsufficientTrainingData(
            "Usable family and factor caps cannot provide unit weight"
        )
    weights = {name: 0.0 for name in names}
    remaining = 1.0
    while active and remaining > 1e-12:
        preference_sum = sum(positive[name] for name in active)
        if preference_sum <= 0:
            raise InsufficientTrainingData(
                "Family model cannot allocate residual weight without evidence"
            )
        proposed = {
            name: remaining * positive[name] / preference_sum for name in active
        }
        saturated = [
            name
            for name in active
            if proposed[name] >= float(capacities[name]) - 1e-12
        ]
        if not saturated:
            for name in active:
                weights[name] = proposed[name]
            remaining = 0.0
            break
        for name in saturated:
            weights[name] = float(capacities[name])
            remaining -= weights[name]
            active.remove(name)
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("Family-model capped allocation did not sum to one")
    return {name: float(weight) for name, weight in weights.items() if weight > 0.0}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
