from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from csi500_alpha.errors import ConfigurationError
from csi500_alpha.research.diagnostics import newey_west_tstat
from csi500_alpha.workflow.contracts import SelectionResult


@dataclass(frozen=True)
class StabilityCostSettings:
    min_coverage: float = 0.70
    min_cross_section: int = 30
    min_ic_dates: int = 24
    min_mean_directed_ic: float = 0.0
    min_direction_consistency: float = 0.50
    segments: int = 4
    min_segment_selection_frequency: float = 0.50
    min_newey_west_t: float = 0.0
    min_quintile_monotonicity: float = 0.0
    max_score_churn: float = 0.35
    max_abs_correlation: float = 0.85
    min_factors: int = 4
    max_factors: int = 10
    max_per_family: int = 2
    lookback_dates: int = 156
    churn_penalty: float = 0.05

    def validate(self) -> None:
        proportions = {
            "min_coverage": self.min_coverage,
            "min_direction_consistency": self.min_direction_consistency,
            "min_segment_selection_frequency": self.min_segment_selection_frequency,
            "max_score_churn": self.max_score_churn,
            "max_abs_correlation": self.max_abs_correlation,
        }
        if any(not 0 <= value <= 1 for value in proportions.values()):
            raise ConfigurationError(
                "Stability/cost selector proportions must be in [0, 1]"
            )
        positive = {
            "min_cross_section": self.min_cross_section,
            "min_ic_dates": self.min_ic_dates,
            "segments": self.segments,
            "min_factors": self.min_factors,
            "max_factors": self.max_factors,
            "max_per_family": self.max_per_family,
            "lookback_dates": self.lookback_dates,
        }
        if any(value < 1 for value in positive.values()):
            raise ConfigurationError(
                "Stability/cost selector counts must be positive"
            )
        if self.min_factors > self.max_factors:
            raise ConfigurationError(
                "selector min_factors cannot exceed max_factors"
            )
        if not -1 <= self.min_quintile_monotonicity <= 1:
            raise ConfigurationError(
                "selector min_quintile_monotonicity must be in [-1, 1]"
            )
        if self.churn_penalty < 0:
            raise ConfigurationError("selector churn_penalty cannot be negative")


class StabilityCostSelector:
    """Select stable, directionally consistent and implementable factors."""

    name = "stability_cost"

    def __init__(
        self,
        *,
        directions: Mapping[str, int],
        families: Mapping[str, str],
        settings: StabilityCostSettings,
        label_column: str = "forward_active_return",
    ) -> None:
        settings.validate()
        self.directions = {str(name): int(value) for name, value in directions.items()}
        invalid_directions = sorted(
            name for name, value in self.directions.items() if value not in {-1, 1}
        )
        if invalid_directions:
            raise ConfigurationError(
                "Stability/cost selector directions must be -1 or 1: "
                f"{invalid_directions}"
            )
        self.families = {str(name): str(value) for name, value in families.items()}
        self.settings = settings
        self.label_column = label_column

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        names = tuple(dict.fromkeys(str(name) for name in candidates))
        if not names:
            raise ConfigurationError("Stability/cost selector received no candidates")
        missing_directions = sorted(set(names).difference(self.directions))
        if missing_directions:
            raise ConfigurationError(
                f"Stability/cost selector lacks directions: {missing_directions}"
            )
        required = {
            "decision_date",
            "instrument",
            "label_available_date",
            self.label_column,
            *(f"{factor}__z" for factor in names),
        }
        missing_columns = sorted(required.difference(training.columns))
        if missing_columns:
            raise ConfigurationError(
                f"Stability/cost training panel lacks columns: {missing_columns}"
            )
        sample = self._visible_sample(training, as_of_date)
        statistics = {
            factor: self._factor_statistics(sample, factor)
            for factor in names
        }
        ordered = sorted(
            names,
            key=lambda factor: (
                -_ranking_value(statistics[factor]["ranking_score"]),
                factor,
            ),
        )
        correlation = sample[
            [f"{factor}__z" for factor in names]
        ].corr(method="spearman")
        selected: list[str] = []
        family_counts: dict[str, int] = {}
        selection_notes: dict[str, list[str]] = {factor: [] for factor in names}

        for factor in ordered:
            hard_failures = statistics[factor]["hard_failures"]
            if hard_failures:
                selection_notes[factor].extend(str(value) for value in hard_failures)
                continue
            family = self._family(factor)
            if family_counts.get(family, 0) >= self.settings.max_per_family:
                selection_notes[factor].append(f"family_cap={family}")
                continue
            redundant = self._correlated_representative(
                factor,
                selected,
                correlation,
            )
            if redundant is not None:
                selection_notes[factor].append(f"correlated_with={redundant}")
                continue
            selected.append(factor)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= self.settings.max_factors:
                break

        fallback_used = len(selected) < min(self.settings.min_factors, len(names))
        if fallback_used:
            for factor in ordered:
                if factor in selected:
                    continue
                selected.append(factor)
                selection_notes[factor].append("selected_by_min_factor_fallback")
                if len(selected) >= min(self.settings.min_factors, len(names)):
                    break

        for factor in names:
            for failure in statistics[factor]["hard_failures"]:
                if failure not in selection_notes[factor]:
                    selection_notes[factor].append(str(failure))
            if factor not in selected and not selection_notes[factor]:
                selection_notes[factor].append("max_factor_limit")
        factor_diagnostics = {
            factor: {
                **{
                    key: _json_value(value)
                    for key, value in statistics[factor].items()
                    if key != "hard_failures"
                },
                "family": self._family(factor),
                "selected": factor in selected,
                "status": (
                    "fallback_selected"
                    if "selected_by_min_factor_fallback" in selection_notes[factor]
                    else "selected"
                    if factor in selected
                    else "rejected"
                ),
                "reasons": selection_notes[factor],
            }
            for factor in names
        }
        return SelectionResult(
            factor_names=tuple(selected[: self.settings.max_factors]),
            diagnostics={
                "candidate_count": len(names),
                "selected_count": min(len(selected), self.settings.max_factors),
                "training_rows_seen": len(sample),
                "training_dates_seen": int(sample["decision_date"].nunique()),
                "as_of_date": str(as_of_date),
                "fallback_used": fallback_used,
                "selected_factors": selected[: self.settings.max_factors],
                "settings": self.settings.__dict__,
                "factor_diagnostics": factor_diagnostics,
            },
        )

    def _visible_sample(
        self,
        training: pd.DataFrame,
        as_of_date: str,
    ) -> pd.DataFrame:
        sample = training.copy()
        visible = sample["decision_date"].astype(str) < str(as_of_date)
        visible &= sample["label_available_date"].notna()
        visible &= sample["label_available_date"].fillna("").astype(str) < str(
            as_of_date
        )
        visible &= pd.to_numeric(sample[self.label_column], errors="coerce").notna()
        sample = sample.loc[visible].copy()
        dates = sorted(sample["decision_date"].astype(str).unique())
        if len(dates) > self.settings.lookback_dates:
            sample = sample[
                sample["decision_date"].astype(str).isin(
                    dates[-self.settings.lookback_dates :]
                )
            ].copy()
        return sample

    def _factor_statistics(
        self,
        sample: pd.DataFrame,
        factor: str,
    ) -> dict[str, Any]:
        column = f"{factor}__z"
        direction = self.directions[factor]
        numeric_score = pd.to_numeric(sample[column], errors="coerce")
        numeric_label = pd.to_numeric(sample[self.label_column], errors="coerce")
        coverage = float(numeric_score.notna().mean()) if len(sample) else 0.0
        working = sample[["decision_date", "instrument"]].copy()
        working["score"] = numeric_score
        working["label"] = numeric_label
        ic_values: list[float] = []
        quintile_rows: list[pd.Series] = []
        previous_rank: pd.Series | None = None
        churn_values: list[float] = []
        for _, frame in working.groupby("decision_date", sort=True):
            valid = frame[["score", "label"]].dropna()
            if (
                len(valid) >= self.settings.min_cross_section
                and valid["score"].nunique() > 1
                and valid["label"].nunique() > 1
            ):
                rank_ic = valid["score"].corr(valid["label"], method="spearman")
                if np.isfinite(rank_ic):
                    ic_values.append(direction * float(rank_ic))
                directed_score = direction * valid["score"]
                quintiles = np.ceil(
                    directed_score.rank(method="first", pct=True) * 5.0
                ).clip(1, 5).astype(int)
                quintile_rows.append(valid.groupby(quintiles)["label"].mean())
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
                            ).abs().median()
                        )
                    )
            previous_rank = current_rank

        ic_series = pd.Series(ic_values, dtype=float)
        mean_ic = float(ic_series.mean()) if not ic_series.empty else np.nan
        consistency = (
            float(ic_series.gt(0.0).mean()) if not ic_series.empty else np.nan
        )
        segment_means = self._segment_means(ic_series)
        segment_frequency = (
            float(
                np.mean(
                    np.asarray(segment_means, dtype=float)
                    >= self.settings.min_mean_directed_ic
                )
            )
            if segment_means
            else np.nan
        )
        nw_t = newey_west_tstat(ic_series)
        churn = float(np.mean(churn_values)) if churn_values else np.nan
        monotonicity = self._quintile_monotonicity(quintile_rows)
        ranking_score = self._ranking_score(
            mean_ic=mean_ic,
            newey_west_t=nw_t,
            segment_frequency=segment_frequency,
            monotonicity=monotonicity,
            churn=churn,
        )
        failures: list[str] = []
        if coverage < self.settings.min_coverage:
            failures.append("coverage_below_minimum")
        if len(ic_series) < self.settings.min_ic_dates:
            failures.append("insufficient_ic_dates")
        if not np.isfinite(mean_ic) or mean_ic < self.settings.min_mean_directed_ic:
            failures.append("mean_directed_ic_below_minimum")
        if (
            not np.isfinite(consistency)
            or consistency < self.settings.min_direction_consistency
        ):
            failures.append("direction_consistency_below_minimum")
        if (
            not np.isfinite(segment_frequency)
            or segment_frequency < self.settings.min_segment_selection_frequency
        ):
            failures.append("segment_selection_frequency_below_minimum")
        if not np.isfinite(nw_t) or nw_t < self.settings.min_newey_west_t:
            failures.append("newey_west_t_below_minimum")
        if (
            not np.isfinite(monotonicity)
            or monotonicity < self.settings.min_quintile_monotonicity
        ):
            failures.append("quintile_monotonicity_below_minimum")
        if np.isfinite(churn) and churn > self.settings.max_score_churn:
            failures.append("score_churn_above_maximum")
        return {
            "coverage": coverage,
            "ic_dates": len(ic_series),
            "mean_directed_ic": mean_ic,
            "direction_consistency": consistency,
            "newey_west_t": nw_t,
            "segment_means": segment_means,
            "segment_selection_frequency": segment_frequency,
            "quintile_monotonicity": monotonicity,
            "mean_score_churn": churn,
            "ranking_score": ranking_score,
            "hard_failures": failures,
        }

    def _segment_means(self, values: pd.Series) -> list[float]:
        if values.empty:
            return []
        count = min(self.settings.segments, len(values))
        return [
            float(values.iloc[index].mean())
            for index in np.array_split(np.arange(len(values)), count)
            if len(index)
        ]

    @staticmethod
    def _quintile_monotonicity(quintile_rows: Sequence[pd.Series]) -> float:
        if not quintile_rows:
            return np.nan
        means = (
            pd.concat(quintile_rows, axis=1)
            .mean(axis=1)
            .reindex(range(1, 6))
        )
        if means.notna().sum() < 3:
            return np.nan
        return float(means.corr(pd.Series(means.index, index=means.index), method="spearman"))

    def _ranking_score(
        self,
        *,
        mean_ic: float,
        newey_west_t: float,
        segment_frequency: float,
        monotonicity: float,
        churn: float,
    ) -> float:
        if not np.isfinite(mean_ic):
            return -np.inf
        t_component = float(np.clip(newey_west_t, -5.0, 5.0)) if np.isfinite(newey_west_t) else -5.0
        segment_component = segment_frequency if np.isfinite(segment_frequency) else 0.0
        monotonicity_component = monotonicity if np.isfinite(monotonicity) else -1.0
        churn_component = churn if np.isfinite(churn) else 1.0
        return float(
            mean_ic
            + 0.002 * t_component
            + 0.010 * segment_component
            + 0.005 * monotonicity_component
            - self.settings.churn_penalty * churn_component
        )

    def _correlated_representative(
        self,
        factor: str,
        selected: Sequence[str],
        correlation: pd.DataFrame,
    ) -> str | None:
        column = f"{factor}__z"
        for kept in selected:
            value = correlation.loc[column, f"{kept}__z"]
            if np.isfinite(value) and abs(float(value)) > self.settings.max_abs_correlation:
                return kept
        return None

    def _family(self, factor: str) -> str:
        return self.families.get(factor, f"unclassified:{factor}")


def _ranking_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -np.inf
    return numeric if np.isfinite(numeric) else -np.inf


def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value
