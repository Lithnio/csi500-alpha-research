from __future__ import annotations

import math
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
    min_active_date_rate: float = 0.80
    min_cross_section: int = 30
    min_ic_dates: int = 24
    min_mean_directed_ic: float = 0.0
    min_direction_consistency: float = 0.50
    segments: int = 4
    min_segment_selection_frequency: float = 0.50
    min_mean_net_quintile_spread: float = 0.0
    min_net_spread_consistency: float = 0.50
    min_joint_segment_frequency: float = 0.50
    min_newey_west_t: float = 0.0
    max_bh_q_value: float = 0.20
    min_quintile_monotonicity: float = 0.0
    max_score_churn: float = 0.35
    max_abs_correlation: float = 0.85
    min_factors: int = 4
    max_factors: int = 10
    max_per_family: int = 2
    max_per_cluster: int = 1
    lookback_dates: int = 156
    churn_penalty: float = 0.05
    linear_cost_bps: float = 5.0
    stamp_duty_change_date: str = "20230828"
    stamp_duty_before: float = 0.001
    stamp_duty_after: float = 0.0005
    excluded_factors: tuple[str, ...] = ()

    def validate(self) -> None:
        proportions = {
            "min_coverage": self.min_coverage,
            "min_active_date_rate": self.min_active_date_rate,
            "min_direction_consistency": self.min_direction_consistency,
            "min_segment_selection_frequency": (
                self.min_segment_selection_frequency
            ),
            "min_net_spread_consistency": self.min_net_spread_consistency,
            "min_joint_segment_frequency": self.min_joint_segment_frequency,
            "max_bh_q_value": self.max_bh_q_value,
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
            "max_per_cluster": self.max_per_cluster,
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
        if self.churn_penalty < 0 or self.linear_cost_bps < 0:
            raise ConfigurationError(
                "selector churn penalty and linear cost cannot be negative"
            )
        if self.stamp_duty_before < 0 or self.stamp_duty_after < 0:
            raise ConfigurationError("selector stamp duty cannot be negative")
        if len(set(self.excluded_factors)) != len(self.excluded_factors):
            raise ConfigurationError("selector excluded_factors cannot contain duplicates")
        if (
            len(self.stamp_duty_change_date) != 8
            or not self.stamp_duty_change_date.isdigit()
        ):
            raise ConfigurationError(
                "selector stamp_duty_change_date must be YYYYMMDD"
            )


@dataclass(frozen=True)
class _DailyFactorStatistics:
    row_count: int
    score_count: int
    directed_rank_ic: float
    valid_cross_section: bool
    quintile_means: pd.Series
    lower_members: frozenset[str]
    upper_members: frozenset[str]
    score_ranks: pd.Series
    trade_date: str


class StabilityCostSelector:
    """Select fold-local factors using stability and implementability evidence."""

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
        self.directions = {
            str(name): int(value) for name, value in directions.items()
        }
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
        self._daily_cache: dict[tuple[str, str], _DailyFactorStatistics] = {}
        self._cache_candidates: tuple[str, ...] | None = None
        self._last_as_of_date: str | None = None

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        provided_names = tuple(dict.fromkeys(str(name) for name in candidates))
        excluded = set(self.settings.excluded_factors)
        names = tuple(name for name in provided_names if name not in excluded)
        if not names:
            raise ConfigurationError("Stability/cost selector received no candidates")
        self._prepare_incremental_cache(names, str(as_of_date))
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
        sample_by_date = {
            str(decision_date): frame
            for decision_date, frame in sample.groupby("decision_date", sort=True)
        }
        visible_dates = set(sample_by_date)
        self._daily_cache = {
            key: value
            for key, value in self._daily_cache.items()
            if key[0] in visible_dates
        }
        statistics = {
            factor: self._factor_statistics(
                factor,
                by_date=sample_by_date,
            )
            for factor in names
        }
        q_values = _benjamini_hochberg(
            {
                factor: statistics[factor]["newey_west_p_value"]
                for factor in names
            }
        )
        for factor in names:
            q_value = q_values[factor]
            statistics[factor]["bh_q_value"] = q_value
            if (
                not np.isfinite(q_value)
                or q_value > self.settings.max_bh_q_value
            ):
                statistics[factor]["hard_failures"].append(
                    "bh_q_value_above_maximum"
                )

        ordered = sorted(
            names,
            key=lambda factor: (
                -_ranking_value(statistics[factor]["ranking_score"]),
                factor,
            ),
        )
        correlation = sample[[f"{factor}__z" for factor in names]].corr(
            method="spearman",
            min_periods=self.settings.min_cross_section,
        )
        cluster_by_factor, cluster_members = _correlation_clusters(
            names,
            correlation,
            threshold=self.settings.max_abs_correlation,
        )
        selected: list[str] = []
        family_counts: dict[str, int] = {}
        cluster_counts: dict[str, int] = {}
        cluster_representatives: dict[str, str] = {}
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
            cluster = cluster_by_factor[factor]
            if cluster_counts.get(cluster, 0) >= self.settings.max_per_cluster:
                representative = cluster_representatives[cluster]
                selection_notes[factor].extend(
                    [
                        f"correlation_cluster_cap={cluster}",
                        f"cluster_representative={representative}",
                    ]
                )
                direct_correlation = correlation.loc[
                    f"{factor}__z", f"{representative}__z"
                ]
                if (
                    np.isfinite(direct_correlation)
                    and abs(float(direct_correlation))
                    >= self.settings.max_abs_correlation
                ):
                    selection_notes[factor].append(
                        f"correlated_with={representative}"
                    )
                continue
            selected.append(factor)
            family_counts[family] = family_counts.get(family, 0) + 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            cluster_representatives.setdefault(cluster, factor)
            if len(selected) >= self.settings.max_factors:
                break

        evidence_selected = tuple(selected)
        selection_shortfall = len(evidence_selected) < self.settings.min_factors
        if selection_shortfall:
            for factor in evidence_selected:
                selection_notes[factor].append("selection_count_below_minimum")
            selected = []

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
                "correlation_cluster": cluster_by_factor[factor],
                "correlation_cluster_members": list(
                    cluster_members[cluster_by_factor[factor]]
                ),
                "selected": factor in selected,
                "status": "selected" if factor in selected else "rejected",
                "reasons": selection_notes[factor],
            }
            for factor in names
        }
        return SelectionResult(
            factor_names=tuple(selected),
            diagnostics={
                "candidate_count": len(names),
                "provided_candidate_count": len(provided_names),
                "excluded_factors": sorted(excluded.intersection(provided_names)),
                "evidence_selected_count": len(evidence_selected),
                "selected_count": len(selected),
                "training_rows_seen": len(sample),
                "training_dates_seen": int(sample["decision_date"].nunique()),
                "as_of_date": str(as_of_date),
                "fallback_used": False,
                "selection_shortfall": selection_shortfall,
                "selected_factors": selected,
                "evidence_selected_factors": list(evidence_selected),
                "settings": self.settings.__dict__,
                "correlation_clusters": {
                    cluster: list(members)
                    for cluster, members in cluster_members.items()
                },
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

    def _prepare_incremental_cache(
        self,
        candidates: tuple[str, ...],
        as_of_date: str,
    ) -> None:
        chronological = (
            self._last_as_of_date is None
            or str(as_of_date) > self._last_as_of_date
        )
        if self._cache_candidates != candidates or not chronological:
            self._daily_cache.clear()
        self._cache_candidates = candidates
        self._last_as_of_date = str(as_of_date)

    def _daily_factor_statistics(
        self,
        decision_date: str,
        frame: pd.DataFrame,
        factor: str,
    ) -> _DailyFactorStatistics:
        cache_key = (str(decision_date), factor)
        cached = self._daily_cache.get(cache_key)
        if cached is not None and cached.row_count == len(frame):
            return cached

        column = f"{factor}__z"
        columns = ["instrument"]
        if "label_entry_date" in frame.columns:
            columns.append("label_entry_date")
        working = frame[columns].copy()
        working["score"] = pd.to_numeric(frame[column], errors="coerce")
        working["label"] = pd.to_numeric(
            frame[self.label_column],
            errors="coerce",
        )
        valid = working[["instrument", "score", "label"]].dropna(
            subset=["score", "label"]
        )
        valid_cross_section = (
            len(valid) >= self.settings.min_cross_section
            and valid["score"].nunique() > 1
            and valid["label"].nunique() > 1
        )
        directed_rank_ic = np.nan
        quintile_means = pd.Series(dtype=float)
        lower_members: frozenset[str] = frozenset()
        upper_members: frozenset[str] = frozenset()
        if valid_cross_section:
            rank_ic = valid["score"].corr(valid["label"], method="spearman")
            if np.isfinite(rank_ic):
                directed_rank_ic = self.directions[factor] * float(rank_ic)
            directed_score = self.directions[factor] * valid["score"]
            quintiles = np.ceil(
                directed_score.rank(method="first", pct=True) * 5.0
            ).clip(1, 5).astype(int)
            quintile_means = valid.groupby(quintiles)["label"].mean()
            lower_members = frozenset(
                valid.loc[quintiles.eq(1), "instrument"].astype(str)
            )
            upper_members = frozenset(
                valid.loc[quintiles.eq(5), "instrument"].astype(str)
            )

        score_ranks = (
            working.dropna(subset=["score"])
            .set_index("instrument")["score"]
            .rank(method="average", pct=True)
        )
        result = _DailyFactorStatistics(
            row_count=len(working),
            score_count=int(working["score"].notna().sum()),
            directed_rank_ic=directed_rank_ic,
            valid_cross_section=valid_cross_section,
            quintile_means=quintile_means,
            lower_members=lower_members,
            upper_members=upper_members,
            score_ranks=score_ranks,
            trade_date=self._trade_date(working, str(decision_date)),
        )
        self._daily_cache[cache_key] = result
        return result

    def _factor_statistics(
        self,
        factor: str,
        *,
        by_date: Mapping[str, pd.DataFrame],
    ) -> dict[str, Any]:
        dates = sorted(by_date)
        daily = [
            self._daily_factor_statistics(
                decision_date,
                by_date[decision_date],
                factor,
            )
            for decision_date in dates
        ]
        row_count = sum(item.row_count for item in daily)
        score_count = sum(item.score_count for item in daily)
        coverage = score_count / row_count if row_count else 0.0
        active_dates = [
            item.score_count >= self.settings.min_cross_section
            and item.score_count / item.row_count >= self.settings.min_coverage
            for item in daily
            if item.row_count > 0
        ]
        active_date_rate = (
            float(np.mean(active_dates)) if active_dates else 0.0
        )

        ic_by_date: dict[str, float] = {}
        net_spread_by_date: dict[str, float] = {}
        gross_spread_by_date: dict[str, float] = {}
        quintile_rows: list[pd.Series] = []
        previous_rank: pd.Series | None = None
        previous_members: dict[int, frozenset[str]] = {}
        churn_values: list[float] = []
        turnover_values: list[float] = []
        linear_rate = self.settings.linear_cost_bps / 10_000.0

        for date, item in zip(dates, daily, strict=True):
            if item.valid_cross_section:
                if np.isfinite(item.directed_rank_ic):
                    ic_by_date[date] = item.directed_rank_ic
                quintile_rows.append(item.quintile_means)
                stamp_rate = (
                    self.settings.stamp_duty_before
                    if item.trade_date < self.settings.stamp_duty_change_date
                    else self.settings.stamp_duty_after
                )
                leg_returns: dict[int, float] = {}
                leg_costs: dict[int, float] = {}
                leg_turnover: dict[int, float] = {}
                for quintile, members in (
                    (1, item.lower_members),
                    (5, item.upper_members),
                ):
                    turnover, cost = _equal_weight_leg_turnover_and_cost(
                        members,
                        previous_members.get(quintile),
                        linear_rate=linear_rate,
                        stamp_rate=stamp_rate,
                    )
                    previous_members[quintile] = members
                    leg_returns[quintile] = float(item.quintile_means.loc[quintile])
                    leg_costs[quintile] = cost
                    leg_turnover[quintile] = turnover
                gross_spread = leg_returns[5] - leg_returns[1]
                gross_spread_by_date[date] = gross_spread
                net_spread_by_date[date] = (
                    gross_spread - leg_costs[1] - leg_costs[5]
                )
                turnover_values.append(
                    0.5 * (leg_turnover[1] + leg_turnover[5])
                )

            current_rank = item.score_ranks
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

        ic_series = pd.Series(ic_by_date, dtype=float).sort_index()
        net_spread = pd.Series(net_spread_by_date, dtype=float).sort_index()
        gross_spread = pd.Series(gross_spread_by_date, dtype=float).sort_index()
        mean_ic = float(ic_series.mean()) if not ic_series.empty else np.nan
        consistency = (
            float(ic_series.gt(0.0).mean()) if not ic_series.empty else np.nan
        )
        mean_net_spread = (
            float(net_spread.mean()) if not net_spread.empty else np.nan
        )
        net_consistency = (
            float(net_spread.gt(0.0).mean()) if not net_spread.empty else np.nan
        )
        (
            segment_means,
            net_segment_means,
            segment_frequency,
            joint_segment_frequency,
        ) = self._segment_statistics(ic_series, net_spread)
        nw_t = newey_west_tstat(ic_series)
        nw_p = _two_sided_normal_p_value(nw_t)
        churn = float(np.mean(churn_values)) if churn_values else np.nan
        mean_turnover = (
            float(np.mean(turnover_values)) if turnover_values else np.nan
        )
        monotonicity = self._quintile_monotonicity(quintile_rows)
        ranking_score = self._ranking_score(
            mean_ic=mean_ic,
            mean_net_spread=mean_net_spread,
            newey_west_t=nw_t,
            joint_segment_frequency=joint_segment_frequency,
            monotonicity=monotonicity,
            churn=churn,
        )
        failures: list[str] = []
        if coverage < self.settings.min_coverage:
            failures.append("coverage_below_minimum")
        if active_date_rate < self.settings.min_active_date_rate:
            failures.append("active_date_rate_below_minimum")
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
        if (
            not np.isfinite(mean_net_spread)
            or mean_net_spread < self.settings.min_mean_net_quintile_spread
        ):
            failures.append("mean_net_quintile_spread_below_minimum")
        if (
            not np.isfinite(net_consistency)
            or net_consistency < self.settings.min_net_spread_consistency
        ):
            failures.append("net_spread_consistency_below_minimum")
        if (
            not np.isfinite(joint_segment_frequency)
            or joint_segment_frequency
            < self.settings.min_joint_segment_frequency
        ):
            failures.append("joint_segment_frequency_below_minimum")
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
            "active_date_rate": active_date_rate,
            "ic_dates": len(ic_series),
            "mean_directed_ic": mean_ic,
            "direction_consistency": consistency,
            "newey_west_t": nw_t,
            "newey_west_p_value": nw_p,
            "segment_means": segment_means,
            "segment_selection_frequency": segment_frequency,
            "mean_gross_quintile_spread": (
                float(gross_spread.mean()) if not gross_spread.empty else np.nan
            ),
            "mean_net_quintile_spread": mean_net_spread,
            "net_spread_dates": len(net_spread),
            "net_spread_consistency": net_consistency,
            "net_segment_means": net_segment_means,
            "joint_segment_frequency": joint_segment_frequency,
            "quintile_monotonicity": monotonicity,
            "mean_score_churn": churn,
            "mean_quintile_turnover": mean_turnover,
            "ranking_score": ranking_score,
            "hard_failures": failures,
        }

    def _segment_statistics(
        self,
        ic_values: pd.Series,
        net_spreads: pd.Series,
    ) -> tuple[list[float], list[float], float, float]:
        dates = sorted(set(ic_values.index).union(net_spreads.index))
        if not dates:
            return [], [], np.nan, np.nan
        count = min(self.settings.segments, len(dates))
        ic_means: list[float] = []
        net_means: list[float] = []
        for positions in np.array_split(np.arange(len(dates)), count):
            block_dates = [dates[position] for position in positions]
            ic_means.append(float(ic_values.reindex(block_dates).mean()))
            net_means.append(float(net_spreads.reindex(block_dates).mean()))
        ic_pass = np.asarray(ic_means, dtype=float) >= (
            self.settings.min_mean_directed_ic
        )
        net_pass = np.asarray(net_means, dtype=float) >= (
            self.settings.min_mean_net_quintile_spread
        )
        segment_frequency = float(np.mean(ic_pass))
        joint_frequency = float(np.mean(ic_pass & net_pass))
        return ic_means, net_means, segment_frequency, joint_frequency

    @staticmethod
    def _quintile_monotonicity(quintile_rows: Sequence[pd.Series]) -> float:
        if not quintile_rows:
            return np.nan
        means = pd.concat(quintile_rows, axis=1).mean(axis=1).reindex(range(1, 6))
        if means.notna().sum() < 3:
            return np.nan
        return float(
            means.corr(
                pd.Series(means.index, index=means.index),
                method="spearman",
            )
        )

    def _ranking_score(
        self,
        *,
        mean_ic: float,
        mean_net_spread: float,
        newey_west_t: float,
        joint_segment_frequency: float,
        monotonicity: float,
        churn: float,
    ) -> float:
        if not np.isfinite(mean_ic):
            return -np.inf
        t_component = (
            float(np.clip(newey_west_t, -5.0, 5.0))
            if np.isfinite(newey_west_t)
            else -5.0
        )
        net_component = mean_net_spread if np.isfinite(mean_net_spread) else -0.01
        segment_component = (
            joint_segment_frequency
            if np.isfinite(joint_segment_frequency)
            else 0.0
        )
        monotonicity_component = monotonicity if np.isfinite(monotonicity) else -1.0
        churn_component = churn if np.isfinite(churn) else 1.0
        return float(
            mean_ic
            + 2.0 * net_component
            + 0.002 * t_component
            + 0.010 * segment_component
            + 0.005 * monotonicity_component
            - self.settings.churn_penalty * churn_component
        )

    @staticmethod
    def _trade_date(frame: pd.DataFrame, fallback: str) -> str:
        if "label_entry_date" not in frame.columns:
            return fallback
        dates = frame["label_entry_date"].dropna().astype(str)
        return str(dates.iloc[0]) if not dates.empty else fallback

    def _family(self, factor: str) -> str:
        return self.families.get(factor, f"unclassified:{factor}")


def _equal_weight_leg_turnover_and_cost(
    current: frozenset[str],
    previous: frozenset[str] | None,
    *,
    linear_rate: float,
    stamp_rate: float,
) -> tuple[float, float]:
    if not current:
        raise ValueError("Equal-weight leg must contain at least one instrument")
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
    turnover = max(buys, sells)
    cost = linear_rate * (buys + sells) + stamp_rate * sells
    return turnover, cost


def _two_sided_normal_p_value(t_value: float) -> float:
    if not np.isfinite(t_value):
        return np.nan
    return float(math.erfc(abs(float(t_value)) / math.sqrt(2.0)))


def _benjamini_hochberg(p_values: Mapping[str, Any]) -> dict[str, float]:
    finite = sorted(
        (
            (factor, float(value))
            for factor, value in p_values.items()
            if _is_finite_probability(value)
        ),
        key=lambda item: (item[1], item[0]),
    )
    q_values = {factor: np.nan for factor in p_values}
    if not finite:
        return q_values
    count = len(finite)
    running = 1.0
    for rank in range(count, 0, -1):
        factor, p_value = finite[rank - 1]
        running = min(running, p_value * count / rank)
        q_values[factor] = float(np.clip(running, 0.0, 1.0))
    return q_values


def _is_finite_probability(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and 0.0 <= numeric <= 1.0)


def _correlation_clusters(
    factors: Sequence[str],
    correlation: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    names = tuple(factors)
    parent = {factor: factor for factor in names}

    def find(factor: str) -> str:
        while parent[factor] != factor:
            parent[factor] = parent[parent[factor]]
            factor = parent[factor]
        return factor

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            value = correlation.loc[f"{left}__z", f"{right}__z"]
            if np.isfinite(value) and abs(float(value)) > threshold:
                union(left, right)

    groups: dict[str, list[str]] = {}
    for factor in names:
        groups.setdefault(find(factor), []).append(factor)
    ordered_groups = sorted(
        (tuple(sorted(members)) for members in groups.values()),
        key=lambda members: members,
    )
    cluster_members = {
        f"cluster_{position:03d}": members
        for position, members in enumerate(ordered_groups, start=1)
    }
    cluster_by_factor = {
        factor: cluster
        for cluster, members in cluster_members.items()
        for factor in members
    }
    return cluster_by_factor, cluster_members


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
