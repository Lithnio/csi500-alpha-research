from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from csi500_alpha.features.catalog import DIRECTIONS, FACTOR_NAMES


@dataclass(frozen=True)
class FactorDiagnostics:
    ic_by_date: pd.DataFrame
    summary: pd.DataFrame
    quintile_returns: pd.DataFrame
    correlation: pd.DataFrame


def compute_factor_diagnostics(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_quality: pd.DataFrame,
    factor_names: Sequence[str] | None = None,
    directions: Mapping[str, int] | None = None,
) -> FactorDiagnostics:
    names = tuple(FACTOR_NAMES if factor_names is None else factor_names)
    direction_map = DIRECTIONS if directions is None else directions
    missing_directions = sorted(set(names).difference(direction_map))
    if missing_directions:
        raise ValueError(f"Missing factor directions: {missing_directions}")
    panel = features.merge(
        labels[["decision_date", "instrument", "forward_active_return"]],
        on=["decision_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    ic_rows: list[dict[str, object]] = []
    quintile_rows: list[dict[str, object]] = []
    churn_by_factor = _factor_churn(panel, names)

    for factor in names:
        score_column = f"{factor}__z"
        direction = direction_map[factor]
        for decision_date, frame in panel.groupby("decision_date", sort=True):
            sample = frame[[score_column, "forward_active_return"]].dropna()
            rank_ic = np.nan
            if len(sample) >= 5 and sample[score_column].nunique() > 1:
                rank_ic = float(
                    sample[score_column].corr(
                        sample["forward_active_return"],
                        method="spearman",
                    )
                )
                percentile = sample[score_column].rank(method="first", pct=True)
                quintile = np.ceil(percentile * 5.0).clip(1, 5).astype(int)
                grouped = sample.groupby(quintile)["forward_active_return"].mean()
                for group, value in grouped.items():
                    quintile_rows.append(
                        {
                            "decision_date": str(decision_date),
                            "factor": factor,
                            "quintile": int(group),
                            "mean_active_return": float(value),
                        }
                    )
            ic_rows.append(
                {
                    "decision_date": str(decision_date),
                    "factor": factor,
                    "observations": int(len(sample)),
                    "rank_ic": rank_ic,
                    "directed_rank_ic": direction * rank_ic,
                    "score_churn": churn_by_factor.get(
                        (factor, str(decision_date)),
                        np.nan,
                    ),
                }
            )

    ic_by_date = pd.DataFrame(ic_rows)
    quintile_returns = pd.DataFrame(quintile_rows)
    summary_rows: list[dict[str, object]] = []
    quality_summary = feature_quality.groupby("factor").agg(
        mean_coverage=("coverage", "mean"),
        active_date_rate=("active", "mean"),
        mean_clipped_fraction=("clipped_fraction", "mean"),
        industry_neutralized_rate=("industry_neutralized", "mean"),
    )
    for factor, frame in ic_by_date.groupby("factor", sort=False):
        directed = frame["directed_rank_ic"].dropna()
        mean_ic = float(frame["rank_ic"].mean())
        mean_directed = float(directed.mean())
        std_directed = float(directed.std(ddof=1))
        icir = (
            mean_directed / std_directed * np.sqrt(52.0)
            if np.isfinite(std_directed) and std_directed > 0
            else np.nan
        )
        group_means = _quintile_means(quintile_returns, factor)
        monotonicity = np.nan
        if group_means.notna().sum() >= 3:
            monotonicity = float(
                spearmanr(group_means.index, group_means.to_numpy()).statistic
            )
        quality = quality_summary.loc[factor]
        summary_rows.append(
            {
                "factor": factor,
                "direction": direction_map[factor],
                "ic_dates": int(len(directed)),
                "mean_rank_ic": mean_ic,
                "mean_directed_rank_ic": mean_directed,
                "directed_ic_std": std_directed,
                "directed_icir": icir,
                "newey_west_t": newey_west_tstat(directed),
                "mean_score_churn": float(frame["score_churn"].mean()),
                "quintile_monotonicity": monotonicity,
                "mean_coverage": float(quality["mean_coverage"]),
                "active_date_rate": float(quality["active_date_rate"]),
                "mean_clipped_fraction": float(quality["mean_clipped_fraction"]),
                "industry_neutralized_rate": float(
                    quality["industry_neutralized_rate"]
                ),
                **{
                    f"q{group}_mean_active_return": float(group_means.get(group, np.nan))
                    for group in range(1, 6)
                },
            }
        )
    summary = pd.DataFrame(summary_rows)
    return FactorDiagnostics(
        ic_by_date=ic_by_date,
        summary=summary,
        quintile_returns=quintile_returns,
        correlation=_average_factor_correlation(panel, names),
    )


def _factor_churn(
    panel: pd.DataFrame,
    factor_names: Sequence[str],
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for factor in factor_names:
        column = f"{factor}__z"
        previous: pd.Series | None = None
        for decision_date, frame in panel.groupby("decision_date", sort=True):
            current = frame.set_index("instrument")[column].dropna().rank(pct=True)
            if previous is not None:
                common = previous.index.intersection(current.index)
                if len(common) >= 5:
                    result[(factor, str(decision_date))] = float(
                        (current.loc[common] - previous.loc[common]).abs().median()
                    )
            previous = current
    return result


def newey_west_tstat(values: pd.Series) -> float:
    mean, standard_error = newey_west_mean_standard_error(values)
    if not np.isfinite(mean) or not np.isfinite(standard_error):
        return np.nan
    if standard_error <= 0:
        return np.nan
    return float(mean / standard_error)


def newey_west_mean_standard_error(
    values: pd.Series,
    *,
    max_lags: int = 4,
) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) < 6:
        return (
            float(np.mean(clean)) if len(clean) else np.nan,
            np.nan,
        )
    model = sm.OLS(clean, np.ones((len(clean), 1), dtype=float)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(max_lags, len(clean) - 1)},
    )
    return float(model.params[0]), float(model.bse[0])


def _quintile_means(quintiles: pd.DataFrame, factor: str) -> pd.Series:
    if quintiles.empty:
        return pd.Series(dtype=float)
    return (
        quintiles[quintiles["factor"] == factor]
        .groupby("quintile")["mean_active_return"]
        .mean()
        .reindex(range(1, 6))
    )


def _average_factor_correlation(
    panel: pd.DataFrame,
    factor_names: Sequence[str],
) -> pd.DataFrame:
    names = tuple(factor_names)
    columns = [f"{factor}__z" for factor in names]
    matrices = [
        frame[columns].corr(method="spearman")
        for _, frame in panel.groupby("decision_date", sort=True)
    ]
    if not matrices:
        return pd.DataFrame(index=names, columns=names, dtype=float)
    stacked = np.stack([matrix.to_numpy(dtype=float) for matrix in matrices])
    average = np.nanmean(stacked, axis=0)
    return pd.DataFrame(average, index=names, columns=names)
