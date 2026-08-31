from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from csi500_alpha.config import AppConfig, ComponentSettings
from csi500_alpha.errors import ConfigurationError, DataQualityError
from csi500_alpha.features.catalog import (
    A3_ALL_DAILY_FACTOR_CATALOG,
    A3_ALL_DAILY_FAMILIES,
    FactorDefinition,
)
from csi500_alpha.features.fundamental import (
    FUNDAMENTAL_FACTOR_CATALOG,
    FUNDAMENTAL_FACTOR_NAMES,
    FUNDAMENTAL_FAMILIES,
)
from csi500_alpha.research.diagnostics import FactorDiagnostics, newey_west_tstat
from csi500_alpha.utils import canonical_json, sha256_text

FACTOR_AUDIT_CONTRACT_VERSION = "csi500-factor-audit-v1"


@dataclass(frozen=True)
class FactorAuditGates:
    min_mean_coverage: float = 0.80
    min_active_date_rate: float = 0.80
    min_ic_dates: int = 100
    min_audited_years: int = 6
    min_positive_year_fraction: float = 0.60
    min_median_yearly_directed_ic: float = 0.0
    min_median_yearly_net_q5_q1: float = 0.0

    def validate(self) -> None:
        for name in (
            "min_mean_coverage",
            "min_active_date_rate",
            "min_positive_year_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"factor_audit.gates.{name} must be in [0, 1]")
        if self.min_ic_dates < 1:
            raise ConfigurationError(
                "factor_audit.gates.min_ic_dates must be positive"
            )
        if self.min_audited_years < 1:
            raise ConfigurationError(
                "factor_audit.gates.min_audited_years must be positive"
            )


@dataclass(frozen=True)
class FactorAuditSpec:
    config_path: Path
    audit_id: str
    base_config_path: Path
    base_config_reference: str
    start_date: str
    end_date: str
    max_financial_age_days: int
    output_subdirectory: str
    feature_provider_name: str
    factor_names: tuple[str, ...]
    gates: FactorAuditGates

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> FactorAuditSpec:
        path = Path(config_path).resolve()
        if not path.is_file():
            raise ConfigurationError(
                f"Factor-audit configuration does not exist: {path}"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        root = _mapping(raw, "factor-audit config")
        _reject_unknown(root, {"factor_audit"}, "factor-audit config")
        section = _mapping(
            _required(root, "factor_audit", "factor-audit config"),
            "factor_audit",
        )
        _reject_unknown(
            section,
            {
                "id",
                "base_config",
                "start_date",
                "end_date",
                "max_financial_age_days",
                "output_subdirectory",
                "feature_provider",
                "factors",
                "gates",
            },
            "factor_audit",
        )
        audit_id = str(_required(section, "id", "factor_audit"))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", audit_id) is None:
            raise ConfigurationError(
                "factor_audit.id must use only letters, numbers, '.', '-' or '_'"
            )
        base_reference = str(_required(section, "base_config", "factor_audit"))
        base_path = (path.parent / base_reference).resolve()
        if not base_path.is_file():
            raise ConfigurationError(
                f"factor_audit.base_config does not exist: {base_path}"
            )
        start_date = _date(
            _required(section, "start_date", "factor_audit"),
            "factor_audit.start_date",
        )
        end_date = _date(
            _required(section, "end_date", "factor_audit"),
            "factor_audit.end_date",
        )
        if start_date > end_date:
            raise ConfigurationError(
                "factor_audit.start_date must not exceed factor_audit.end_date"
            )
        output_subdirectory = str(
            section.get("output_subdirectory", "factor-audits")
        )
        output_path = Path(output_subdirectory)
        if (
            output_path.is_absolute()
            or ".." in output_path.parts
            or output_subdirectory.strip() == ""
        ):
            raise ConfigurationError(
                "factor_audit.output_subdirectory must be a safe relative path"
            )
        factor_names = tuple(
            str(value) for value in _sequence(section.get("factors", ()), "factors")
        )
        if len(set(factor_names)) != len(factor_names):
            raise ConfigurationError("factor_audit.factors must be unique")
        feature_provider_name = str(
            section.get("feature_provider", "builtin_daily_fundamental")
        ).strip()
        if not feature_provider_name:
            raise ConfigurationError("factor_audit.feature_provider cannot be empty")
        raw_gates = _mapping(section.get("gates", {}), "factor_audit.gates")
        _reject_unknown(
            raw_gates,
            {
                "min_mean_coverage",
                "min_active_date_rate",
                "min_ic_dates",
                "min_audited_years",
                "min_positive_year_fraction",
                "min_median_yearly_directed_ic",
                "min_median_yearly_net_q5_q1",
            },
            "factor_audit.gates",
        )
        gates = FactorAuditGates(
            min_mean_coverage=float(raw_gates.get("min_mean_coverage", 0.80)),
            min_active_date_rate=float(
                raw_gates.get("min_active_date_rate", 0.80)
            ),
            min_ic_dates=int(raw_gates.get("min_ic_dates", 100)),
            min_audited_years=int(raw_gates.get("min_audited_years", 6)),
            min_positive_year_fraction=float(
                raw_gates.get("min_positive_year_fraction", 0.60)
            ),
            min_median_yearly_directed_ic=float(
                raw_gates.get("min_median_yearly_directed_ic", 0.0)
            ),
            min_median_yearly_net_q5_q1=float(
                raw_gates.get("min_median_yearly_net_q5_q1", 0.0)
            ),
        )
        gates.validate()
        max_age = int(section.get("max_financial_age_days", 180))
        if max_age < 1:
            raise ConfigurationError(
                "factor_audit.max_financial_age_days must be positive"
            )
        result = cls(
            config_path=path,
            audit_id=audit_id,
            base_config_path=base_path,
            base_config_reference=base_reference,
            start_date=start_date,
            end_date=end_date,
            max_financial_age_days=max_age,
            output_subdirectory=output_subdirectory,
            feature_provider_name=feature_provider_name,
            factor_names=factor_names,
            gates=gates,
        )
        result.resolved_config()
        return result

    def resolved_config(self) -> AppConfig:
        base = AppConfig.from_yaml(self.base_config_path)
        if self.start_date < base.dates.raw_start or self.end_date > base.dates.end:
            raise ConfigurationError(
                "Factor-audit dates must stay inside the base data snapshot"
            )
        provider_params = (
            {"max_financial_age_days": self.max_financial_age_days}
            if self.feature_provider_name.endswith("_fundamental")
            else {}
        )
        config = replace(
            base,
            workflow=replace(
                base.workflow,
                feature_start=self.start_date,
                portfolio_start=self.start_date,
                factor_names=self.factor_names,
                feature_provider=ComponentSettings(
                    self.feature_provider_name,
                    provider_params,
                ),
            ),
            experiment=replace(base.experiment, stage="walk_forward"),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": FACTOR_AUDIT_CONTRACT_VERSION,
            "factor_audit": {
                "id": self.audit_id,
                "base_config": self.base_config_reference,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "max_financial_age_days": self.max_financial_age_days,
                "output_subdirectory": self.output_subdirectory,
                "feature_provider": self.feature_provider_name,
                "factors": list(self.factor_names),
                "gates": asdict(self.gates),
            },
        }

    @property
    def spec_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass(frozen=True)
class FactorAuditTables:
    rebalance_spreads: pd.DataFrame
    yearly: pd.DataFrame
    summary: pd.DataFrame
    distribution: pd.DataFrame
    industry_dependence: pd.DataFrame
    industry_distribution: pd.DataFrame


def factor_catalog_frame(factor_names: Sequence[str]) -> pd.DataFrame:
    definitions = {
        definition.name: definition
        for definition in (
            *A3_ALL_DAILY_FACTOR_CATALOG,
            *FUNDAMENTAL_FACTOR_CATALOG,
        )
    }
    missing = sorted(set(factor_names).difference(definitions))
    if missing:
        raise DataQualityError(f"Factor catalog lacks definitions: {missing}")
    return pd.DataFrame(
        [
            {
                **asdict(definitions[name]),
                "input_fields": ",".join(definitions[name].input_fields),
            }
            for name in factor_names
        ]
    )


def combined_factor_families() -> dict[str, str]:
    return {**A3_ALL_DAILY_FAMILIES, **FUNDAMENTAL_FAMILIES}


def build_factor_audit_tables(
    *,
    raw_features: pd.DataFrame,
    processed_features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_quality: pd.DataFrame,
    diagnostics: FactorDiagnostics,
    market_panel: pd.DataFrame,
    open_dates: Sequence[str],
    factor_names: Sequence[str],
    directions: Mapping[str, int],
    families: Mapping[str, str],
    gates: FactorAuditGates,
    label_horizon: int,
    linear_cost_bps: float,
    stamp_duty_change_date: str,
    stamp_duty_before: float,
    stamp_duty_after: float,
    adv_window: int,
    max_adv_participation: float,
) -> FactorAuditTables:
    names = tuple(factor_names)
    _validate_audit_inputs(
        raw_features=raw_features,
        processed_features=processed_features,
        labels=labels,
        feature_quality=feature_quality,
        factor_names=names,
        directions=directions,
        families=families,
    )
    panel = processed_features.merge(
        labels[
            [
                "decision_date",
                "instrument",
                "label_entry_date",
                "label_valid",
                "forward_active_return",
            ]
        ],
        on=["decision_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        _trailing_adv(
            processed_features[["decision_date", "instrument"]],
            market_panel,
            open_dates=open_dates,
            window=adv_window,
        ),
        on=["decision_date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    rebalance_spreads = _rebalance_spreads(
        panel,
        names,
        directions,
        linear_cost_bps=linear_cost_bps,
        stamp_duty_change_date=stamp_duty_change_date,
        stamp_duty_before=stamp_duty_before,
        stamp_duty_after=stamp_duty_after,
        max_adv_participation=max_adv_participation,
    )
    distribution = _distribution_audit(raw_features, names)
    industry_dependence = _industry_dependence(raw_features, names, directions)
    industry_distribution = _industry_distribution(
        processed_features,
        names,
        directions,
    )
    yearly = _yearly_audit(
        diagnostics=diagnostics,
        rebalance_spreads=rebalance_spreads,
        feature_quality=feature_quality,
        industry_dependence=industry_dependence,
        label_horizon=label_horizon,
    )
    summary = _factor_summary(
        raw_features=raw_features,
        diagnostics=diagnostics,
        yearly=yearly,
        rebalance_spreads=rebalance_spreads,
        feature_quality=feature_quality,
        industry_dependence=industry_dependence,
        factor_names=names,
        directions=directions,
        families=families,
        gates=gates,
    )
    return FactorAuditTables(
        rebalance_spreads=rebalance_spreads,
        yearly=yearly,
        summary=summary,
        distribution=distribution,
        industry_dependence=industry_dependence,
        industry_distribution=industry_distribution,
    )


def _validate_audit_inputs(
    *,
    raw_features: pd.DataFrame,
    processed_features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_quality: pd.DataFrame,
    factor_names: tuple[str, ...],
    directions: Mapping[str, int],
    families: Mapping[str, str],
) -> None:
    keys = {"decision_date", "instrument"}
    for frame, label in (
        (raw_features, "raw features"),
        (processed_features, "processed features"),
        (labels, "labels"),
    ):
        missing = sorted(keys.difference(frame.columns))
        if missing:
            raise DataQualityError(f"Factor audit {label} lacks columns: {missing}")
        if frame.duplicated(list(keys)).any():
            raise DataQualityError(f"Factor audit {label} has duplicate keys")
    missing_raw = sorted(set(factor_names).difference(raw_features.columns))
    missing_scores = sorted(
        f"{factor}__z"
        for factor in factor_names
        if f"{factor}__z" not in processed_features
    )
    if missing_raw or missing_scores:
        raise DataQualityError(
            f"Factor audit is missing raw={missing_raw}, scores={missing_scores}"
        )
    if sorted(set(factor_names).difference(directions)):
        raise DataQualityError("Factor audit lacks one or more factor directions")
    if sorted(set(factor_names).difference(families)):
        raise DataQualityError("Factor audit lacks one or more factor families")
    quality_required = {
        "decision_date",
        "factor",
        "coverage",
        "active",
        "clipped_fraction",
    }
    if not quality_required.issubset(feature_quality.columns):
        missing = sorted(quality_required.difference(feature_quality.columns))
        raise DataQualityError(f"Factor audit feature quality lacks columns: {missing}")


def _trailing_adv(
    keys: pd.DataFrame,
    market_panel: pd.DataFrame,
    *,
    open_dates: Sequence[str],
    window: int,
) -> pd.DataFrame:
    if window < 1:
        raise DataQualityError("Factor audit ADV window must be positive")
    required = {"trade_date", "instrument", "amount_cny"}
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise DataQualityError(f"Factor audit market panel lacks columns: {missing}")
    dates = [str(value) for value in open_dates]
    amount = market_panel.pivot(
        index="trade_date",
        columns="instrument",
        values="amount_cny",
    ).reindex(dates)
    amount = amount.apply(pd.to_numeric, errors="coerce")
    adv = amount.rolling(window, min_periods=window).mean()
    rows: list[pd.DataFrame] = []
    for decision_date, frame in keys.groupby("decision_date", sort=True):
        date = str(decision_date)
        instruments = frame["instrument"].astype(str)
        values = (
            adv.loc[date].reindex(instruments).to_numpy()
            if date in adv.index
            else np.full(len(frame), np.nan)
        )
        rows.append(
            pd.DataFrame(
                {
                    "decision_date": date,
                    "instrument": instruments.to_numpy(),
                    "adv20_cny": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["decision_date", "instrument", "adv20_cny"]
    )


def _rebalance_spreads(
    panel: pd.DataFrame,
    factor_names: tuple[str, ...],
    directions: Mapping[str, int],
    *,
    linear_cost_bps: float,
    stamp_duty_change_date: str,
    stamp_duty_before: float,
    stamp_duty_after: float,
    max_adv_participation: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    linear_rate = linear_cost_bps / 10_000.0
    for factor in factor_names:
        previous: dict[int, pd.Series] = {}
        score_column = f"{factor}__z"
        for decision_date, frame in panel.groupby("decision_date", sort=True):
            date = str(decision_date)
            score = pd.to_numeric(frame[score_column], errors="coerce")
            valid_score = score.notna()
            sample = frame.loc[valid_score].copy()
            sample["_directed_score"] = (
                int(directions[factor]) * score.loc[valid_score]
            )
            if len(sample) < 5 or sample["_directed_score"].nunique() < 2:
                continue
            percentile = sample["_directed_score"].rank(method="first", pct=True)
            sample["_quintile"] = np.ceil(percentile * 5.0).clip(1, 5).astype(int)
            outcomes = pd.to_numeric(sample["forward_active_return"], errors="coerce")
            sample["_outcome"] = outcomes.where(sample["label_valid"].fillna(False))
            entry_dates = sample["label_entry_date"].dropna().astype(str)
            trade_date = entry_dates.iloc[0] if not entry_dates.empty else date
            stamp_rate = (
                stamp_duty_before
                if trade_date < stamp_duty_change_date
                else stamp_duty_after
            )
            leg: dict[int, dict[str, float]] = {}
            for quintile in (1, 5):
                group = sample.loc[sample["_quintile"].eq(quintile)].copy()
                weights = pd.Series(
                    1.0 / len(group),
                    index=group["instrument"].astype(str),
                    dtype=float,
                )
                turnover, cost = _leg_turnover_and_cost(
                    weights,
                    previous.get(quintile),
                    linear_rate=linear_rate,
                    stamp_rate=stamp_rate,
                )
                previous[quintile] = weights
                adv = pd.to_numeric(group["adv20_cny"], errors="coerce")
                capacity_values = (
                    max_adv_participation * adv / (1.0 / len(group))
                ).where(adv > 0)
                leg[quintile] = {
                    "return": float(group["_outcome"].mean()),
                    "observations": float(group["_outcome"].notna().sum()),
                    "members": float(len(group)),
                    "turnover": turnover,
                    "cost": cost,
                    "median_adv_cny": float(adv.median()),
                    "entry_capacity_cny": float(capacity_values.min()),
                }
            q1 = leg[1]
            q5 = leg[5]
            gross_spread = q5["return"] - q1["return"]
            rows.append(
                {
                    "decision_date": date,
                    "year": date[:4],
                    "factor": factor,
                    "score_observations": int(len(sample)),
                    "q1_members": int(q1["members"]),
                    "q5_members": int(q5["members"]),
                    "q1_return_observations": int(q1["observations"]),
                    "q5_return_observations": int(q5["observations"]),
                    "q1_mean_active_return": q1["return"],
                    "q5_mean_active_return": q5["return"],
                    "q5_minus_q1_gross": gross_spread,
                    "q1_turnover": q1["turnover"],
                    "q5_turnover": q5["turnover"],
                    "q1_estimated_cost": q1["cost"],
                    "q5_estimated_cost": q5["cost"],
                    "q5_net_active_return": q5["return"] - q5["cost"],
                    "q5_minus_q1_net": gross_spread - q1["cost"] - q5["cost"],
                    "q5_median_adv20_cny": q5["median_adv_cny"],
                    "q5_entry_capacity_cny": q5["entry_capacity_cny"],
                }
            )
    return pd.DataFrame(rows)


def _leg_turnover_and_cost(
    current: pd.Series,
    previous: pd.Series | None,
    *,
    linear_rate: float,
    stamp_rate: float,
) -> tuple[float, float]:
    prior = previous if previous is not None else pd.Series(dtype=float)
    instruments = current.index.union(prior.index)
    delta = current.reindex(instruments, fill_value=0.0) - prior.reindex(
        instruments,
        fill_value=0.0,
    )
    buys = float(delta.clip(lower=0.0).sum())
    sells = float((-delta.clip(upper=0.0)).sum())
    turnover = max(buys, sells)
    cost = linear_rate * (buys + sells) + stamp_rate * sells
    return turnover, cost


def _distribution_audit(
    raw_features: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fundamental = set(FUNDAMENTAL_FACTOR_NAMES)
    for factor in factor_names:
        for decision_date, frame in raw_features.groupby("decision_date", sort=True):
            values = pd.to_numeric(frame[factor], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            clean = values.dropna()
            quantiles = clean.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
            median = float(quantiles.get(0.50, np.nan))
            mad = float((clean - median).abs().median()) if not clean.empty else np.nan
            lookahead = 0
            stale_values = 0
            future_periods = 0
            if factor in fundamental:
                available = frame.get(
                    "financial_available_date",
                    pd.Series(pd.NA, index=frame.index),
                )
                stale = frame.get(
                    "financial_stale",
                    pd.Series(False, index=frame.index),
                ).fillna(True)
                periods = frame.get(
                    "financial_report_period",
                    pd.Series(pd.NA, index=frame.index),
                )
                lookahead = int(
                    (
                        values.notna()
                        & available.notna()
                        & available.astype(str).gt(str(decision_date))
                    ).sum()
                )
                stale_values = int((values.notna() & stale.astype(bool)).sum())
                future_periods = int(
                    (
                        values.notna()
                        & periods.notna()
                        & periods.astype(str).gt(str(decision_date))
                    ).sum()
                )
            rows.append(
                {
                    "decision_date": str(decision_date),
                    "year": str(decision_date)[:4],
                    "factor": factor,
                    "observations": int(clean.size),
                    "coverage": float(clean.size / len(frame)) if len(frame) else 0.0,
                    "minimum": float(clean.min()) if not clean.empty else np.nan,
                    "p01": float(quantiles.get(0.01, np.nan)),
                    "p05": float(quantiles.get(0.05, np.nan)),
                    "median": median,
                    "p95": float(quantiles.get(0.95, np.nan)),
                    "p99": float(quantiles.get(0.99, np.nan)),
                    "maximum": float(clean.max()) if not clean.empty else np.nan,
                    "mad": mad,
                    "zero_fraction": float(clean.eq(0.0).mean()) if not clean.empty else np.nan,
                    "lookahead_violations": lookahead,
                    "stale_value_violations": stale_values,
                    "future_report_period_violations": future_periods,
                }
            )
    return pd.DataFrame(rows)


def _industry_dependence(
    raw_features: pd.DataFrame,
    factor_names: tuple[str, ...],
    directions: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in factor_names:
        for decision_date, frame in raw_features.groupby("decision_date", sort=True):
            values = pd.to_numeric(frame[factor], errors="coerce")
            valid = values.notna() & frame["industry_code"].notna()
            sample = pd.DataFrame(
                {
                    "value": values.loc[valid],
                    "industry": frame.loc[valid, "industry_code"].astype(str),
                }
            )
            industry_r2 = np.nan
            rank_mean_range = np.nan
            if len(sample) >= 5 and sample["value"].nunique() > 1:
                overall = float(sample["value"].mean())
                total_ss = float(np.square(sample["value"] - overall).sum())
                grouped = sample.groupby("industry")["value"]
                counts = grouped.size()
                means = grouped.mean()
                between_ss = float((counts * np.square(means - overall)).sum())
                industry_r2 = between_ss / total_ss if total_ss > 0 else np.nan
                ranks = int(directions[factor]) * sample["value"].rank(pct=True)
                rank_means = ranks.groupby(sample["industry"]).mean()
                rank_mean_range = float(rank_means.max() - rank_means.min())
            rows.append(
                {
                    "decision_date": str(decision_date),
                    "year": str(decision_date)[:4],
                    "factor": factor,
                    "observations": int(len(sample)),
                    "industry_groups": int(sample["industry"].nunique()),
                    "industry_r2": industry_r2,
                    "industry_rank_mean_range": rank_mean_range,
                }
            )
    return pd.DataFrame(rows)


def _industry_distribution(
    processed_features: pd.DataFrame,
    factor_names: tuple[str, ...],
    directions: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = processed_features.copy()
    base["year"] = base["decision_date"].astype(str).str[:4]
    base["industry_code"] = base["industry_code"].astype("string").fillna(
        "__MISSING__"
    )
    for factor in factor_names:
        for (year, industry), frame in base.groupby(
            ["year", "industry_code"],
            sort=True,
        ):
            raw = pd.to_numeric(frame[factor], errors="coerce")
            score = int(directions[factor]) * pd.to_numeric(
                frame[f"{factor}__z"],
                errors="coerce",
            )
            rows.append(
                {
                    "year": str(year),
                    "factor": factor,
                    "industry_code": str(industry),
                    "rows": int(len(frame)),
                    "raw_observations": int(raw.notna().sum()),
                    "raw_coverage": float(raw.notna().mean()),
                    "raw_median": float(raw.median()),
                    "mean_directed_score": float(score.mean()),
                }
            )
    return pd.DataFrame(rows)


def _yearly_audit(
    *,
    diagnostics: FactorDiagnostics,
    rebalance_spreads: pd.DataFrame,
    feature_quality: pd.DataFrame,
    industry_dependence: pd.DataFrame,
    label_horizon: int,
) -> pd.DataFrame:
    ic = diagnostics.ic_by_date.copy()
    ic["year"] = ic["decision_date"].astype(str).str[:4]
    quality = feature_quality.copy()
    quality["year"] = quality["decision_date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    factors = sorted(ic["factor"].astype(str).unique())
    years = sorted(ic["year"].astype(str).unique())
    annualization = 252.0 / label_horizon
    for factor in factors:
        for year in years:
            ic_sample = ic.loc[
                ic["factor"].eq(factor) & ic["year"].eq(year)
            ]
            if ic_sample.empty:
                continue
            directed = pd.to_numeric(
                ic_sample["directed_rank_ic"], errors="coerce"
            ).dropna()
            spread = rebalance_spreads.loc[
                rebalance_spreads["factor"].eq(factor)
                & rebalance_spreads["year"].eq(year)
            ]
            q = quality.loc[
                quality["factor"].eq(factor) & quality["year"].eq(year)
            ]
            industry = industry_dependence.loc[
                industry_dependence["factor"].eq(factor)
                & industry_dependence["year"].eq(year)
            ]
            gross = pd.to_numeric(spread["q5_minus_q1_gross"], errors="coerce")
            net = pd.to_numeric(spread["q5_minus_q1_net"], errors="coerce")
            mean_directed = float(directed.mean())
            mean_net = float(net.mean())
            rows.append(
                {
                    "year": year,
                    "factor": factor,
                    "ic_dates": int(len(directed)),
                    "mean_directed_rank_ic": mean_directed,
                    "median_directed_rank_ic": float(directed.median()),
                    "positive_ic_date_fraction": float(directed.gt(0).mean()),
                    "newey_west_t": newey_west_tstat(directed),
                    "spread_dates": int(net.notna().sum()),
                    "mean_q5_active_return": float(
                        pd.to_numeric(
                            spread["q5_mean_active_return"], errors="coerce"
                        ).mean()
                    ),
                    "mean_q5_minus_q1_gross": float(gross.mean()),
                    "mean_q5_minus_q1_net": mean_net,
                    "annualized_q5_minus_q1_gross": float(gross.mean() * annualization),
                    "annualized_q5_minus_q1_net": float(net.mean() * annualization),
                    "positive_net_spread_date_fraction": float(net.gt(0).mean()),
                    "mean_q5_turnover": float(
                        pd.to_numeric(spread["q5_turnover"], errors="coerce").mean()
                    ),
                    "p10_q5_entry_capacity_cny": float(
                        pd.to_numeric(
                            spread["q5_entry_capacity_cny"], errors="coerce"
                        ).quantile(0.10)
                    ),
                    "mean_coverage": float(
                        pd.to_numeric(q["coverage"], errors="coerce").mean()
                    ),
                    "active_date_rate": float(q["active"].astype(float).mean()),
                    "mean_industry_r2": float(
                        pd.to_numeric(
                            industry["industry_r2"], errors="coerce"
                        ).mean()
                    ),
                    "positive_joint_block": bool(
                        np.isfinite(mean_directed)
                        and mean_directed > 0
                        and np.isfinite(mean_net)
                        and mean_net > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _factor_summary(
    *,
    raw_features: pd.DataFrame,
    diagnostics: FactorDiagnostics,
    yearly: pd.DataFrame,
    rebalance_spreads: pd.DataFrame,
    feature_quality: pd.DataFrame,
    industry_dependence: pd.DataFrame,
    factor_names: tuple[str, ...],
    directions: Mapping[str, int],
    families: Mapping[str, str],
    gates: FactorAuditGates,
) -> pd.DataFrame:
    diagnostic = diagnostics.summary.set_index("factor")
    rows: list[dict[str, Any]] = []
    fundamental = set(FUNDAMENTAL_FACTOR_NAMES)
    for factor in factor_names:
        years = yearly.loc[yearly["factor"].eq(factor)].copy()
        spreads = rebalance_spreads.loc[rebalance_spreads["factor"].eq(factor)]
        quality = feature_quality.loc[feature_quality["factor"].eq(factor)]
        industry = industry_dependence.loc[
            industry_dependence["factor"].eq(factor)
        ]
        base = diagnostic.loc[factor].to_dict()
        raw_values = pd.to_numeric(raw_features[factor], errors="coerce")
        lookahead = 0
        stale_values = 0
        future_periods = 0
        if factor in fundamental:
            available = raw_features["financial_available_date"]
            stale = raw_features["financial_stale"].fillna(True).astype(bool)
            periods = raw_features["financial_report_period"]
            decisions = raw_features["decision_date"].astype(str)
            lookahead = int(
                (
                    raw_values.notna()
                    & available.notna()
                    & available.astype(str).gt(decisions)
                ).sum()
            )
            stale_values = int((raw_values.notna() & stale).sum())
            future_periods = int(
                (
                    raw_values.notna()
                    & periods.notna()
                    & periods.astype(str).gt(decisions)
                ).sum()
            )
        audited_years = int(years["ic_dates"].gt(0).sum())
        positive_year_fraction = float(
            years.loc[years["ic_dates"].gt(0), "positive_joint_block"].mean()
        )
        mean_coverage = float(
            pd.to_numeric(quality["coverage"], errors="coerce").mean()
        )
        active_rate = float(quality["active"].astype(float).mean())
        median_yearly_ic = float(years["mean_directed_rank_ic"].median())
        median_yearly_net = float(years["mean_q5_minus_q1_net"].median())
        reasons: list[str] = []
        if not np.isfinite(mean_coverage) or mean_coverage < gates.min_mean_coverage:
            reasons.append("mean_coverage_below_minimum")
        if not np.isfinite(active_rate) or active_rate < gates.min_active_date_rate:
            reasons.append("active_date_rate_below_minimum")
        if int(base["ic_dates"]) < gates.min_ic_dates:
            reasons.append("insufficient_ic_dates")
        if audited_years < gates.min_audited_years:
            reasons.append("insufficient_audited_years")
        if (
            not np.isfinite(positive_year_fraction)
            or positive_year_fraction < gates.min_positive_year_fraction
        ):
            reasons.append("positive_year_fraction_below_minimum")
        if (
            not np.isfinite(median_yearly_ic)
            or median_yearly_ic < gates.min_median_yearly_directed_ic
        ):
            reasons.append("median_yearly_directed_ic_below_minimum")
        if (
            not np.isfinite(median_yearly_net)
            or median_yearly_net < gates.min_median_yearly_net_q5_q1
        ):
            reasons.append("median_yearly_net_q5_q1_below_minimum")
        if lookahead or stale_values or future_periods:
            reasons.append("point_in_time_violation")
        t_value = float(base["newey_west_t"])
        p_value = (
            float(2.0 * norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
        )
        rows.append(
            {
                "factor": factor,
                "family": families[factor],
                "direction": int(directions[factor]),
                **base,
                "audited_years": audited_years,
                "positive_joint_year_fraction": positive_year_fraction,
                "median_yearly_directed_rank_ic": median_yearly_ic,
                "worst_year_directed_rank_ic": float(
                    years["mean_directed_rank_ic"].min()
                ),
                "median_yearly_q5_minus_q1_gross": float(
                    years["mean_q5_minus_q1_gross"].median()
                ),
                "median_yearly_q5_minus_q1_net": median_yearly_net,
                "worst_year_q5_minus_q1_net": float(
                    years["mean_q5_minus_q1_net"].min()
                ),
                "mean_q5_turnover": float(
                    pd.to_numeric(spreads["q5_turnover"], errors="coerce").mean()
                ),
                "p10_q5_entry_capacity_cny": float(
                    pd.to_numeric(
                        spreads["q5_entry_capacity_cny"], errors="coerce"
                    ).quantile(0.10)
                ),
                "minimum_date_coverage": float(
                    pd.to_numeric(quality["coverage"], errors="coerce").min()
                ),
                "mean_industry_r2": float(
                    pd.to_numeric(industry["industry_r2"], errors="coerce").mean()
                ),
                "p95_industry_r2": float(
                    pd.to_numeric(industry["industry_r2"], errors="coerce").quantile(
                        0.95
                    )
                ),
                "industry_dependency_warning": bool(
                    pd.to_numeric(industry["industry_r2"], errors="coerce").mean()
                    > 0.50
                ),
                "lookahead_violations": lookahead,
                "stale_value_violations": stale_values,
                "future_report_period_violations": future_periods,
                "newey_west_p_value": p_value,
                "gate_reasons": ";".join(reasons),
                "eligible": not reasons,
            }
        )
    result = pd.DataFrame(rows)
    result["benjamini_hochberg_q_value"] = _benjamini_hochberg(
        pd.to_numeric(result["newey_west_p_value"], errors="coerce")
    )
    return result.sort_values(
        ["eligible", "positive_joint_year_fraction", "median_yearly_q5_minus_q1_net"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result.loc[valid.index] = adjusted
    return result


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing configuration key: {section}.{key}")
    return mapping[key]


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{section} must be a mapping")
    return value


def _sequence(value: Any, key: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"factor_audit.{key} must be a sequence")
    return tuple(value)


def _reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        names = ", ".join(f"{section}.{name}" for name in unknown)
        raise ConfigurationError(f"Unknown configuration keys: {names}")


def _date(value: Any, key: str) -> str:
    text = str(value)
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ConfigurationError(f"{key} must use YYYYMMDD: {text}") from exc
    return text


def finite_or_none(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def factor_definition_map() -> dict[str, FactorDefinition]:
    return {
        definition.name: definition
        for definition in (
            *A3_ALL_DAILY_FACTOR_CATALOG,
            *FUNDAMENTAL_FACTOR_CATALOG,
        )
    }
