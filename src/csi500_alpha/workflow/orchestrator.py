from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.normalize import build_market_panel
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import BacktestResult, SmokeEventBacktester
from csi500_alpha.features.builder import ProcessedFactors, process_factor_panel
from csi500_alpha.features.labels import build_forward_labels
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.diagnostics import FactorDiagnostics, compute_factor_diagnostics
from csi500_alpha.risk.model import LedoitWolfRiskModel
from csi500_alpha.workflow.calibration import WalkForwardReturnCalibrationEngine
from csi500_alpha.workflow.components import (
    ResearchComponentRegistry,
    default_component_registry,
)
from csi500_alpha.workflow.contracts import FeatureBuildContext
from csi500_alpha.workflow.samples import ResearchSamplePolicy
from csi500_alpha.workflow.signals import WalkForwardSignalEngine


@dataclass(frozen=True)
class WorkflowResult:
    factor_names: tuple[str, ...]
    raw_features: pd.DataFrame
    processed: ProcessedFactors
    labels: pd.DataFrame
    research_panel: pd.DataFrame
    diagnostics: FactorDiagnostics
    signals: pd.DataFrame
    evaluation_signals: pd.DataFrame
    model_fits: pd.DataFrame
    calibration_fits: pd.DataFrame
    sample_policy: dict[str, object]
    backtest: BacktestResult


class ResearchWorkflow:
    def __init__(
        self,
        config: AppConfig,
        *,
        registry: ResearchComponentRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or default_component_registry()

    def run(self, tables: Mapping[str, pd.DataFrame]) -> WorkflowResult:
        required = {
            "calendar",
            "benchmark_weights",
            "index_bars",
            "stock_bars",
            "adjustments",
            "price_limits",
            "daily_characteristics",
        }
        missing = sorted(required.difference(tables))
        if missing:
            raise ConfigurationError(f"Workflow is missing silver tables: {missing}")
        market_panel = build_market_panel(
            tables["stock_bars"],
            tables["adjustments"],
            tables["price_limits"],
        )
        open_dates = tables["calendar"].loc[
            tables["calendar"]["is_open"] == 1,
            "trade_date",
        ].astype(str).tolist()
        sample_policy = ResearchSamplePolicy(
            self.config.experiment,
            tuple(open_dates),
        )
        research_end = (
            sample_policy.signal_end
            if sample_policy.enabled
            else self.config.dates.end
        )
        provider_spec = self.config.workflow.feature_provider
        provider = self.registry.create_feature_provider(
            provider_spec.name,
            provider_spec.params,
        )
        context = FeatureBuildContext(
            market_panel=market_panel,
            index_bars=tables["index_bars"],
            daily_characteristics=tables["daily_characteristics"],
            benchmark_weights=tables["benchmark_weights"],
            open_dates=open_dates,
            start_date=self.config.workflow.feature_start,
            end_date=research_end,
            rebalance_every=self.config.research.rebalance_every,
            industry_membership=tables.get("industry_membership", pd.DataFrame()),
            industry_transition_date=self.config.features.industry_transition_date,
        )
        raw_features = provider.build_raw(context)
        factor_names = self._factor_names(provider.factor_names)
        directions = {factor: int(provider.directions[factor]) for factor in factor_names}
        processed = process_factor_panel(
            raw_features,
            self.config.features,
            factor_names=factor_names,
        )
        labels = build_forward_labels(
            features=processed.features,
            market_panel=market_panel,
            index_bars=tables["index_bars"],
            open_dates=[date for date in open_dates if date <= research_end],
            horizon=self.config.features.label_horizon,
            suspensions=tables.get("suspensions"),
        )
        research_panel = processed.features.merge(
            labels,
            on=["decision_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        diagnostics = compute_factor_diagnostics(
            features=processed.features,
            labels=labels,
            feature_quality=processed.quality,
            factor_names=factor_names,
            directions=directions,
        )
        selector_spec = self.config.workflow.selector
        selector = self.registry.create_selector(
            selector_spec.name,
            selector_spec.params,
            directions,
        )
        model_spec = self.config.workflow.model
        engine = WalkForwardSignalEngine(
            selector=selector,
            model_factory=lambda: self.registry.create_model(
                model_spec.name,
                model_spec.params,
                directions,
            ),
            refit_every=self.config.workflow.refit_every,
            sample_policy=sample_policy,
        )
        signal_start = (
            sample_policy.signal_start
            if sample_policy.enabled
            else self.config.workflow.portfolio_start
        )
        signal_end = (
            sample_policy.signal_end if sample_policy.enabled else self.config.dates.end
        )
        signal_result = engine.run(
            research_panel,
            factor_names,
            prediction_start=signal_start,
            prediction_end=signal_end,
        )
        calibrator_spec = self.config.workflow.calibrator
        calibration_result = WalkForwardReturnCalibrationEngine(
            calibrator_factory=lambda: self.registry.create_calibrator(
                calibrator_spec.name,
                calibrator_spec.params,
            ),
            refit_every=self.config.workflow.refit_every,
            sample_policy=sample_policy,
        ).run(
            signal_result.signals,
            labels,
            prediction_start=signal_start,
            prediction_end=signal_end,
        )
        risk_model = None
        optimizer = None
        if self.config.optimizer.enabled:
            risk_model = LedoitWolfRiskModel(
                self.config.risk,
                market_panel,
                open_dates,
            )
            optimizer = ActivePortfolioOptimizer(
                self.config.optimizer,
                self.config.research,
                self.config.risk,
            )
        evaluation_start = (
            sample_policy.evaluation_start
            if sample_policy.enabled
            else self.config.workflow.portfolio_start
        )
        evaluation_end = (
            sample_policy.evaluation_end
            if sample_policy.enabled
            else self.config.dates.end
        )
        evaluation_mask = calibration_result.signals["decision_date"].astype(str).between(
            evaluation_start,
            evaluation_end,
        )
        evaluation_signals = calibration_result.signals.loc[evaluation_mask].copy()
        portfolio_signals = evaluation_signals.rename(
            columns={"decision_date": "trade_date"}
        )[["trade_date", "instrument", "score", "expected_return"]]
        signal_dates = sorted(
            evaluation_signals["decision_date"].astype(str).unique()
        )
        if not signal_dates:
            raise ConfigurationError("Workflow produced no portfolio signal dates")
        portfolio_exposures = _industry_exposures(processed.features)
        portfolio_restrictions = _name_history_restrictions(
            processed.features,
            tables.get("name_history"),
        )
        backtest = SmokeEventBacktester(
            self.config.research,
            risk_model=risk_model,
            optimizer=optimizer,
        ).run(
            calendar=tables["calendar"],
            benchmark_weights=tables["benchmark_weights"],
            index_bars=tables["index_bars"],
            market_panel=market_panel,
            signals=portfolio_signals,
            portfolio_exposures=portfolio_exposures,
            portfolio_restrictions=portfolio_restrictions,
            suspensions=tables.get("suspensions"),
            start_date=signal_dates[0],
            end_date=evaluation_end,
            rebalance_dates=signal_dates,
        )
        return WorkflowResult(
            factor_names=factor_names,
            raw_features=raw_features,
            processed=processed,
            labels=labels,
            research_panel=research_panel,
            diagnostics=diagnostics,
            signals=calibration_result.signals,
            evaluation_signals=evaluation_signals,
            model_fits=signal_result.model_fits,
            calibration_fits=calibration_result.calibration_fits,
            sample_policy=sample_policy.manifest(),
            backtest=backtest,
        )

    def _factor_names(self, available: tuple[str, ...]) -> tuple[str, ...]:
        requested = self.config.workflow.factor_names
        names = requested or available
        missing = sorted(set(names).difference(available))
        if missing:
            raise ConfigurationError(
                f"Feature provider lacks requested factors: {missing}"
            )
        if not names:
            raise ConfigurationError("Workflow requires at least one factor")
        return tuple(names)


def _industry_exposures(features: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_date", "instrument", "industry_code"}
    if features.empty or not required.issubset(features.columns):
        return pd.DataFrame()
    base = features[["decision_date", "instrument", "industry_code"]].copy()
    codes = base["industry_code"].astype("string").fillna("__MISSING__")
    codes = codes.mask(codes.eq(""), "__MISSING__")
    dummies = pd.get_dummies(codes, prefix="industry", dtype=float)
    result = pd.concat(
        [
            base[["decision_date", "instrument"]].rename(
                columns={"decision_date": "trade_date"}
            ),
            dummies,
        ],
        axis=1,
    )
    return result


def _name_history_restrictions(
    features: pd.DataFrame,
    name_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build point-in-time ST buy restrictions from effective name intervals."""

    columns = (
        "trade_date",
        "instrument",
        "cannot_buy",
        "cannot_sell",
        "restriction_reason",
    )
    if name_history is None or name_history.empty or features.empty:
        return pd.DataFrame(columns=columns)
    required = {"instrument", "start_date", "end_date"}
    if not required.issubset(name_history.columns):
        return pd.DataFrame(columns=columns)
    history = name_history.copy()
    if "is_st" not in history:
        if "name" not in history:
            return pd.DataFrame(columns=columns)
        history["is_st"] = (
            history["name"].fillna("").astype(str).str.upper().str.contains("ST", regex=False)
        )
    history = history[history["is_st"].fillna(False).astype(bool)].copy()
    if history.empty:
        return pd.DataFrame(columns=columns)
    history["start_date"] = history["start_date"].fillna("").astype(str)
    history["end_date"] = history["end_date"].fillna("").astype(str)
    if "announcement_date" in history:
        history["announcement_date"] = history["announcement_date"].fillna("").astype(str)
    feature_instruments = {
        str(date): set(frame["instrument"].astype(str))
        for date, frame in features.groupby("decision_date", sort=True)
    }
    rows: list[dict[str, object]] = []
    for decision_date, instruments in feature_instruments.items():
        active = history[
            history["start_date"].ne("")
            & history["start_date"].le(decision_date)
            & (history["end_date"].eq("") | history["end_date"].ge(decision_date))
        ]
        if "announcement_date" in active:
            active = active[
                active["announcement_date"].eq("")
                | active["announcement_date"].le(decision_date)
            ]
        active_names = sorted(instruments.intersection(active["instrument"].astype(str)))
        rows.extend(
            {
                "trade_date": decision_date,
                "instrument": instrument,
                "cannot_buy": True,
                "cannot_sell": False,
                "restriction_reason": "historical_st_name",
            }
            for instrument in active_names
        )
    return pd.DataFrame(rows, columns=columns)
