from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from csi500_alpha.config import AppConfig
from csi500_alpha.data.normalize import build_market_panel
from csi500_alpha.errors import ConfigurationError
from csi500_alpha.execution.backtest import BacktestResult, SmokeEventBacktester
from csi500_alpha.features.builder import ProcessedFactors, process_factor_panel
from csi500_alpha.features.labels import build_forward_labels
from csi500_alpha.logging_utils import ProgressCallback, emit_progress
from csi500_alpha.portfolio.optimizer import ActivePortfolioOptimizer
from csi500_alpha.research.diagnostics import FactorDiagnostics, compute_factor_diagnostics
from csi500_alpha.risk.model import build_risk_model
from csi500_alpha.workflow.calibration import WalkForwardReturnCalibrationEngine
from csi500_alpha.workflow.components import (
    ResearchComponentRegistry,
    default_component_registry,
)
from csi500_alpha.workflow.contracts import FeatureBuildContext
from csi500_alpha.workflow.samples import ResearchSamplePolicy
from csi500_alpha.workflow.signals import WalkForwardSignalEngine

LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class PreparedResearchData:
    factor_names: tuple[str, ...]
    directions: dict[str, int]
    market_panel: pd.DataFrame
    open_dates: list[str]
    research_end: str
    sample_policy: ResearchSamplePolicy
    raw_features: pd.DataFrame
    processed: ProcessedFactors
    labels: pd.DataFrame
    research_panel: pd.DataFrame
    diagnostics: FactorDiagnostics


class ResearchWorkflow:
    def __init__(
        self,
        config: AppConfig,
        *,
        registry: ResearchComponentRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or default_component_registry()

    def prepare(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        research_end_override: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PreparedResearchData:
        """Build the point-in-time factor and label layer without fitting a model."""

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
        self._report_progress(progress_callback, "market_panel", "running")
        market_panel = build_market_panel(
            tables["stock_bars"],
            tables["adjustments"],
            tables["price_limits"],
        )
        self._report_progress(
            progress_callback,
            "market_panel",
            "completed",
            rows=len(market_panel),
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
        if research_end_override is not None:
            if not (
                self.config.workflow.feature_start
                <= research_end_override
                <= research_end
            ):
                raise ConfigurationError(
                    "Research end override must stay inside the configured feature window"
                )
            research_end = research_end_override
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
            benchmark_membership_intervals=tables.get(
                "benchmark_membership_intervals"
            ),
            financial_tables={
                name.removeprefix("financial_"): table
                for name, table in tables.items()
                if name.startswith("financial_")
            },
            progress_callback=progress_callback,
        )
        self._report_progress(progress_callback, "raw_features", "running")
        raw_features = provider.build_raw(context)
        self._report_progress(
            progress_callback,
            "raw_features",
            "completed",
            rows=len(raw_features),
        )
        factor_names = self._factor_names(provider.factor_names)
        directions = {factor: int(provider.directions[factor]) for factor in factor_names}
        self._report_progress(progress_callback, "factor_preprocessing", "running")
        processed = process_factor_panel(
            raw_features,
            self.config.features,
            factor_names=factor_names,
            progress_callback=progress_callback,
        )
        self._report_progress(
            progress_callback,
            "factor_preprocessing",
            "completed",
            rows=len(processed.features),
        )
        self._report_progress(progress_callback, "forward_labels", "running")
        labels = build_forward_labels(
            features=processed.features,
            market_panel=market_panel,
            index_bars=tables["index_bars"],
            open_dates=[date for date in open_dates if date <= research_end],
            horizon=self.config.features.label_horizon,
            suspensions=tables.get("suspensions"),
        )
        self._report_progress(
            progress_callback,
            "forward_labels",
            "completed",
            rows=len(labels),
        )
        research_panel = processed.features.merge(
            labels,
            on=["decision_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        self._report_progress(progress_callback, "factor_diagnostics", "running")
        diagnostics = compute_factor_diagnostics(
            features=processed.features,
            labels=labels,
            feature_quality=processed.quality,
            factor_names=factor_names,
            directions=directions,
        )
        self._report_progress(progress_callback, "factor_diagnostics", "completed")
        return PreparedResearchData(
            factor_names=factor_names,
            directions=directions,
            market_panel=market_panel,
            open_dates=open_dates,
            research_end=research_end,
            sample_policy=sample_policy,
            raw_features=raw_features,
            processed=processed,
            labels=labels,
            research_panel=research_panel,
            diagnostics=diagnostics,
        )

    def run(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> WorkflowResult:
        prepared = self.prepare(tables, progress_callback=progress_callback)
        return self.run_prepared(
            tables,
            prepared,
            progress_callback=progress_callback,
        )

    def fold_view(
        self,
        tables: Mapping[str, pd.DataFrame],
        prepared: PreparedResearchData,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> PreparedResearchData:
        """Create one future-safe fold view from a shared prepared feature layer."""

        self._validate_prepared_contract(prepared)
        sample_policy = ResearchSamplePolicy(
            self.config.experiment,
            tuple(prepared.open_dates),
        )
        research_end = (
            sample_policy.signal_end
            if sample_policy.enabled
            else self.config.dates.end
        )
        if research_end > prepared.research_end:
            raise ConfigurationError(
                "Shared prepared data do not cover the requested fold end: "
                f"prepared={prepared.research_end}, requested={research_end}"
            )

        def through_end(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or "decision_date" not in frame:
                return frame.copy()
            return frame.loc[
                frame["decision_date"].astype(str).le(research_end)
            ].copy()

        raw_features = through_end(prepared.raw_features)
        features = through_end(prepared.processed.features)
        feature_quality = through_end(prepared.processed.quality)
        open_dates = [
            date for date in prepared.open_dates if date <= research_end
        ]
        self._report_progress(progress_callback, "fold_labels", "running")
        labels = build_forward_labels(
            features=features,
            market_panel=prepared.market_panel,
            index_bars=tables["index_bars"],
            open_dates=open_dates,
            horizon=self.config.features.label_horizon,
            suspensions=tables.get("suspensions"),
        )
        self._report_progress(
            progress_callback,
            "fold_labels",
            "completed",
            rows=len(labels),
        )
        research_panel = features.merge(
            labels,
            on=["decision_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        self._report_progress(progress_callback, "fold_diagnostics", "running")
        diagnostics = compute_factor_diagnostics(
            features=features,
            labels=labels,
            feature_quality=feature_quality,
            factor_names=prepared.factor_names,
            directions=prepared.directions,
        )
        self._report_progress(progress_callback, "fold_diagnostics", "completed")
        return PreparedResearchData(
            factor_names=prepared.factor_names,
            directions=dict(prepared.directions),
            market_panel=prepared.market_panel,
            open_dates=list(prepared.open_dates),
            research_end=research_end,
            sample_policy=sample_policy,
            raw_features=raw_features,
            processed=ProcessedFactors(
                features=features,
                quality=feature_quality,
            ),
            labels=labels,
            research_panel=research_panel,
            diagnostics=diagnostics,
        )

    def run_prepared(
        self,
        tables: Mapping[str, pd.DataFrame],
        prepared: PreparedResearchData,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> WorkflowResult:
        """Fit and backtest against an immutable, already prepared fold layer."""

        self._validate_prepared_contract(prepared)
        expected_end = (
            self.config.experiment.validation_end
            if self.config.experiment.stage == "validation"
            else self.config.experiment.test_end
            if self.config.experiment.stage == "frozen_test"
            else self.config.dates.end
        )
        if prepared.research_end != expected_end:
            raise ConfigurationError(
                "Prepared research end does not match the workflow stage: "
                f"prepared={prepared.research_end}, expected={expected_end}"
            )
        factor_names = prepared.factor_names
        directions = prepared.directions
        market_panel = prepared.market_panel
        open_dates = prepared.open_dates
        sample_policy = ResearchSamplePolicy(
            self.config.experiment,
            tuple(prepared.open_dates),
        )
        raw_features = prepared.raw_features
        processed = prepared.processed
        labels = prepared.labels
        research_panel = prepared.research_panel
        diagnostics = prepared.diagnostics
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
        self._report_progress(progress_callback, "walk_forward_signals", "running")
        signal_result = engine.run(
            research_panel,
            factor_names,
            prediction_start=signal_start,
            prediction_end=signal_end,
            progress_callback=progress_callback,
        )
        self._report_progress(
            progress_callback,
            "walk_forward_signals",
            "completed",
            model_fits=len(signal_result.model_fits),
            rows=len(signal_result.signals),
        )
        calibrator_spec = self.config.workflow.calibrator
        self._report_progress(progress_callback, "return_calibration", "running")
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
            progress_callback=progress_callback,
        )
        self._report_progress(
            progress_callback,
            "return_calibration",
            "completed",
            fits=len(calibration_result.calibration_fits),
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
        portfolio_styles = _style_exposures(processed.features)
        portfolio_restrictions = _name_history_restrictions(
            processed.features,
            tables.get("name_history"),
        )
        risk_model = None
        optimizer = None
        if self.config.optimizer.enabled:
            risk_model = build_risk_model(
                self.config.risk,
                market_panel,
                open_dates,
                index_bars=tables["index_bars"],
                industry_exposures=portfolio_exposures,
                style_exposures=portfolio_styles,
            )
            optimizer = ActivePortfolioOptimizer(
                self.config.optimizer,
                self.config.research,
                self.config.risk,
            )
        self._report_progress(progress_callback, "event_backtest", "running")
        backtest = SmokeEventBacktester(
            self.config.research,
            risk_model=risk_model,
            optimizer=optimizer,
        ).run(
            calendar=tables["calendar"],
            benchmark_weights=tables["benchmark_weights"],
            benchmark_membership_intervals=tables.get(
                "benchmark_membership_intervals"
            ),
            index_bars=tables["index_bars"],
            market_panel=market_panel,
            signals=portfolio_signals,
            portfolio_exposures=portfolio_exposures,
            portfolio_styles=portfolio_styles,
            portfolio_restrictions=portfolio_restrictions,
            suspensions=tables.get("suspensions"),
            start_date=signal_dates[0],
            end_date=evaluation_end,
            rebalance_dates=signal_dates,
            progress_callback=progress_callback,
        )
        self._report_progress(
            progress_callback,
            "event_backtest",
            "completed",
            observations=len(backtest.daily),
            optimization_attempts=len(backtest.optimization),
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

    def _validate_prepared_contract(
        self,
        prepared: PreparedResearchData,
    ) -> None:
        provider_spec = self.config.workflow.feature_provider
        provider = self.registry.create_feature_provider(
            provider_spec.name,
            provider_spec.params,
        )
        factor_names = self._factor_names(provider.factor_names)
        directions = {
            factor: int(provider.directions[factor]) for factor in factor_names
        }
        if prepared.factor_names != factor_names or prepared.directions != directions:
            raise ConfigurationError(
                "Prepared feature contract differs from the workflow factor contract"
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

    def _report_progress(
        self,
        callback: ProgressCallback | None,
        stage: str,
        status: str,
        **details: object,
    ) -> None:
        suffix = "".join(
            f" | {key}={value}" for key, value in sorted(details.items())
        )
        LOGGER.info(
            "workflow=%s | stage=%s | status=%s%s",
            self.config.experiment.protocol_id,
            stage,
            status,
            suffix,
        )
        emit_progress(
            callback,
            stage=stage,
            status=status,
            protocol_id=self.config.experiment.protocol_id,
            **details,
        )


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


def _style_exposures(features: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time beta and standardized style audit exposures."""

    keys = {"decision_date", "instrument"}
    if features.empty or not keys.issubset(features.columns):
        return pd.DataFrame()
    candidates = {
        "market_beta_60": "beta_60",
        "small_size_z": "size__z",
        "momentum_120_20_z": "momentum_120_20__z",
        "low_idio_volatility_z": "low_idio_vol_60__z",
        "turnover_z": "free_turnover_20__z",
        "value_z": "book_to_price__z",
    }
    available = {
        output: source
        for output, source in candidates.items()
        if source in features.columns
    }
    if not available:
        return pd.DataFrame()
    result = features[["decision_date", "instrument", *available.values()]].copy()
    result = result.rename(
        columns={
            "decision_date": "trade_date",
            **{source: output for output, source in available.items()},
        }
    )
    for column in available:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
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
