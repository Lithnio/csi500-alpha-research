from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.features.builder import build_raw_factor_panel
from csi500_alpha.features.catalog import DIRECTIONS, FACTOR_NAMES, FAMILIES
from csi500_alpha.workflow.calibration import (
    RobustCrossSectionCalibrator,
    RollingRidgeCalibrator,
)
from csi500_alpha.workflow.contracts import (
    AlphaModel,
    FactorProvider,
    FactorSelector,
    FeatureBuildContext,
    ModelFitSummary,
    ReturnCalibrator,
    SelectionResult,
)
from csi500_alpha.workflow.ic_shrinkage import (
    ICShrinkageAlphaModel,
    ICShrinkageSettings,
)
from csi500_alpha.workflow.selection import (
    StabilityCostSelector,
    StabilityCostSettings,
)

FeatureProviderFactory = Callable[[Mapping[str, Any]], FactorProvider]
SelectorFactory = Callable[[Mapping[str, Any], Mapping[str, int]], FactorSelector]
ModelFactory = Callable[[Mapping[str, Any], Mapping[str, int]], AlphaModel]
CalibratorFactory = Callable[[Mapping[str, Any]], ReturnCalibrator]


class BuiltinDailyFactorProvider:
    name = "builtin_daily"
    factor_names = FACTOR_NAMES
    directions: Mapping[str, int] = DIRECTIONS

    def build_raw(self, context: FeatureBuildContext) -> pd.DataFrame:
        return build_raw_factor_panel(
            market_panel=context.market_panel,
            index_bars=context.index_bars,
            daily_characteristics=context.daily_characteristics,
            benchmark_weights=context.benchmark_weights,
            open_dates=context.open_dates,
            start_date=context.start_date,
            end_date=context.end_date,
            rebalance_every=context.rebalance_every,
            industry_membership=context.industry_membership,
            industry_transition_date=context.industry_transition_date,
        )


class AllFactorsSelector:
    name = "all"

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        del as_of_date
        names = tuple(candidates)
        return SelectionResult(
            factor_names=names,
            diagnostics={
                "candidate_count": len(names),
                "selected_count": len(names),
                "training_rows_seen": len(training),
            },
        )


class CoverageCorrelationSelector:
    name = "coverage_correlation"

    def __init__(self, *, min_coverage: float, max_abs_correlation: float) -> None:
        if not 0 < min_coverage <= 1:
            raise ConfigurationError("selector min_coverage must be in (0, 1]")
        if not 0 < max_abs_correlation <= 1:
            raise ConfigurationError("selector max_abs_correlation must be in (0, 1]")
        self.min_coverage = min_coverage
        self.max_abs_correlation = max_abs_correlation

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        del as_of_date
        names = tuple(candidates)
        if training.empty:
            return SelectionResult(
                factor_names=names,
                diagnostics={"fallback": "empty_training_sample"},
            )
        columns = [f"{factor}__z" for factor in names]
        coverage = training[columns].notna().mean()
        eligible = [
            factor
            for factor in names
            if float(coverage[f"{factor}__z"]) >= self.min_coverage
        ]
        if not eligible:
            eligible = [max(names, key=lambda name: float(coverage[f"{name}__z"]))]
        correlation = training[[f"{factor}__z" for factor in eligible]].corr(
            method="spearman"
        )
        selected: list[str] = []
        for factor in eligible:
            column = f"{factor}__z"
            redundant = any(
                abs(float(correlation.loc[column, f"{kept}__z"]))
                > self.max_abs_correlation
                for kept in selected
                if np.isfinite(correlation.loc[column, f"{kept}__z"])
            )
            if not redundant:
                selected.append(factor)
        return SelectionResult(
            factor_names=tuple(selected),
            diagnostics={
                "candidate_count": len(names),
                "coverage_eligible_count": len(eligible),
                "selected_count": len(selected),
                "min_coverage": self.min_coverage,
                "max_abs_correlation": self.max_abs_correlation,
            },
        )


class DirectionEqualWeightModel:
    name = "direction_equal_weight"

    def __init__(
        self,
        *,
        directions: Mapping[str, int],
        min_factor_fraction: float,
    ) -> None:
        if not 0 < min_factor_fraction <= 1:
            raise ConfigurationError("model min_factor_fraction must be in (0, 1]")
        self.directions = dict(directions)
        self.min_factor_fraction = min_factor_fraction
        self.factor_names: tuple[str, ...] = ()

    def fit(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> ModelFitSummary:
        del label_column, as_of_date
        names = tuple(factor_names)
        missing = sorted(set(names).difference(self.directions))
        if missing:
            raise ConfigurationError(f"Equal-weight model lacks directions: {missing}")
        if not names:
            raise InsufficientTrainingData("No factors were selected")
        self.factor_names = names
        return ModelFitSummary(
            observations=len(training),
            decision_dates=(
                int(training["decision_date"].nunique()) if not training.empty else 0
            ),
            parameters={
                "weights": {
                    factor: self.directions[factor] / len(names) for factor in names
                },
                "min_factor_fraction": self.min_factor_fraction,
            },
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        columns = [f"{factor}__z" for factor in self.factor_names]
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
        directed = matrix.mul(
            [self.directions[factor] for factor in self.factor_names],
            axis="columns",
        )
        minimum = max(1, int(np.ceil(len(columns) * self.min_factor_fraction)))
        score = directed.mean(axis=1, skipna=True)
        return score.where(directed.notna().sum(axis=1) >= minimum).rename("score")


class RidgeAlphaModel:
    name = "ridge"

    def __init__(
        self,
        *,
        alpha: float,
        min_training_rows: int,
        min_training_dates: int,
        min_factor_fraction: float,
    ) -> None:
        if alpha < 0:
            raise ConfigurationError("ridge alpha cannot be negative")
        if min_training_rows < 1 or min_training_dates < 1:
            raise ConfigurationError("ridge minimum training sizes must be positive")
        if not 0 < min_factor_fraction <= 1:
            raise ConfigurationError("model min_factor_fraction must be in (0, 1]")
        self.alpha = alpha
        self.min_training_rows = min_training_rows
        self.min_training_dates = min_training_dates
        self.min_factor_fraction = min_factor_fraction
        self.factor_names: tuple[str, ...] = ()
        self.estimator: Ridge | None = None

    def fit(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> ModelFitSummary:
        del as_of_date
        names = tuple(factor_names)
        if not names:
            raise InsufficientTrainingData("No factors were selected")
        columns = [f"{factor}__z" for factor in names]
        sample = training[["decision_date", label_column, *columns]].copy()
        sample[label_column] = pd.to_numeric(sample[label_column], errors="coerce")
        features = sample[columns].apply(pd.to_numeric, errors="coerce")
        minimum = max(1, int(np.ceil(len(columns) * self.min_factor_fraction)))
        valid = sample[label_column].notna() & features.notna().sum(axis=1).ge(minimum)
        sample = sample.loc[valid]
        features = features.loc[valid].fillna(0.0)
        decision_dates = int(sample["decision_date"].nunique())
        if len(sample) < self.min_training_rows or decision_dates < self.min_training_dates:
            raise InsufficientTrainingData(
                "Ridge requires at least "
                f"{self.min_training_rows} rows and {self.min_training_dates} dates; "
                f"received {len(sample)} rows and {decision_dates} dates"
            )
        date_counts = sample.groupby("decision_date")["decision_date"].transform("size")
        sample_weight = 1.0 / date_counts.to_numpy(dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        estimator = Ridge(alpha=self.alpha, fit_intercept=True)
        estimator.fit(
            features.to_numpy(dtype=float),
            sample[label_column].to_numpy(dtype=float),
            sample_weight=sample_weight,
        )
        self.factor_names = names
        self.estimator = estimator
        return ModelFitSummary(
            observations=len(sample),
            decision_dates=decision_dates,
            parameters={
                "alpha": self.alpha,
                "intercept": float(estimator.intercept_),
                "coefficients": {
                    factor: float(coefficient)
                    for factor, coefficient in zip(names, estimator.coef_, strict=True)
                },
                "min_factor_fraction": self.min_factor_fraction,
            },
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise RuntimeError("Ridge model must be fitted before prediction")
        columns = [f"{factor}__z" for factor in self.factor_names]
        features = frame[columns].apply(pd.to_numeric, errors="coerce")
        minimum = max(1, int(np.ceil(len(columns) * self.min_factor_fraction)))
        valid = features.notna().sum(axis=1).ge(minimum)
        output = pd.Series(np.nan, index=frame.index, name="score", dtype=float)
        if valid.any():
            output.loc[valid] = self.estimator.predict(
                features.loc[valid].fillna(0.0).to_numpy(dtype=float)
            )
        return output


class ResearchComponentRegistry:
    def __init__(self) -> None:
        self._feature_providers: dict[str, FeatureProviderFactory] = {}
        self._selectors: dict[str, SelectorFactory] = {}
        self._models: dict[str, ModelFactory] = {}
        self._calibrators: dict[str, CalibratorFactory] = {}

    def register_feature_provider(
        self,
        name: str,
        factory: FeatureProviderFactory,
    ) -> None:
        self._register(self._feature_providers, name, factory)

    def register_selector(self, name: str, factory: SelectorFactory) -> None:
        self._register(self._selectors, name, factory)

    def register_model(self, name: str, factory: ModelFactory) -> None:
        self._register(self._models, name, factory)

    def register_calibrator(self, name: str, factory: CalibratorFactory) -> None:
        self._register(self._calibrators, name, factory)

    def create_feature_provider(
        self,
        name: str,
        params: Mapping[str, Any],
    ) -> FactorProvider:
        return self._create(self._feature_providers, "feature provider", name)(params)

    def create_selector(
        self,
        name: str,
        params: Mapping[str, Any],
        directions: Mapping[str, int] | None = None,
    ) -> FactorSelector:
        return self._create(self._selectors, "selector", name)(
            params,
            directions or {},
        )

    def create_model(
        self,
        name: str,
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> AlphaModel:
        return self._create(self._models, "model", name)(params, directions)

    def create_calibrator(
        self,
        name: str,
        params: Mapping[str, Any],
    ) -> ReturnCalibrator:
        return self._create(self._calibrators, "calibrator", name)(params)

    @staticmethod
    def _register(registry: dict[str, Any], name: str, factory: Any) -> None:
        if not name or name in registry:
            raise ConfigurationError(f"Duplicate or empty workflow component: {name!r}")
        registry[name] = factory

    @staticmethod
    def _create(registry: Mapping[str, Any], kind: str, name: str) -> Any:
        if name not in registry:
            available = ", ".join(sorted(registry))
            raise ConfigurationError(
                f"Unknown workflow {kind} {name!r}; available: {available}"
            )
        return registry[name]


def _reject_unknown(params: Mapping[str, Any], allowed: set[str], component: str) -> None:
    unknown = sorted(set(params).difference(allowed))
    if unknown:
        raise ConfigurationError(f"Unknown {component} parameters: {unknown}")


def default_component_registry() -> ResearchComponentRegistry:
    registry = ResearchComponentRegistry()

    def builtin_provider(params: Mapping[str, Any]) -> FactorProvider:
        _reject_unknown(params, set(), "builtin_daily")
        return BuiltinDailyFactorProvider()

    def all_selector(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> FactorSelector:
        del directions
        _reject_unknown(params, set(), "all selector")
        return AllFactorsSelector()

    def coverage_selector(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> FactorSelector:
        del directions
        _reject_unknown(
            params,
            {"min_coverage", "max_abs_correlation"},
            "coverage_correlation selector",
        )
        return CoverageCorrelationSelector(
            min_coverage=float(params.get("min_coverage", 0.80)),
            max_abs_correlation=float(params.get("max_abs_correlation", 0.95)),
        )

    def stability_cost_selector(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> FactorSelector:
        allowed = set(StabilityCostSettings.__dataclass_fields__)
        _reject_unknown(params, allowed, "stability_cost selector")
        return StabilityCostSelector(
            directions=directions,
            families=FAMILIES,
            settings=StabilityCostSettings(
                min_coverage=float(params.get("min_coverage", 0.70)),
                min_cross_section=int(params.get("min_cross_section", 30)),
                min_ic_dates=int(params.get("min_ic_dates", 24)),
                min_mean_directed_ic=float(
                    params.get("min_mean_directed_ic", 0.0)
                ),
                min_direction_consistency=float(
                    params.get("min_direction_consistency", 0.50)
                ),
                segments=int(params.get("segments", 4)),
                min_segment_selection_frequency=float(
                    params.get("min_segment_selection_frequency", 0.50)
                ),
                min_newey_west_t=float(params.get("min_newey_west_t", 0.0)),
                min_quintile_monotonicity=float(
                    params.get("min_quintile_monotonicity", 0.0)
                ),
                max_score_churn=float(params.get("max_score_churn", 0.35)),
                max_abs_correlation=float(
                    params.get("max_abs_correlation", 0.85)
                ),
                min_factors=int(params.get("min_factors", 4)),
                max_factors=int(params.get("max_factors", 10)),
                max_per_family=int(params.get("max_per_family", 2)),
                lookback_dates=int(params.get("lookback_dates", 156)),
                churn_penalty=float(params.get("churn_penalty", 0.05)),
            ),
        )

    def equal_model(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> AlphaModel:
        _reject_unknown(
            params,
            {"min_factor_fraction", "directions"},
            "direction_equal_weight model",
        )
        overrides = params.get("directions", {})
        if not isinstance(overrides, dict):
            raise ConfigurationError("model directions must be a mapping")
        merged_directions = {
            **directions,
            **{str(name): int(value) for name, value in overrides.items()},
        }
        return DirectionEqualWeightModel(
            directions=merged_directions,
            min_factor_fraction=float(params.get("min_factor_fraction", 0.50)),
        )

    def ridge_model(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> AlphaModel:
        del directions
        _reject_unknown(
            params,
            {
                "alpha",
                "min_training_rows",
                "min_training_dates",
                "min_factor_fraction",
            },
            "ridge model",
        )
        return RidgeAlphaModel(
            alpha=float(params.get("alpha", 10.0)),
            min_training_rows=int(params.get("min_training_rows", 1000)),
            min_training_dates=int(params.get("min_training_dates", 12)),
            min_factor_fraction=float(params.get("min_factor_fraction", 0.80)),
        )

    def ic_shrinkage_model(
        params: Mapping[str, Any],
        directions: Mapping[str, int],
    ) -> AlphaModel:
        _reject_unknown(
            params,
            set(ICShrinkageSettings.__dataclass_fields__),
            "ic_shrinkage model",
        )
        shrinkage_enabled = params.get("shrinkage_enabled", True)
        if not isinstance(shrinkage_enabled, bool):
            raise ConfigurationError("model shrinkage_enabled must be a boolean")
        raw_solvers = params.get("solvers", ("CLARABEL", "OSQP"))
        if isinstance(raw_solvers, str) or not isinstance(raw_solvers, Sequence):
            raise ConfigurationError("model solvers must be a sequence of names")
        if any(not isinstance(solver, str) or not solver for solver in raw_solvers):
            raise ConfigurationError("model solvers must contain non-empty names")
        solvers = tuple(raw_solvers)
        return ICShrinkageAlphaModel(
            directions=directions,
            settings=ICShrinkageSettings(
                min_cross_section=int(params.get("min_cross_section", 100)),
                min_ic_dates=int(params.get("min_ic_dates", 52)),
                min_churn_dates=int(params.get("min_churn_dates", 12)),
                lookback_dates=int(params.get("lookback_dates", 156)),
                hac_max_lags=int(params.get("hac_max_lags", 4)),
                min_mean_directed_ic=float(
                    params.get("min_mean_directed_ic", 0.0)
                ),
                shrinkage_enabled=shrinkage_enabled,
                prior_variance_floor=float(
                    params.get("prior_variance_floor", 1e-8)
                ),
                correlation_penalty=float(
                    params.get("correlation_penalty", 0.005)
                ),
                cost_penalty=float(params.get("cost_penalty", 0.01)),
                weight_turnover_penalty=float(
                    params.get("weight_turnover_penalty", 0.01)
                ),
                max_factor_weight=float(params.get("max_factor_weight", 0.35)),
                min_active_factors=int(params.get("min_active_factors", 3)),
                min_factor_fraction=float(params.get("min_factor_fraction", 0.50)),
                solvers=solvers,
                feasibility_tolerance=float(
                    params.get("feasibility_tolerance", 1e-7)
                ),
            ),
        )

    def robust_calibrator(params: Mapping[str, Any]) -> ReturnCalibrator:
        _reject_unknown(
            params,
            {"target_scale", "score_clip"},
            "robust_cross_section calibrator",
        )
        return RobustCrossSectionCalibrator(
            target_scale=float(params.get("target_scale", 0.01)),
            score_clip=float(params.get("score_clip", 3.0)),
        )

    def rolling_ridge_calibrator(params: Mapping[str, Any]) -> ReturnCalibrator:
        _reject_unknown(
            params,
            {
                "alpha",
                "min_training_rows",
                "min_training_dates",
                "score_clip",
                "label_clip",
                "max_abs_slope",
                "max_abs_expected_return",
            },
            "rolling_ridge calibrator",
        )
        return RollingRidgeCalibrator(
            alpha=float(params.get("alpha", 10.0)),
            min_training_rows=int(params.get("min_training_rows", 1000)),
            min_training_dates=int(params.get("min_training_dates", 12)),
            score_clip=float(params.get("score_clip", 3.0)),
            label_clip=float(params.get("label_clip", 0.20)),
            max_abs_slope=float(params.get("max_abs_slope", 0.05)),
            max_abs_expected_return=float(
                params.get("max_abs_expected_return", 0.10)
            ),
        )

    registry.register_feature_provider("builtin_daily", builtin_provider)
    registry.register_selector("all", all_selector)
    registry.register_selector("coverage_correlation", coverage_selector)
    registry.register_selector("stability_cost", stability_cost_selector)
    registry.register_model("direction_equal_weight", equal_model)
    registry.register_model("ridge", ridge_model)
    registry.register_model("ic_shrinkage", ic_shrinkage_model)
    registry.register_calibrator("robust_cross_section", robust_calibrator)
    registry.register_calibrator("rolling_ridge", rolling_ridge_calibrator)
    return registry
