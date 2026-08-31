from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from csi500_alpha.logging_utils import ProgressCallback


@dataclass(frozen=True)
class FeatureBuildContext:
    market_panel: pd.DataFrame
    index_bars: pd.DataFrame
    daily_characteristics: pd.DataFrame
    benchmark_weights: pd.DataFrame
    open_dates: list[str]
    start_date: str
    end_date: str
    rebalance_every: int
    industry_membership: pd.DataFrame
    industry_transition_date: str
    financial_tables: Mapping[str, pd.DataFrame] | None = None
    benchmark_membership_intervals: pd.DataFrame | None = None
    progress_callback: ProgressCallback | None = None


class FactorProvider(Protocol):
    name: str
    factor_names: tuple[str, ...]
    directions: Mapping[str, int]

    def build_raw(self, context: FeatureBuildContext) -> pd.DataFrame:
        """Return one row per decision date and instrument with raw factor columns."""


@dataclass(frozen=True)
class SelectionResult:
    factor_names: tuple[str, ...]
    diagnostics: Mapping[str, Any]


class FactorSelector(Protocol):
    name: str

    def select(
        self,
        training: pd.DataFrame,
        candidates: Sequence[str],
        *,
        as_of_date: str,
    ) -> SelectionResult:
        """Select factors using only the supplied matured training sample."""


@dataclass(frozen=True)
class ModelFitSummary:
    observations: int
    decision_dates: int
    parameters: Mapping[str, Any]


class AlphaModel(Protocol):
    name: str

    def fit(
        self,
        training: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        label_column: str,
        as_of_date: str,
    ) -> ModelFitSummary:
        """Fit using a sample whose labels are known strictly before ``as_of_date``."""

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return one score for each row, indexed like ``frame``."""


@runtime_checkable
class RefitStateAwareAlphaModel(Protocol):
    def inherit_refit_state(self, previous_model: AlphaModel | None) -> None:
        """Copy only immutable fit state needed to penalize changes at the next refit."""


@dataclass(frozen=True)
class CalibrationFitSummary:
    observations: int
    decision_dates: int
    parameters: Mapping[str, Any]


class ReturnCalibrator(Protocol):
    name: str

    def fit(
        self,
        training: pd.DataFrame,
        *,
        score_column: str,
        label_column: str,
        as_of_date: str,
    ) -> CalibrationFitSummary:
        """Fit only on rows whose labels are known strictly before ``as_of_date``."""

    def calibrate(self, scores: pd.Series) -> pd.Series:
        """Convert one score cross-section into expected active returns."""


@dataclass(frozen=True)
class SignalEngineResult:
    signals: pd.DataFrame
    model_fits: pd.DataFrame


@dataclass(frozen=True)
class ReturnCalibrationResult:
    signals: pd.DataFrame
    calibration_fits: pd.DataFrame
