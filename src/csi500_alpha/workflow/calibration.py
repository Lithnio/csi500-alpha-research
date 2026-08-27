from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from csi500_alpha.errors import ConfigurationError, InsufficientTrainingData
from csi500_alpha.workflow.contracts import (
    CalibrationFitSummary,
    ReturnCalibrationResult,
    ReturnCalibrator,
)
from csi500_alpha.workflow.samples import ResearchSamplePolicy


def robust_cross_section(scores: pd.Series, *, clip: float) -> pd.Series:
    """Return a median/MAD standardized cross-section while preserving missingness."""
    numeric = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna()
    output = pd.Series(np.nan, index=scores.index, dtype=float)
    if int(valid.sum()) < 2:
        return output
    sample = numeric.loc[valid]
    median = float(sample.median())
    scale = float((sample - median).abs().median() * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(sample.std(ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        return output
    output.loc[valid] = ((sample - median) / scale).clip(-clip, clip)
    return output


class RobustCrossSectionCalibrator:
    """A stateless baseline mapping robust score z-scores to a fixed return scale."""

    name = "robust_cross_section"

    def __init__(self, *, target_scale: float, score_clip: float) -> None:
        if target_scale <= 0:
            raise ConfigurationError("calibrator target_scale must be positive")
        if score_clip <= 0:
            raise ConfigurationError("calibrator score_clip must be positive")
        self.target_scale = target_scale
        self.score_clip = score_clip

    def fit(
        self,
        training: pd.DataFrame,
        *,
        score_column: str,
        label_column: str,
        as_of_date: str,
    ) -> CalibrationFitSummary:
        del training, score_column, label_column, as_of_date
        return CalibrationFitSummary(
            observations=0,
            decision_dates=0,
            parameters={
                "target_scale": self.target_scale,
                "score_clip": self.score_clip,
                "uses_labels": False,
            },
        )

    def calibrate(self, scores: pd.Series) -> pd.Series:
        return (
            robust_cross_section(scores, clip=self.score_clip) * self.target_scale
        ).rename("expected_return")


class RollingRidgeCalibrator:
    """Estimate return-per-score-z using only matured historical signal outcomes."""

    name = "rolling_ridge"

    def __init__(
        self,
        *,
        alpha: float,
        min_training_rows: int,
        min_training_dates: int,
        score_clip: float,
        label_clip: float,
        max_abs_slope: float,
        max_abs_expected_return: float,
    ) -> None:
        if alpha < 0:
            raise ConfigurationError("calibrator ridge alpha cannot be negative")
        if min_training_rows < 1 or min_training_dates < 1:
            raise ConfigurationError("calibrator minimum training sizes must be positive")
        positive = {
            "score_clip": score_clip,
            "label_clip": label_clip,
            "max_abs_slope": max_abs_slope,
            "max_abs_expected_return": max_abs_expected_return,
        }
        if any(value <= 0 for value in positive.values()):
            raise ConfigurationError("calibrator clipping parameters must be positive")
        self.alpha = alpha
        self.min_training_rows = min_training_rows
        self.min_training_dates = min_training_dates
        self.score_clip = score_clip
        self.label_clip = label_clip
        self.max_abs_slope = max_abs_slope
        self.max_abs_expected_return = max_abs_expected_return
        self.slope: float | None = None

    def fit(
        self,
        training: pd.DataFrame,
        *,
        score_column: str,
        label_column: str,
        as_of_date: str,
    ) -> CalibrationFitSummary:
        self._assert_matured(training, as_of_date)
        required = {"decision_date", score_column, label_column}
        missing = sorted(required.difference(training.columns))
        if missing:
            raise ValueError(f"Calibration training is missing columns: {missing}")

        sample = training[["decision_date", score_column, label_column]].copy()
        sample[score_column] = pd.to_numeric(sample[score_column], errors="coerce")
        sample[label_column] = (
            pd.to_numeric(sample[label_column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(-self.label_clip, self.label_clip)
        )
        standardized = pd.Series(np.nan, index=sample.index, dtype=float)
        for indices in sample.groupby("decision_date", sort=True).groups.values():
            standardized.loc[indices] = robust_cross_section(
                sample.loc[indices, score_column],
                clip=self.score_clip,
            )
        sample["score_z"] = standardized
        sample = sample.dropna(subset=["score_z", label_column])
        decision_dates = int(sample["decision_date"].nunique())
        if len(sample) < self.min_training_rows or decision_dates < self.min_training_dates:
            raise InsufficientTrainingData(
                "Rolling calibration requires at least "
                f"{self.min_training_rows} rows and {self.min_training_dates} dates; "
                f"received {len(sample)} rows and {decision_dates} dates"
            )

        date_counts = sample.groupby("decision_date")["decision_date"].transform("size")
        sample_weight = 1.0 / date_counts.to_numpy(dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        estimator = Ridge(alpha=self.alpha, fit_intercept=False)
        estimator.fit(
            sample[["score_z"]].to_numpy(dtype=float),
            sample[label_column].to_numpy(dtype=float),
            sample_weight=sample_weight,
        )
        raw_slope = float(estimator.coef_[0])
        self.slope = float(np.clip(raw_slope, -self.max_abs_slope, self.max_abs_slope))
        return CalibrationFitSummary(
            observations=len(sample),
            decision_dates=decision_dates,
            parameters={
                "alpha": self.alpha,
                "raw_slope": raw_slope,
                "slope": self.slope,
                "score_clip": self.score_clip,
                "label_clip": self.label_clip,
                "max_abs_expected_return": self.max_abs_expected_return,
                "uses_labels": True,
            },
        )

    def calibrate(self, scores: pd.Series) -> pd.Series:
        if self.slope is None:
            raise RuntimeError("Rolling calibrator must be fitted before calibration")
        return (
            robust_cross_section(scores, clip=self.score_clip)
            .mul(self.slope)
            .clip(-self.max_abs_expected_return, self.max_abs_expected_return)
            .rename("expected_return")
        )

    @staticmethod
    def _assert_matured(training: pd.DataFrame, as_of_date: str) -> None:
        if training.empty:
            return
        if "label_available_date" not in training:
            raise ValueError("Calibration training lacks label_available_date")
        available = training["label_available_date"].fillna("").astype(str)
        decisions = training["decision_date"].astype(str)
        if available.ge(as_of_date).any() or decisions.ge(as_of_date).any():
            raise ValueError("Calibration training contains information unavailable at fit time")


class WalkForwardReturnCalibrationEngine:
    """Calibrate historical model signals with an explicit point-in-time boundary."""

    def __init__(
        self,
        *,
        calibrator_factory: Callable[[], ReturnCalibrator],
        refit_every: int,
        sample_policy: ResearchSamplePolicy | None = None,
        score_column: str = "score",
        label_column: str = "forward_active_return",
    ) -> None:
        if refit_every < 1:
            raise ValueError("refit_every must be positive")
        self.calibrator_factory = calibrator_factory
        self.refit_every = refit_every
        self.sample_policy = sample_policy
        self.score_column = score_column
        self.label_column = label_column

    def run(
        self,
        signals: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        prediction_start: str,
        prediction_end: str,
    ) -> ReturnCalibrationResult:
        self._validate(signals, labels)
        history = signals.merge(
            labels[
                [
                    "decision_date",
                    "instrument",
                    "label_available_date",
                    self.label_column,
                ]
            ],
            on=["decision_date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        all_dates = sorted(history["decision_date"].astype(str).unique())
        dates = [
            date
            for date in all_dates
            if prediction_start <= date <= prediction_end
            and (
                self.sample_policy is None
                or self.sample_policy.includes_signal_date(date)
            )
        ]
        output_rows: list[pd.DataFrame] = []
        fit_rows: list[dict[str, object]] = []
        current: ReturnCalibrator | None = None
        current_fit_date: str | None = None
        current_training_rows = 0
        previous_phase: str | None = None
        phase_positions: dict[str, int] = {}

        for position, decision_date in enumerate(dates):
            phase = (
                self.sample_policy.phase_for(decision_date)
                if self.sample_policy is not None
                else "walk_forward"
            )
            phase_position = phase_positions.get(phase, 0)
            training = self._matured_training(history, decision_date)
            should_refit = (
                self.sample_policy.should_refit(
                    decision_date=decision_date,
                    phase_position=phase_position,
                    previous_phase=previous_phase,
                    has_component=current is not None,
                    refit_every=self.refit_every,
                )
                if self.sample_policy is not None
                else current is None or position % self.refit_every == 0
            )
            if should_refit:
                candidate = self.calibrator_factory()
                maximum_available = self._maximum_available_date(training)
                try:
                    fit = candidate.fit(
                        training,
                        score_column=self.score_column,
                        label_column=self.label_column,
                        as_of_date=decision_date,
                    )
                    current = candidate
                    current_fit_date = decision_date
                    current_training_rows = fit.observations
                    fit_rows.append(
                        {
                            "fit_date": decision_date,
                            "experiment_phase": phase,
                            "calibrator": candidate.name,
                            "status": "fitted",
                            "action": "replace_calibrator",
                            "training_rows": fit.observations,
                            "training_dates": fit.decision_dates,
                            "max_label_available_date": maximum_available,
                            "min_training_decision_date": self._minimum_decision_date(
                                training
                            ),
                            "max_training_decision_date": self._maximum_decision_date(
                                training
                            ),
                            "parameters": json.dumps(
                                dict(fit.parameters),
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "error": "",
                        }
                    )
                except InsufficientTrainingData as exc:
                    fit_rows.append(
                        {
                            "fit_date": decision_date,
                            "experiment_phase": phase,
                            "calibrator": candidate.name,
                            "status": "insufficient_training_data",
                            "action": (
                                "keep_previous_calibrator"
                                if current is not None
                                else "emit_missing_expected_return"
                            ),
                            "training_rows": len(training),
                            "training_dates": int(training["decision_date"].nunique()),
                            "max_label_available_date": maximum_available,
                            "min_training_decision_date": self._minimum_decision_date(
                                training
                            ),
                            "max_training_decision_date": self._maximum_decision_date(
                                training
                            ),
                            "parameters": "{}",
                            "error": str(exc),
                        }
                    )

            day = signals[signals["decision_date"].astype(str) == decision_date].copy()
            expected_return = (
                current.calibrate(day[self.score_column])
                if current is not None
                else pd.Series(np.nan, index=day.index, dtype=float)
            )
            if not expected_return.index.equals(day.index):
                raise ValueError("Calibrator output index must match its score input")
            day["expected_return"] = pd.to_numeric(expected_return, errors="coerce")
            day["calibrator"] = current.name if current is not None else None
            day["calibrator_fit_date"] = current_fit_date
            day["calibration_training_rows"] = current_training_rows
            output_rows.append(day)
            previous_phase = phase
            phase_positions[phase] = phase_position + 1

        signal_columns = [
            *signals.columns,
            "expected_return",
            "calibrator",
            "calibrator_fit_date",
            "calibration_training_rows",
        ]
        fit_columns = [
            "fit_date",
            "experiment_phase",
            "calibrator",
            "status",
            "action",
            "training_rows",
            "training_dates",
            "max_label_available_date",
            "min_training_decision_date",
            "max_training_decision_date",
            "parameters",
            "error",
        ]
        calibrated = (
            pd.concat(output_rows, ignore_index=True)
            if output_rows
            else pd.DataFrame(columns=signal_columns)
        )
        return ReturnCalibrationResult(
            signals=calibrated.loc[:, signal_columns],
            calibration_fits=pd.DataFrame(fit_rows, columns=fit_columns),
        )

    def _matured_training(self, history: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        if self.sample_policy is not None:
            return self.sample_policy.select_training(
                history,
                as_of_date=as_of_date,
                label_column=self.label_column,
                score_column=self.score_column,
            )
        available = history["label_available_date"].notna()
        available &= history["label_available_date"].fillna("").astype(str) < as_of_date
        available &= history["decision_date"].astype(str) < as_of_date
        available &= pd.to_numeric(history[self.score_column], errors="coerce").notna()
        available &= pd.to_numeric(history[self.label_column], errors="coerce").notna()
        return history.loc[available].copy()

    @staticmethod
    def _maximum_available_date(training: pd.DataFrame) -> str | None:
        if training.empty:
            return None
        return str(training["label_available_date"].astype(str).max())

    @staticmethod
    def _minimum_decision_date(training: pd.DataFrame) -> str | None:
        if training.empty:
            return None
        return str(training["decision_date"].astype(str).min())

    @staticmethod
    def _maximum_decision_date(training: pd.DataFrame) -> str | None:
        if training.empty:
            return None
        return str(training["decision_date"].astype(str).max())

    def _validate(self, signals: pd.DataFrame, labels: pd.DataFrame) -> None:
        signal_required = {"decision_date", "instrument", self.score_column}
        label_required = {
            "decision_date",
            "instrument",
            "label_available_date",
            self.label_column,
        }
        missing_signals = sorted(signal_required.difference(signals.columns))
        missing_labels = sorted(label_required.difference(labels.columns))
        if missing_signals:
            raise ValueError(f"Signals are missing columns: {missing_signals}")
        if missing_labels:
            raise ValueError(f"Labels are missing columns: {missing_labels}")
        keys = ["decision_date", "instrument"]
        if signals.duplicated(keys).any() or labels.duplicated(keys).any():
            raise ValueError("Signal and label keys must be unique")
