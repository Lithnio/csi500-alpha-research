from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from csi500_alpha.errors import InsufficientTrainingData
from csi500_alpha.workflow.contracts import (
    AlphaModel,
    FactorSelector,
    RefitStateAwareAlphaModel,
    SignalEngineResult,
)
from csi500_alpha.workflow.samples import ResearchSamplePolicy


class WalkForwardSignalEngine:
    """Fit and score chronologically while enforcing label maturity."""

    def __init__(
        self,
        *,
        selector: FactorSelector,
        model_factory: Callable[[], AlphaModel],
        refit_every: int,
        sample_policy: ResearchSamplePolicy | None = None,
        label_column: str = "forward_active_return",
    ) -> None:
        if refit_every < 1:
            raise ValueError("refit_every must be positive")
        self.selector = selector
        self.model_factory = model_factory
        self.refit_every = refit_every
        self.sample_policy = sample_policy
        self.label_column = label_column

    def run(
        self,
        panel: pd.DataFrame,
        factor_names: Sequence[str],
        *,
        prediction_start: str,
        prediction_end: str,
    ) -> SignalEngineResult:
        candidates = tuple(factor_names)
        self._validate_panel(panel, candidates)
        dates = sorted(
            date
            for date in panel["decision_date"].astype(str).unique()
            if prediction_start <= date <= prediction_end
            and (
                self.sample_policy is None
                or self.sample_policy.includes_signal_date(date)
            )
        )
        signals: list[dict[str, object]] = []
        fit_rows: list[dict[str, object]] = []
        current_model: AlphaModel | None = None
        current_factors: tuple[str, ...] = ()
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
            training = self._matured_training(panel, decision_date)
            should_refit = (
                self.sample_policy.should_refit(
                    decision_date=decision_date,
                    phase_position=phase_position,
                    previous_phase=previous_phase,
                    has_component=current_model is not None,
                    refit_every=self.refit_every,
                )
                if self.sample_policy is not None
                else current_model is None or position % self.refit_every == 0
            )
            if should_refit:
                selection = self.selector.select(
                    training,
                    candidates,
                    as_of_date=decision_date,
                )
                candidate_model = self.model_factory()
                if isinstance(candidate_model, RefitStateAwareAlphaModel):
                    candidate_model.inherit_refit_state(current_model)
                max_available = self._maximum_available_date(training)
                try:
                    fit = candidate_model.fit(
                        training,
                        selection.factor_names,
                        label_column=self.label_column,
                        as_of_date=decision_date,
                    )
                    current_model = candidate_model
                    current_factors = selection.factor_names
                    current_fit_date = decision_date
                    current_training_rows = fit.observations
                    fit_rows.append(
                        {
                            "fit_date": decision_date,
                            "experiment_phase": phase,
                            "selector": self.selector.name,
                            "model": candidate_model.name,
                            "status": "fitted",
                            "action": "replace_model",
                            "training_rows": fit.observations,
                            "training_dates": fit.decision_dates,
                            "max_label_available_date": max_available,
                            "min_training_decision_date": self._minimum_decision_date(
                                training
                            ),
                            "max_training_decision_date": self._maximum_decision_date(
                                training
                            ),
                            "selected_factors": json.dumps(
                                selection.factor_names,
                                ensure_ascii=False,
                            ),
                            "selector_diagnostics": json.dumps(
                                dict(selection.diagnostics),
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "model_parameters": json.dumps(
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
                            "selector": self.selector.name,
                            "model": candidate_model.name,
                            "status": "insufficient_training_data",
                            "action": (
                                "keep_previous_model"
                                if current_model is not None
                                else "emit_missing_signal"
                            ),
                            "training_rows": len(training),
                            "training_dates": int(training["decision_date"].nunique()),
                            "max_label_available_date": max_available,
                            "min_training_decision_date": self._minimum_decision_date(
                                training
                            ),
                            "max_training_decision_date": self._maximum_decision_date(
                                training
                            ),
                            "selected_factors": json.dumps(
                                selection.factor_names,
                                ensure_ascii=False,
                            ),
                            "selector_diagnostics": json.dumps(
                                dict(selection.diagnostics),
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "model_parameters": "{}",
                            "error": str(exc),
                        }
                    )

            day = panel[panel["decision_date"].astype(str) == decision_date]
            score = (
                current_model.predict(day)
                if current_model is not None
                else pd.Series(np.nan, index=day.index, dtype=float, name="score")
            )
            if not score.index.equals(day.index):
                raise ValueError("Alpha model prediction index must match its input frame")
            score = pd.to_numeric(score, errors="coerce")
            for index, row in day.iterrows():
                value = float(score.loc[index])
                signals.append(
                    {
                        "decision_date": decision_date,
                        "instrument": str(row["instrument"]),
                        "experiment_phase": phase,
                        "score": value if np.isfinite(value) else np.nan,
                        "model": current_model.name if current_model is not None else None,
                        "model_fit_date": current_fit_date,
                        "selected_factor_count": len(current_factors),
                        "training_rows": current_training_rows,
                    }
                )
            previous_phase = phase
            phase_positions[phase] = phase_position + 1

        signal_columns = [
            "decision_date",
            "instrument",
            "experiment_phase",
            "score",
            "model",
            "model_fit_date",
            "selected_factor_count",
            "training_rows",
        ]
        fit_columns = [
            "fit_date",
            "experiment_phase",
            "selector",
            "model",
            "status",
            "action",
            "training_rows",
            "training_dates",
            "max_label_available_date",
            "min_training_decision_date",
            "max_training_decision_date",
            "selected_factors",
            "selector_diagnostics",
            "model_parameters",
            "error",
        ]
        return SignalEngineResult(
            signals=pd.DataFrame(signals, columns=signal_columns),
            model_fits=pd.DataFrame(fit_rows, columns=fit_columns),
        )

    def _matured_training(self, panel: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        if self.sample_policy is not None:
            return self.sample_policy.select_training(
                panel,
                as_of_date=as_of_date,
                label_column=self.label_column,
            )
        available = panel["label_available_date"].notna()
        available &= panel["label_available_date"].fillna("").astype(str) < as_of_date
        available &= panel["decision_date"].astype(str) < as_of_date
        available &= pd.to_numeric(panel[self.label_column], errors="coerce").notna()
        return panel.loc[available].copy()

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

    def _validate_panel(
        self,
        panel: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> None:
        required = {
            "decision_date",
            "instrument",
            "label_available_date",
            self.label_column,
            *(f"{factor}__z" for factor in factor_names),
        }
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(f"Research panel is missing columns: {missing}")
        if panel.duplicated(["decision_date", "instrument"]).any():
            raise ValueError("Research panel decision_date/instrument key is not unique")
