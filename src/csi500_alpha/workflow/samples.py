from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from csi500_alpha.config import ExperimentSettings
from csi500_alpha.errors import ConfigurationError


@dataclass(frozen=True)
class ResearchSamplePolicy:
    """One chronological sample contract shared by model and calibration stages.

    Training dates are explicit.  Label maturity supplies the purge rule, while
    ``embargo_days`` removes the final open days immediately before a holdout
    boundary.  Validation and test components are fitted once at their boundary
    and then held fixed throughout that phase.
    """

    settings: ExperimentSettings
    open_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        dates = tuple(str(date) for date in self.open_dates)
        if dates != tuple(sorted(set(dates))):
            raise ConfigurationError("Sample-policy open dates must be sorted and unique")
        object.__setattr__(self, "open_dates", dates)
        if self.settings.enabled:
            required = {
                self.settings.train_start,
                self.settings.validation_start,
            }
            if self.settings.stage == "frozen_test":
                required.add(self.settings.test_start)
            missing = sorted(required.difference(dates))
            if missing:
                raise ConfigurationError(
                    f"Experiment phase starts must be open trading dates: {missing}"
                )
            boundaries = [self.settings.validation_start]
            if self.settings.stage == "frozen_test":
                boundaries.append(self.settings.test_start)
            for boundary in boundaries:
                self._embargo_cutoff(boundary)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def evaluation_start(self) -> str:
        if self.settings.stage == "validation":
            return self.settings.validation_start
        if self.settings.stage == "frozen_test":
            return self.settings.test_start
        return self.settings.validation_start

    @property
    def evaluation_end(self) -> str:
        if self.settings.stage == "validation":
            return self.settings.validation_end
        if self.settings.stage == "frozen_test":
            return self.settings.test_end
        return self.settings.validation_end

    @property
    def signal_start(self) -> str:
        return self.settings.train_start

    @property
    def signal_end(self) -> str:
        if self.settings.stage == "validation":
            return self.settings.validation_end
        if self.settings.stage == "frozen_test":
            return self.settings.test_end
        return self.settings.validation_end

    def phase_for(self, decision_date: str) -> str:
        date = str(decision_date)
        if not self.enabled:
            return "walk_forward"
        if self.settings.train_start <= date <= self.settings.train_end:
            return "train"
        if self.settings.validation_start <= date <= self.settings.validation_end:
            return "validation"
        if self.settings.test_start <= date <= self.settings.test_end:
            return "test"
        raise ValueError(f"Date is outside configured experiment phases: {date}")

    def includes_signal_date(self, decision_date: str) -> bool:
        if not self.enabled:
            return True
        date = str(decision_date)
        in_train = self.settings.train_start <= date <= self.settings.train_end
        in_validation = (
            self.settings.validation_start <= date <= self.settings.validation_end
        )
        if self.settings.stage == "validation":
            return in_train or in_validation
        in_test = self.settings.test_start <= date <= self.settings.test_end
        return in_train or in_validation or in_test

    def select_training(
        self,
        frame: pd.DataFrame,
        *,
        as_of_date: str,
        label_column: str,
        score_column: str | None = None,
    ) -> pd.DataFrame:
        """Return rows permissible at a fit boundary under this protocol."""
        available = frame["label_available_date"].notna()
        available_dates = frame["label_available_date"].fillna("").astype(str)
        decisions = frame["decision_date"].astype(str)
        available &= available_dates < as_of_date
        available &= decisions < as_of_date
        available &= pd.to_numeric(frame[label_column], errors="coerce").notna()
        if score_column is not None:
            available &= pd.to_numeric(frame[score_column], errors="coerce").notna()

        if not self.enabled:
            return frame.loc[available].copy()

        phase = self.phase_for(as_of_date)
        if phase == "train":
            allowed_decisions = (
                decisions.ge(self.settings.train_start)
                & decisions.le(self.settings.train_end)
            )
        elif phase == "validation":
            allowed_decisions = (
                decisions.ge(self.settings.train_start)
                & decisions.le(self.settings.train_end)
            )
            available &= available_dates < self._embargo_cutoff(
                self.settings.validation_start
            )
        else:
            in_train = (
                decisions.ge(self.settings.train_start)
                & decisions.le(self.settings.train_end)
            )
            in_validation = (
                decisions.ge(self.settings.validation_start)
                & decisions.le(self.settings.validation_end)
            )
            allowed_decisions = in_train | in_validation
            available &= available_dates < self._embargo_cutoff(self.settings.test_start)
        return frame.loc[available & allowed_decisions].copy()

    def should_refit(
        self,
        *,
        decision_date: str,
        phase_position: int,
        previous_phase: str | None,
        has_component: bool,
        refit_every: int,
    ) -> bool:
        phase = self.phase_for(decision_date)
        if not has_component or phase != previous_phase:
            return True
        if self.enabled and phase in {"validation", "test"}:
            return False
        return phase_position % refit_every == 0

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self.settings)
        payload.update(
            {
                "purge_rule": "label_available_date_strictly_before_fit",
                "embargo_unit": "open_trading_days",
                "evaluation_start": self.evaluation_start,
                "evaluation_end": self.evaluation_end,
            }
        )
        return payload

    def _embargo_cutoff(self, boundary: str) -> str:
        position = bisect_left(self.open_dates, boundary)
        cutoff_position = position - self.settings.embargo_days
        if cutoff_position < 0 or cutoff_position >= len(self.open_dates):
            raise ConfigurationError(
                f"Not enough open dates for {self.settings.embargo_days}-day embargo "
                f"before {boundary}"
            )
        return self.open_dates[cutoff_position]
