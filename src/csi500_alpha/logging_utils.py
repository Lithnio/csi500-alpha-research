from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

ProgressCallback = Callable[[Mapping[str, Any]], None]
LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


class ProgressLogger:
    """Emit rate-limited progress logs and optional machine-readable events."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        stage: str,
        total: int,
        min_interval_seconds: float = 30.0,
        callback: ProgressCallback | None = None,
    ) -> None:
        if total < 1:
            raise ValueError("Progress total must be positive")
        if min_interval_seconds < 0:
            raise ValueError("Progress interval cannot be negative")
        self.logger = logger
        self.stage = str(stage)
        self.total = int(total)
        self.min_interval_seconds = float(min_interval_seconds)
        self.callback = callback
        self._started = time.perf_counter()
        self._last_reported = self._started
        self._last_completed = 0

    def update(
        self,
        completed: int,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        completed = int(completed)
        if not 1 <= completed <= self.total or completed < self._last_completed:
            raise ValueError(
                "Progress must be monotone and cannot exceed the configured total"
            )
        now = time.perf_counter()
        should_report = (
            self._last_completed == 0
            or completed == self.total
            or now - self._last_reported >= self.min_interval_seconds
        )
        self._last_completed = completed
        if not should_report:
            return

        elapsed = max(0.0, now - self._started)
        eta = (
            elapsed * (self.total - completed) / completed
            if completed > 0
            else math.inf
        )
        details = dict(context or {})
        suffix = "".join(
            f" | {key}={value}" for key, value in sorted(details.items())
        )
        self.logger.info(
            "%s | %d/%d (%.1f%%) | elapsed=%s | eta=%s%s",
            self.stage,
            completed,
            self.total,
            100.0 * completed / self.total,
            _format_duration(elapsed),
            _format_duration(eta),
            suffix,
        )
        event: dict[str, Any] = {
            "stage": self.stage,
            "status": "completed" if completed == self.total else "running",
            "completed_units": completed,
            "total_units": self.total,
            "progress_fraction": completed / self.total,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            **details,
        }
        _dispatch_progress(self.callback, event)
        self._last_reported = now


def emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    _dispatch_progress(
        callback,
        {"stage": str(stage), "status": str(status), **details},
    )


def _dispatch_progress(
    callback: ProgressCallback | None,
    event: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - observability must not fail research
        LOGGER.warning("Progress callback failed; research continues", exc_info=True)


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
