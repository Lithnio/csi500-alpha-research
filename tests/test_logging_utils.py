from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from csi500_alpha.logging_utils import ProgressLogger


def test_progress_logger_rate_limits_and_always_reports_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[Mapping[str, Any]] = []
    logger = logging.getLogger("tests.progress")
    progress = ProgressLogger(
        logger,
        stage="signals",
        total=3,
        min_interval_seconds=3600.0,
        callback=events.append,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        progress.update(1, context={"decision_date": "20200102"})
        progress.update(2, context={"decision_date": "20200103"})
        progress.update(3, context={"decision_date": "20200106"})

    assert [event["completed_units"] for event in events] == [1, 3]
    assert [event["status"] for event in events] == ["running", "completed"]
    assert "signals | 1/3" in caplog.messages[0]
    assert "signals | 3/3" in caplog.messages[-1]


def test_progress_logger_rejects_non_monotone_updates() -> None:
    progress = ProgressLogger(
        logging.getLogger("tests.progress"),
        stage="signals",
        total=3,
    )
    progress.update(2)

    with pytest.raises(ValueError, match="monotone"):
        progress.update(1)


def test_progress_callback_failure_does_not_fail_research(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_: Mapping[str, Any]) -> None:
        raise OSError("monitor unavailable")

    progress = ProgressLogger(
        logging.getLogger("tests.progress"),
        stage="signals",
        total=1,
        callback=fail,
    )

    with caplog.at_level(logging.WARNING):
        progress.update(1)

    assert "Progress callback failed" in caplog.text
