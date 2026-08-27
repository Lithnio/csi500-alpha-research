from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def iter_months(start_date: str, end_date: str) -> Iterator[tuple[str, str]]:
    current = pd.Period(pd.Timestamp(start_date), freq="M")
    final = pd.Period(pd.Timestamp(end_date), freq="M")
    while current <= final:
        yield current.start_time.strftime("%Y%m%d"), current.end_time.strftime("%Y%m%d")
        current += 1


def frame_date_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    for column in ("trade_date", "cal_date", "snapshot_date", "event_date"):
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                return values.min(), values.max()
    return None, None

