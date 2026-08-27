from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import tushare as ts

from csi500_alpha.data.manifest import RequestManifest
from csi500_alpha.data.storage import write_parquet_atomic
from csi500_alpha.errors import DataFetchError
from csi500_alpha.utils import frame_date_bounds, sha256_file

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    request_key: str
    cache_hit: bool


class TushareClient:
    """Document-oriented Tushare wrapper with durable cache and bounded retries."""

    def __init__(
        self,
        *,
        token: str,
        raw_root: Path,
        manifest: RequestManifest,
        max_attempts: int,
        backoff_base_seconds: float,
        backoff_max_seconds: float,
        request_timeout_seconds: float = 30.0,
        min_request_interval_seconds: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._pro = ts.pro_api(token, timeout=request_timeout_seconds)
        self.raw_root = raw_root
        self.manifest = manifest
        self.max_attempts = max_attempts
        self.request_timeout_seconds = request_timeout_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._sleeper = sleeper
        self._last_request_started: float | None = None

    def fetch(
        self,
        api_name: str,
        *,
        params: dict[str, Any],
        fields: tuple[str, ...],
        force: bool = False,
        cache_tag: str | None = None,
    ) -> FetchResult:
        clean_params = {key: value for key, value in params.items() if value is not None}
        request_key = self.manifest.request_key(
            api_name,
            clean_params,
            fields,
            cache_tag=cache_tag,
        )
        if not force:
            cached = self.manifest.cached(request_key)
            if cached is not None:
                try:
                    cache_valid = bool(cached.content_hash)
                    cache_valid &= sha256_file(cached.local_path) == cached.content_hash
                    if cache_valid:
                        frame = pd.read_parquet(cached.local_path)
                        cache_valid &= len(frame) == cached.row_count
                        cache_valid &= set(fields).issubset(frame.columns)
                        if cache_valid:
                            return FetchResult(
                                frame=frame.loc[:, list(fields)].copy(),
                                request_key=request_key,
                                cache_hit=True,
                            )
                except Exception:  # cache corruption must fall back to the vendor
                    cache_valid = False
                LOGGER.warning(
                    "Ignoring invalid cache entry for api=%s request=%s",
                    api_name,
                    request_key[:12],
                )

        for attempt in range(1, self.max_attempts + 1):
            self.manifest.mark_started(
                request_key,
                api_name,
                clean_params,
                fields,
                cache_tag,
            )
            try:
                self._throttle()
                frame = self._query(api_name, clean_params, fields)
                path = self.raw_root / api_name / f"{request_key}.parquet"
                content_hash = write_parquet_atomic(frame, path)
                min_date, max_date = frame_date_bounds(frame)
                self.manifest.mark_success(
                    request_key,
                    row_count=len(frame),
                    min_date=min_date,
                    max_date=max_date,
                    content_hash=content_hash,
                    local_path=path,
                )
                return FetchResult(frame=frame, request_key=request_key, cache_hit=False)
            except Exception as exc:
                retryable = self._is_retryable(exc)
                message = self._sanitize(str(exc))
                self.manifest.mark_failure(
                    request_key,
                    retryable=retryable,
                    error_type=type(exc).__name__,
                    error_message=message[:1000],
                )
                if not retryable or attempt >= self.max_attempts:
                    raise DataFetchError(
                        f"Tushare request failed: api={api_name}, attempt={attempt}, "
                        f"retryable={retryable}, error={message[:240]}"
                    ) from exc
                delay = min(
                    self.backoff_max_seconds,
                    self.backoff_base_seconds * (2 ** (attempt - 1)),
                )
                delay *= random.uniform(0.8, 1.2)
                LOGGER.warning(
                    "Transient Tushare failure for %s; retrying attempt %d/%d in %.2fs",
                    api_name,
                    attempt + 1,
                    self.max_attempts,
                    delay,
                )
                self._sleeper(delay)

        raise AssertionError("Retry loop terminated unexpectedly")

    def _throttle(self) -> None:
        now = time.monotonic()
        if self._last_request_started is not None:
            elapsed = now - self._last_request_started
            remaining = self.min_request_interval_seconds - elapsed
            if remaining > 0:
                self._sleeper(remaining)
        self._last_request_started = time.monotonic()

    def _query(
        self, api_name: str, params: dict[str, Any], fields: tuple[str, ...]
    ) -> pd.DataFrame:
        result = self._pro.query(api_name, fields=",".join(fields), **params)
        if result is None:
            return pd.DataFrame(columns=list(fields))
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame from {api_name}")
        if result.empty:
            return pd.DataFrame(columns=list(fields))
        missing = [field for field in fields if field not in result.columns]
        if missing:
            raise ValueError(f"Tushare schema mismatch for {api_name}; missing fields={missing}")
        return result.loc[:, list(fields)].copy()

    def _sanitize(self, message: str) -> str:
        return message.replace(self._token, "<redacted>") if self._token else message

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (requests.RequestException, TimeoutError, ConnectionError)):
            return True
        message = str(exc).lower()
        fatal_markers = ("token", "权限", "积分", "参数", "字段", "permission")
        if any(marker in message for marker in fatal_markers):
            return False
        retryable_markers = (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "429",
            "频率",
            "每分钟",
            "稍后",
        )
        return any(marker in message for marker in retryable_markers)
