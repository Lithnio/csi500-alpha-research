from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csi500_alpha.utils import canonical_json, sha256_text, utc_now


@dataclass(frozen=True)
class CachedRequest:
    request_key: str
    status: str
    local_path: Path
    row_count: int
    content_hash: str


class RequestManifest:
    """Durable request state used for cache reuse and interrupted-run recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_key TEXT PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    fields TEXT NOT NULL,
                    cache_tag TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER,
                    min_date TEXT,
                    max_date TEXT,
                    content_hash TEXT,
                    local_path TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(requests)").fetchall()
            }
            if "cache_tag" not in columns:
                connection.execute("ALTER TABLE requests ADD COLUMN cache_tag TEXT")

    @staticmethod
    def request_key(
        api_name: str,
        params: dict[str, Any],
        fields: tuple[str, ...],
        *,
        cache_tag: str | None = None,
    ) -> str:
        payload = {"api_name": api_name, "params": params, "fields": fields}
        if cache_tag is not None:
            payload["cache_tag"] = cache_tag
        return sha256_text(canonical_json(payload))

    def cached(self, request_key: str) -> CachedRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_key, status, local_path,
                       COALESCE(row_count, 0) AS row_count,
                       COALESCE(content_hash, '') AS content_hash
                FROM requests WHERE request_key = ?
                """,
                (request_key,),
            ).fetchone()
        if row is None or row["status"] not in {"success", "empty"} or not row["local_path"]:
            return None
        path = Path(row["local_path"])
        if not path.exists():
            return None
        return CachedRequest(
            request_key=row["request_key"],
            status=row["status"],
            local_path=path,
            row_count=int(row["row_count"]),
            content_hash=str(row["content_hash"]),
        )

    def mark_started(
        self,
        request_key: str,
        api_name: str,
        params: dict[str, Any],
        fields: tuple[str, ...],
        cache_tag: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO requests (
                    request_key, api_name, params_json, fields, cache_tag,
                    status, attempts, started_at
                ) VALUES (?, ?, ?, ?, ?, 'running', 1, ?)
                ON CONFLICT(request_key) DO UPDATE SET
                    status = 'running',
                    attempts = requests.attempts + 1,
                    started_at = excluded.started_at,
                    cache_tag = excluded.cache_tag,
                    error_type = NULL,
                    error_message = NULL
                """,
                (
                    request_key,
                    api_name,
                    canonical_json(params),
                    ",".join(fields),
                    cache_tag,
                    utc_now(),
                ),
            )

    def mark_success(
        self,
        request_key: str,
        *,
        row_count: int,
        min_date: str | None,
        max_date: str | None,
        content_hash: str,
        local_path: Path,
    ) -> None:
        status = "empty" if row_count == 0 else "success"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE requests SET
                    status = ?, row_count = ?, min_date = ?, max_date = ?,
                    content_hash = ?, local_path = ?, finished_at = ?,
                    error_type = NULL, error_message = NULL
                WHERE request_key = ?
                """,
                (
                    status,
                    row_count,
                    min_date,
                    max_date,
                    content_hash,
                    str(local_path.resolve()),
                    utc_now(),
                    request_key,
                ),
            )

    def mark_failure(
        self,
        request_key: str,
        *,
        retryable: bool,
        error_type: str,
        error_message: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE requests SET
                    status = ?, error_type = ?, error_message = ?, finished_at = ?
                WHERE request_key = ?
                """,
                (
                    "retryable_error" if retryable else "fatal_error",
                    error_type,
                    error_message,
                    utc_now(),
                    request_key,
                ),
            )

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM requests GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
