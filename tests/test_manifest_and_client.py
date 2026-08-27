import sqlite3
from pathlib import Path

import pandas as pd
import requests

from csi500_alpha.data.client import TushareClient
from csi500_alpha.data.manifest import RequestManifest


def test_client_retries_transient_failure_and_reuses_cache(tmp_path: Path) -> None:
    manifest = RequestManifest(tmp_path / "manifest.sqlite")
    delays: list[float] = []
    client = TushareClient(
        token="secret-token-for-test",
        raw_root=tmp_path / "raw",
        manifest=manifest,
        max_attempts=2,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.01,
        sleeper=delays.append,
    )
    calls = 0

    def fake_query(
        api_name: str, params: dict[str, object], fields: tuple[str, ...]
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.ConnectionError("temporary connection failure")
        return pd.DataFrame([["SSE", "20250102", 1, "20241231"]], columns=list(fields))

    client._query = fake_query  # type: ignore[method-assign]
    fields = ("exchange", "cal_date", "is_open", "pretrade_date")
    first = client.fetch(
        "trade_cal",
        params={"exchange": "SSE", "start_date": "20250102", "end_date": "20250102"},
        fields=fields,
    )
    second = client.fetch(
        "trade_cal",
        params={"exchange": "SSE", "start_date": "20250102", "end_date": "20250102"},
        fields=fields,
    )

    assert calls == 2
    assert len(delays) == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert first.frame.equals(second.frame)
    assert manifest.summary()["success"] == 1

    cached = manifest.cached(first.request_key)
    assert cached is not None
    cached.local_path.write_bytes(b"corrupt parquet cache")
    recovered = client.fetch(
        "trade_cal",
        params={"exchange": "SSE", "start_date": "20250102", "end_date": "20250102"},
        fields=fields,
    )
    assert calls == 3
    assert not recovered.cache_hit
    assert recovered.frame.equals(first.frame)


def test_cache_tag_versions_raw_response_without_reaching_vendor(tmp_path: Path) -> None:
    manifest = RequestManifest(tmp_path / "manifest.sqlite")
    client = TushareClient(
        token="secret-token-for-test",
        raw_root=tmp_path / "raw",
        manifest=manifest,
        max_attempts=1,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.01,
    )
    vendor_params: list[dict[str, object]] = []

    def fake_query(
        api_name: str, params: dict[str, object], fields: tuple[str, ...]
    ) -> pd.DataFrame:
        vendor_params.append(params)
        return pd.DataFrame([["000001.SZ"]], columns=list(fields))

    client._query = fake_query  # type: ignore[method-assign]
    plain = client.fetch(
        "stock_basic",
        params={"list_status": "L"},
        fields=("ts_code",),
    )
    versioned = client.fetch(
        "stock_basic",
        params={"list_status": "L"},
        fields=("ts_code",),
        cache_tag="final-20260630",
    )
    versioned_again = client.fetch(
        "stock_basic",
        params={"list_status": "L"},
        fields=("ts_code",),
        cache_tag="final-20260630",
    )

    assert plain.request_key != versioned.request_key
    assert not plain.cache_hit
    assert not versioned.cache_hit
    assert versioned_again.cache_hit
    assert vendor_params == [{"list_status": "L"}, {"list_status": "L"}]
    with sqlite3.connect(manifest.path) as connection:
        cache_tags = connection.execute(
            "SELECT cache_tag FROM requests ORDER BY cache_tag"
        ).fetchall()
    assert cache_tags == [(None,), ("final-20260630",)]
