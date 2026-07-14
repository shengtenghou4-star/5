import urllib.error

import pytest

from fencha.datasets import parlgov_download
from fencha.datasets.parlgov_download import (
    ParlGovDownloadError,
    download_snapshot_cached,
)

CSV = b"country_name_short,cabinet_id,cabinet_name,start_date\nGBR,1,Smith I,2020-01-01\n"


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self.sent:
            return b""
        self.sent = True
        return self.payload


def test_valid_cached_csv_avoids_network(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "view_cabinet.csv"
    destination.write_bytes(CSV)

    def forbidden(*args, **kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(parlgov_download.urllib.request, "urlopen", forbidden)
    manifest = download_snapshot_cached(
        "https://parlgov.org/data/view_cabinet.csv",
        destination,
    )

    assert manifest.bytes == len(CSV)
    assert manifest.sha256
    assert destination.read_bytes() == CSV


def test_download_retries_and_writes_atomically(tmp_path, monkeypatch) -> None:
    calls = 0

    def urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary failure")
        return _Response(CSV)

    monkeypatch.setattr(parlgov_download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(parlgov_download.time, "sleep", lambda _: None)
    destination = tmp_path / "view_cabinet.csv"
    manifest = download_snapshot_cached(
        "https://parlgov.org/data/view_cabinet.csv",
        destination,
        retries=2,
        reuse_existing=False,
    )

    assert calls == 2
    assert destination.read_bytes() == CSV
    assert not destination.with_suffix(".csv.tmp").exists()
    assert manifest.bytes == len(CSV)


def test_html_error_page_is_not_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        parlgov_download.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(b"<html>upstream error</html>"),
    )
    destination = tmp_path / "view_cabinet.csv"

    with pytest.raises(ParlGovDownloadError, match="returned HTML"):
        download_snapshot_cached(
            "https://parlgov.org/data/view_cabinet.csv",
            destination,
            retries=0,
            reuse_existing=False,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".csv.tmp").exists()
