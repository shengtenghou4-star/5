import hashlib
import io
import socket
import ssl
import urllib.error
import zipfile
from datetime import datetime, timezone

from fencha.datasets.gdelt_download import _error_label, download_one_cached

UTC = timezone.utc


def _row() -> str:
    values = [""] * 61
    values[25] = "1"
    values[28] = "14"
    values[29] = "3"
    values[30] = "-2"
    values[33] = "4"
    values[34] = "-6"
    values[53] = "UK"
    return "\t".join(values) + "\n"


def _payload() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("sample.export.CSV", _row())
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: bytes, url: str | None = None) -> None:
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload

    def geturl(self) -> str:
        assert self.url is not None
        return self.url


def test_verified_cache_avoids_network(tmp_path) -> None:
    requested = datetime(2022, 1, 2, 12, tzinfo=UTC)
    filename = "20220102120000.export.CSV.zip"
    payload = _payload()
    archive = tmp_path / filename
    archive.write_bytes(payload)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )

    slices, record = download_one_cached(
        requested,
        ["GBR"],
        base_url="https://invalid.example/gdeltv2",
        cache_dir=tmp_path,
        retries=0,
    )

    assert record.status == "ok"
    assert record.error == "cache_hit"
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert slices[0].country_code == "GBR"
    assert slices[0].stats.events == 1


def test_invalid_cache_is_replaced_atomically(tmp_path, monkeypatch) -> None:
    requested = datetime(2022, 1, 2, 12, tzinfo=UTC)
    filename = "20220102120000.export.CSV.zip"
    archive = tmp_path / filename
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    archive.write_bytes(b"not a zip")
    checksum.write_text("deadbeef\n", encoding="ascii")
    payload = _payload()

    monkeypatch.setattr(
        "fencha.datasets.gdelt_download.urllib.request.urlopen",
        lambda request, **_kwargs: _Response(payload, request.full_url),
    )

    slices, record = download_one_cached(
        requested,
        ["GBR"],
        base_url="https://example.test/gdeltv2",
        cache_dir=tmp_path,
        retries=0,
    )

    assert record.status == "ok"
    assert record.error is None
    assert slices[0].stats.events == 1
    assert archive.read_bytes() == payload
    assert checksum.read_text(encoding="ascii").strip() == hashlib.sha256(
        payload
    ).hexdigest()


def test_official_certificate_failure_uses_host_restricted_fallback(
    tmp_path, monkeypatch
) -> None:
    requested = datetime(2022, 1, 2, 12, tzinfo=UTC)
    payload = _payload()
    calls: list[object | None] = []

    def urlopen(request, **kwargs):
        context = kwargs.get("context")
        calls.append(context)
        if context is None:
            error = ssl.SSLCertVerificationError(1, "hostname mismatch")
            raise urllib.error.URLError(error)
        return _Response(payload, request.full_url)

    monkeypatch.setattr(
        "fencha.datasets.gdelt_download.urllib.request.urlopen", urlopen
    )
    slices, record = download_one_cached(
        requested,
        ["GBR"],
        base_url="https://data.gdeltproject.org/gdeltv2",
        cache_dir=tmp_path,
        retries=0,
        allow_official_tls_fallback=True,
    )

    assert len(calls) == 2
    assert calls[0] is None
    assert isinstance(calls[1], ssl.SSLContext)
    assert record.status == "ok"
    assert record.error == "official_tls_fallback"
    assert slices[0].country_code == "GBR"


def test_tls_fallback_is_never_used_for_nonofficial_host(monkeypatch) -> None:
    requested = datetime(2022, 1, 2, 12, tzinfo=UTC)
    calls = 0

    def urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        error = ssl.SSLCertVerificationError(1, "hostname mismatch")
        raise urllib.error.URLError(error)

    monkeypatch.setattr(
        "fencha.datasets.gdelt_download.urllib.request.urlopen", urlopen
    )
    slices, record = download_one_cached(
        requested,
        ["GBR"],
        base_url="https://mirror.example/gdeltv2",
        retries=0,
        allow_official_tls_fallback=True,
    )

    assert calls == 4  # four quarter-hour candidate URLs, no fallback retries
    assert slices == []
    assert record.status == "missing"
    assert record.error.startswith("tls_error:")


def test_official_tls_fallback_rejects_cross_host_redirect(monkeypatch) -> None:
    requested = datetime(2022, 1, 2, 12, tzinfo=UTC)

    def urlopen(request, **kwargs):
        if kwargs.get("context") is None:
            error = ssl.SSLCertVerificationError(1, "hostname mismatch")
            raise urllib.error.URLError(error)
        return _Response(_payload(), "https://evil.example/payload.zip")

    monkeypatch.setattr(
        "fencha.datasets.gdelt_download.urllib.request.urlopen", urlopen
    )
    slices, record = download_one_cached(
        requested,
        ["GBR"],
        base_url="https://data.gdeltproject.org/gdeltv2",
        retries=0,
        allow_official_tls_fallback=True,
    )

    assert slices == []
    assert record.status == "missing"
    assert record.error.startswith("tls_error:")
    assert "refused redirect" in record.error


def test_error_categories_are_stable() -> None:
    assert _error_label(
        urllib.error.HTTPError("https://x", 429, "rate", {}, None)
    ) == "http_429"
    assert _error_label(
        urllib.error.URLError(socket.gaierror(-2, "name not known"))
    ) == "dns_error"
