import hashlib
import io
import socket
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

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(
        "fencha.datasets.gdelt_download.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
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


def test_error_categories_are_stable() -> None:
    assert _error_label(
        urllib.error.HTTPError("https://x", 429, "rate", {}, None)
    ) == "http_429"
    assert _error_label(
        urllib.error.URLError(socket.gaierror(-2, "name not known"))
    ) == "dns_error"
