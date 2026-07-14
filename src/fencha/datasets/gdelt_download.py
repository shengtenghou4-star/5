from __future__ import annotations

import hashlib
import socket
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .gdelt import (
    GDELT2_BASE_URL,
    CountrySlice,
    DownloadRecord,
    _candidate_urls,
    parse_export_zip,
)


def _error_label(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return "http_404"
        if exc.code == 429:
            return "http_429"
        if 500 <= exc.code <= 599:
            return f"http_{exc.code}"
        return f"http_{exc.code}"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, ssl.SSLError):
        return "tls_error"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        if isinstance(reason, socket.gaierror):
            return "dns_error"
        if isinstance(reason, ssl.SSLError):
            return "tls_error"
        return "connection_error"
    if isinstance(exc, zipfile.BadZipFile):
        return "zip_corrupt"
    if isinstance(exc, ValueError):
        return "parse_error"
    return type(exc).__name__.lower()


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    archive = cache_dir / url.rsplit("/", 1)[-1]
    return archive, archive.with_suffix(archive.suffix + ".sha256")


def _remove_invalid_cache(archive: Path, checksum: Path) -> None:
    archive.unlink(missing_ok=True)
    checksum.unlink(missing_ok=True)


def _read_cache(
    archive: Path,
    checksum: Path,
    *,
    observed_at: datetime,
    url: str,
    countries: Iterable[str],
) -> tuple[list[CountrySlice], DownloadRecord]:
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if checksum.exists():
        expected = checksum.read_text(encoding="ascii").strip().lower()
        if expected and expected != digest:
            raise ValueError("cache sha256 mismatch")
    slices = parse_export_zip(
        payload,
        observed_at,
        url.rsplit("/", 1)[-1],
        countries,
    )
    return slices, DownloadRecord(
        requested_at=observed_at.isoformat(),
        observed_at=observed_at.isoformat(),
        url=url,
        sha256=digest,
        bytes=len(payload),
        status="ok",
        error=None,
        cache_hit=True,
    )


def _write_cache(archive: Path, checksum: Path, payload: bytes, digest: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_tmp = archive.with_suffix(archive.suffix + ".tmp")
    checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
    archive_tmp.write_bytes(payload)
    checksum_tmp.write_text(digest + "\n", encoding="ascii")
    archive_tmp.replace(archive)
    checksum_tmp.replace(checksum)


def download_one_cached(
    requested: datetime,
    countries: Iterable[str],
    *,
    base_url: str = GDELT2_BASE_URL,
    cache_dir: str | Path | None = None,
    timeout: int = 90,
    retries: int = 2,
    insecure_tls: bool = False,
) -> tuple[list[CountrySlice], DownloadRecord]:
    """Download one deterministic slice with verified persistent caching.

    A cached archive is accepted only after SHA-256 verification and successful
    ZIP/CSV parsing. Invalid cache entries are removed and downloaded again.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")

    wanted = tuple(countries)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    context = ssl._create_unverified_context() if insecure_tls else None
    last_error = "not_found"

    for observed_at, url in _candidate_urls(requested, base_url):
        archive: Path | None = None
        checksum: Path | None = None
        if cache_root is not None:
            archive, checksum = _cache_paths(cache_root, url)
            if archive.exists():
                try:
                    slices, record = _read_cache(
                        archive,
                        checksum,
                        observed_at=observed_at,
                        url=url,
                        countries=wanted,
                    )
                    return slices, DownloadRecord(
                        requested_at=requested.isoformat(),
                        observed_at=record.observed_at,
                        url=record.url,
                        sha256=record.sha256,
                        bytes=record.bytes,
                        status=record.status,
                        error=record.error,
                        cache_hit=True,
                    )
                except Exception as exc:
                    last_error = f"cache_{_error_label(exc)}: {exc}"
                    _remove_invalid_cache(archive, checksum)

        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "FENCHA/0.3 historical-forecasting-research"
                    },
                )
                open_kwargs: dict[str, object] = {"timeout": timeout}
                if context is not None:
                    open_kwargs["context"] = context
                with urllib.request.urlopen(request, **open_kwargs) as response:
                    payload = response.read()

                digest = hashlib.sha256(payload).hexdigest()
                slices = parse_export_zip(
                    payload,
                    observed_at,
                    url.rsplit("/", 1)[-1],
                    wanted,
                )
                if archive is not None and checksum is not None:
                    _write_cache(archive, checksum, payload, digest)
                return slices, DownloadRecord(
                    requested_at=requested.isoformat(),
                    observed_at=observed_at.isoformat(),
                    url=url,
                    sha256=digest,
                    bytes=len(payload),
                    status="ok",
                    error=None,
                    cache_hit=False,
                )
            except Exception as exc:
                label = _error_label(exc)
                last_error = f"{label}: {exc}"
                if label == "http_404":
                    break
                if attempt < retries:
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
                    continue
                break

    return [], DownloadRecord(
        requested_at=requested.isoformat(),
        observed_at=None,
        url=None,
        sha256=None,
        bytes=0,
        status="missing",
        error=last_error,
        cache_hit=False,
    )
