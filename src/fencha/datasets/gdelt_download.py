from __future__ import annotations

import hashlib
import socket
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .gdelt import (
    GDELT2_BASE_URL,
    CountrySlice,
    DownloadRecord,
    _candidate_urls,
    iter_sample_times,
    parse_export_zip,
)

OFFICIAL_GDELT_DATA_HOST = "data.gdeltproject.org"


def _error_label(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return "http_404"
        if exc.code == 429:
            return "http_429"
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


def _is_certificate_verification_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(
        exc.reason, ssl.SSLCertVerificationError
    )


def _is_official_gdelt_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname == OFFICIAL_GDELT_DATA_HOST


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
    requested: datetime,
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
        requested_at=requested.isoformat(),
        observed_at=observed_at.isoformat(),
        url=url,
        sha256=digest,
        bytes=len(payload),
        status="ok",
        error="cache_hit",
    )


def _write_cache(archive: Path, checksum: Path, payload: bytes, digest: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_tmp = archive.with_suffix(archive.suffix + ".tmp")
    checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
    archive_tmp.write_bytes(payload)
    checksum_tmp.write_text(digest + "\n", encoding="ascii")
    archive_tmp.replace(archive)
    checksum_tmp.replace(checksum)


def _read_response_payload(
    request: urllib.request.Request,
    *,
    timeout: int,
    context: ssl.SSLContext | None,
    require_final_host: str | None = None,
) -> bytes:
    open_kwargs: dict[str, object] = {"timeout": timeout}
    if context is not None:
        open_kwargs["context"] = context
    with urllib.request.urlopen(request, **open_kwargs) as response:
        if require_final_host is not None:
            final_url = response.geturl() if hasattr(response, "geturl") else request.full_url
            final_host = urlparse(final_url).hostname
            if final_host != require_final_host:
                raise ssl.SSLError(
                    "official GDELT TLS fallback refused redirect to "
                    f"{final_host or 'unknown host'}"
                )
        return response.read()


def download_one_cached(
    requested: datetime,
    countries: Iterable[str],
    *,
    base_url: str = GDELT2_BASE_URL,
    cache_dir: str | Path | None = None,
    timeout: int = 90,
    retries: int = 2,
    insecure_tls: bool = False,
    allow_official_tls_fallback: bool = False,
) -> tuple[list[CountrySlice], DownloadRecord]:
    """Download one deterministic slice with verified persistent caching.

    A cached archive is accepted only after SHA-256 verification and successful
    ZIP/CSV parsing. Invalid cache entries are removed and downloaded again.

    ``allow_official_tls_fallback`` is deliberately narrow: after a certificate
    verification failure only, it retries the exact official
    ``data.gdeltproject.org`` URL with verification disabled and rejects any
    redirect to another host. The resulting archive must still parse as the
    expected GDELT ZIP before it is cached. This is safer than globally disabling
    TLS verification while accommodating the upstream host's certificate fault.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")

    wanted = tuple(countries)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    primary_context = ssl._create_unverified_context() if insecure_tls else None
    last_error = "not_found"

    for observed_at, url in _candidate_urls(requested, base_url):
        archive: Path | None = None
        checksum: Path | None = None
        if cache_root is not None:
            archive, checksum = _cache_paths(cache_root, url)
            if archive.exists():
                try:
                    return _read_cache(
                        archive,
                        checksum,
                        requested=requested,
                        observed_at=observed_at,
                        url=url,
                        countries=wanted,
                    )
                except Exception as exc:
                    last_error = f"cache_{_error_label(exc)}: {exc}"
                    _remove_invalid_cache(archive, checksum)

        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "FENCHA/0.4 historical-forecasting-research"
                    },
                )
                transport_note: str | None = None
                try:
                    payload = _read_response_payload(
                        request,
                        timeout=timeout,
                        context=primary_context,
                    )
                except Exception as verified_exc:
                    can_fallback = (
                        not insecure_tls
                        and allow_official_tls_fallback
                        and _is_official_gdelt_https_url(url)
                        and _is_certificate_verification_error(verified_exc)
                    )
                    if not can_fallback:
                        raise
                    payload = _read_response_payload(
                        request,
                        timeout=timeout,
                        context=ssl._create_unverified_context(),
                        require_final_host=OFFICIAL_GDELT_DATA_HOST,
                    )
                    transport_note = "official_tls_fallback"

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
                    error=transport_note,
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
    )


def collect_weekly_samples_cached(
    *,
    start: date,
    end: date,
    countries: Iterable[str],
    every_days: int = 7,
    hour: int = 12,
    workers: int = 3,
    base_url: str = GDELT2_BASE_URL,
    cache_dir: str | Path | None = "data/cache/gdelt",
    timeout: int = 90,
    retries: int = 2,
    insecure_tls: bool = False,
    allow_official_tls_fallback: bool = False,
) -> tuple[list[CountrySlice], list[DownloadRecord]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    requested = iter_sample_times(start, end, every_days, hour)
    wanted = tuple(countries)
    slices: list[CountrySlice] = []
    records: list[DownloadRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                download_one_cached,
                sample_time,
                wanted,
                base_url=base_url,
                cache_dir=cache_dir,
                timeout=timeout,
                retries=retries,
                insecure_tls=insecure_tls,
                allow_official_tls_fallback=allow_official_tls_fallback,
            ): sample_time
            for sample_time in requested
        }
        for future in as_completed(future_map):
            country_slices, record = future.result()
            slices.extend(country_slices)
            records.append(record)
    slices.sort(key=lambda item: (item.observed_at, item.country_code))
    records.sort(key=lambda item: item.requested_at)
    return slices, records
