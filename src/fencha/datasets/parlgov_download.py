from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .parlgov import BUILDER_VERSION, SnapshotManifest


class ParlGovDownloadError(RuntimeError):
    pass


def _manifest(url: str, path: Path) -> SnapshotManifest:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return SnapshotManifest(
        source_url=url,
        retrieved_at=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        sha256=digest.hexdigest(),
        bytes=total,
        builder_version=BUILDER_VERSION,
    )


def _validate_payload(path: Path, source_url: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("empty ParlGov source")
    suffix = Path(source_url).suffix.lower()
    if suffix == ".zip":
        if not zipfile.is_zipfile(path):
            raise ValueError("ParlGov ZIP is invalid")
        return

    head = path.read_bytes()[:4096].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise ValueError("ParlGov endpoint returned HTML instead of CSV")
    first_line = head.splitlines()[0] if head else b""
    if b"," not in first_line:
        raise ValueError("ParlGov CSV header is missing")


def download_snapshot_cached(
    url: str,
    destination: str | Path,
    *,
    timeout: int = 120,
    retries: int = 3,
    reuse_existing: bool = True,
) -> SnapshotManifest:
    """Download ParlGov atomically, with retries and reusable validated cache."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and destination.exists():
        try:
            _validate_payload(destination, url)
            return _manifest(url, destination)
        except ValueError:
            destination.unlink(missing_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    errors: list[str] = []
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "FENCHA/0.4 historical-forecasting-research"
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            _validate_payload(temporary, url)
            temporary.replace(destination)
            return _manifest(url, destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if isinstance(exc, urllib.error.HTTPError):
                label = f"HTTP {exc.code}"
            else:
                label = type(exc).__name__
            errors.append(f"attempt {attempt + 1}: {label}: {exc}")
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * (2**attempt)))

    raise ParlGovDownloadError(
        f"failed to download ParlGov after {len(errors)} attempt(s): "
        + " | ".join(errors)
    )
