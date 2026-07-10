from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    url: str
    revision: str
    sha256: str
    bytes: int
    license: str
    homepage: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_source_lock(root: Path | None = None) -> list[SourceSpec]:
    root = root or project_root()
    lock_path = root / "sources.lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported source lock schema in {lock_path}")
    specs = [SourceSpec(**item) for item in payload["sources"]]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("sources.lock.json contains duplicate source names")
    return specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified(path: Path, spec: SourceSpec) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec.bytes
        and sha256_file(path) == spec.sha256
    )


def _download(spec: SourceSpec, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "acuitybench-rebuild/0.1 (+research reproducibility)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_size = partial.stat().st_size
        actual_hash = sha256_file(partial)
        if actual_size != spec.bytes or actual_hash != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for {spec.name}: expected "
                f"{spec.bytes} bytes/{spec.sha256}, got "
                f"{actual_size} bytes/{actual_hash}"
            )
        os.replace(partial, destination)
    except (OSError, urllib.error.URLError, ValueError):
        partial.unlink(missing_ok=True)
        raise


def fetch_sources(
    data_dir: Path,
    *,
    root: Path | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Path]:
    root = root or project_root()
    cache_dir = data_dir / "cache" / "sources"
    result: dict[str, Path] = {}

    for spec in load_source_lock(root):
        destination = cache_dir / spec.filename
        valid = _is_verified(destination, spec)
        if valid and not refresh:
            print(f"[source] verified cache: {spec.name}")
        elif offline:
            state = "invalid" if destination.exists() else "missing"
            raise FileNotFoundError(
                f"Offline source cache is {state}: {destination} ({spec.name})"
            )
        else:
            reason = "refreshing" if refresh and valid else "fetching"
            print(f"[source] {reason}: {spec.name}")
            try:
                _download(spec, destination)
            except Exception as exc:
                print(f"[source] failed URL: {spec.url}", file=sys.stderr)
                raise RuntimeError(f"Failed to fetch {spec.name}") from exc
        result[spec.name] = destination

    return result


def source_report(
    paths: dict[str, Path], root: Path | None = None
) -> list[dict[str, object]]:
    root = root or project_root()
    by_name = {spec.name: spec for spec in load_source_lock(root)}
    rows: list[dict[str, object]] = []
    for name in sorted(paths):
        spec = by_name[name]
        path = paths[name]
        rows.append(
            {
                "name": name,
                "filename": spec.filename,
                "url": spec.url,
                "revision": spec.revision,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "license": spec.license,
                "homepage": spec.homepage,
            }
        )
    return rows
