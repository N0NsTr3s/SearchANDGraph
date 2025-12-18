from __future__ import annotations

import json
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """Normalize URL for stable de-dupe keys.

    - Strips fragments
    - Lowercases scheme + hostname
    - Removes trailing slash on path (except root)
    """
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        if path != "/":
            path = path.rstrip("/")
        normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        return normalized
    except Exception:
        return url.strip()


def sha256_bytes(data: bytes) -> str:
    h = sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    url: str
    normalized_url: str
    sha256: str
    path: str
    size: int
    created_at: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "normalized_url": self.normalized_url,
            "sha256": self.sha256,
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
        }


class DownloadManifest:
    """A tiny scan-local manifest used to avoid duplicate downloads.

    It supports dedupe by URL and by content hash.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.by_url: Dict[str, ManifestEntry] = {}
        self.by_hash: Dict[str, ManifestEntry] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.manifest_path.exists():
            return

        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return

        entries = raw.get("entries", []) if isinstance(raw, dict) else []
        for e in entries:
            if not isinstance(e, dict):
                continue
            try:
                entry = ManifestEntry(
                    url=str(e.get("url", "")),
                    normalized_url=str(e.get("normalized_url", "")),
                    sha256=str(e.get("sha256", "")),
                    path=str(e.get("path", "")),
                    size=int(e.get("size", 0) or 0),
                    created_at=float(e.get("created_at", 0.0) or 0.0),
                )
            except Exception:
                continue

            if entry.normalized_url:
                self.by_url[entry.normalized_url] = entry
            if entry.sha256:
                self.by_hash[entry.sha256] = entry

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": time.time(),
            "entries": [e.to_json() for e in self.by_url.values()],
        }
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def get_by_url(self, url: str) -> Optional[ManifestEntry]:
        self.load()
        return self.by_url.get(normalize_url(url))

    def get_by_hash(self, content_hash: str) -> Optional[ManifestEntry]:
        self.load()
        return self.by_hash.get(content_hash)

    def upsert(self, *, url: str, content_hash: str, path: Path) -> ManifestEntry:
        self.load()
        normalized = normalize_url(url)
        entry = ManifestEntry(
            url=url,
            normalized_url=normalized,
            sha256=content_hash,
            path=str(path),
            size=path.stat().st_size if path.exists() else 0,
            created_at=time.time(),
        )
        self.by_url[normalized] = entry
        self.by_hash[content_hash] = entry
        self.save()
        return entry
