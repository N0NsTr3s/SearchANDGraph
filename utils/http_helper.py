"""
Lightweight HTTP helper providing retries, timeouts and checksum verification.
Intended to be used by `updater.py` and scraper modules.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HTTPClient:
    def __init__(self, timeout: float = 30.0, max_retries: int = 3, backoff: float = 0.3):
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(total=max_retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def download(self, url: str, dest: Path, expected_sha256: Optional[str] = None) -> Path:
        """Download `url` to `dest` atomically. If `expected_sha256` is provided, verify file integrity.

        Raises:
            requests.HTTPError: for non-2xx responses
            ValueError: if checksum mismatch
        Returns:
            Path to final downloaded file (same as `dest`)
        """
        dest = Path(dest)
        tmp = dest.with_suffix(dest.suffix + ".download")
        logger.info("Downloading %s -> %s", url, dest)

        with self.session.get(url, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            h = hashlib.sha256()
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    h.update(chunk)

        digest = h.hexdigest()
        if expected_sha256 and digest != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {digest}")

        tmp.replace(dest)
        logger.info("Downloaded %s (%s)", dest, digest)
        return dest

    def get_json(self, url: str, **kwargs):
        r = self.session.get(url, timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()
