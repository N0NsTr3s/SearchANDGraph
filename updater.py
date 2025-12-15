"""GitHub Releases-based update checker.

Flow:
- Query GitHub Releases API for latest release
- Compare tag_name (e.g. v1.0.1) to CURRENT_VERSION
- If newer, download an .exe asset (installer) and launch it
- Exit current process so installer can replace files

This is intentionally simple and Windows-focused.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

import requests
from packaging import version as pkg_version

from version import CURRENT_VERSION, REPO_NAME, REPO_OWNER


def check_for_updates(timeout_s: float = 5.0) -> Optional[dict[str, Any]]:
    """Return release JSON if a newer version exists; otherwise None."""

    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        headers = {"User-Agent": f"{REPO_NAME}-updater"}
        response = requests.get(url, headers=headers, timeout=timeout_s)
        if response.status_code != 200:
            return None

        data = response.json()
        latest_tag = str(data.get("tag_name") or "").strip()
        if not latest_tag:
            return None

        latest_clean = latest_tag.lstrip("v").strip()
        if pkg_version.parse(latest_clean) > pkg_version.parse(CURRENT_VERSION):
            return data
    except Exception:
        return None

    return None


def _pick_installer_asset_url(release_data: dict[str, Any]) -> Optional[str]:
    assets = release_data.get("assets") or []
    if not isinstance(assets, list):
        return None

    # Prefer something that looks like an installer, but fall back to any .exe.
    preferred = []
    fallback = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        url = asset.get("browser_download_url")
        if not url:
            continue
        lower = name.lower()
        if not lower.endswith(".exe"):
            continue
        if "setup" in lower or "installer" in lower:
            preferred.append(str(url))
        else:
            fallback.append(str(url))

    return (preferred[0] if preferred else None) or (fallback[0] if fallback else None)


def perform_update(release_data: dict[str, Any], *, silent: bool = True) -> None:
    """Download installer asset to temp, run it, then exit this process."""

    download_url = _pick_installer_asset_url(release_data)
    if not download_url:
        raise RuntimeError("No .exe installer asset found in the latest GitHub release.")

    temp_dir = tempfile.gettempdir()
    installer_path = os.path.join(temp_dir, f"{REPO_NAME}_Update.exe")

    headers = {"User-Agent": f"{REPO_NAME}-updater"}
    r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
    r.raise_for_status()

    with open(installer_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    args = [installer_path]
    if silent:
        # Common Inno Setup flags
        args += ["/SILENT", "/NORESTART"]

    subprocess.Popen(args, close_fds=True)
    sys.exit(0)
