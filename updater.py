"""GitHub Releases-based update checker that handles .exe installers and
archives containing a PyInstaller "onedir" build.

Behavior:
- Prefer a .exe installer asset (Inno Setup) and run it silently.
- If only an archive (.zip/.tar.gz) is available, download and extract it,
  then spawn a PowerShell script that uses robocopy to mirror files into
  the installation directory after the app exits.

This module is Windows-focused (PowerShell + robocopy)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import shutil
import zipfile
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


def _pick_asset_url(release_data: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return (url, kind) where kind is 'exe' or 'zip'. Prefer .exe, then .zip."""
    assets = release_data.get("assets") or []
    if not isinstance(assets, list):
        return None

    exe_url = None
    archive_url = None
    archive_exts = (".zip", ".tar.gz", ".tgz", ".tar")

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").lower()
        url = asset.get("browser_download_url")
        if not url:
            continue
        if name.endswith(".exe"):
            exe_url = url
            break
        if any(name.endswith(ext) for ext in archive_exts):
            archive_url = url

    if exe_url:
        return exe_url, "exe"
    if archive_url:
        return archive_url, "zip"
    return None


def perform_update(release_data: dict[str, Any], *, silent: bool = True) -> None:
    """Download asset, handle .exe installer or archive containing onedir files.

    For archives: extract to temp and launch a PowerShell script that copies files
    into the install dir (uses robocopy) after the app exits.
    """

    picked = _pick_asset_url(release_data)
    if not picked:
        raise RuntimeError("No suitable installer or archive asset found in the latest release.")

    download_url, kind = picked
    temp_dir = tempfile.gettempdir()

    headers = {"User-Agent": f"{REPO_NAME}-updater"}

    if kind == "exe":
        installer_path = os.path.join(temp_dir, f"{REPO_NAME}_Update.exe")
        r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
        r.raise_for_status()
        with open(installer_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        args = [installer_path]
        if silent:
            args += ["/SILENT", "/NORESTART"]

        subprocess.Popen(args, close_fds=True)
        sys.exit(0)

    # kind == "zip" (archive)
    archive_path = os.path.join(temp_dir, f"{REPO_NAME}_Update.zip")
    r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
    r.raise_for_status()
    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    extract_dir = os.path.join(temp_dir, f"{REPO_NAME}_Update_Extract")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    # Try to unpack archive
    try:
        shutil.unpack_archive(archive_path, extract_dir)
    except Exception:
        # Fallback to zipfile
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)

    # Determine install directory
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        # In development, assume current working dir is project root; prefer installation dir if known
        app_dir = os.path.abspath(os.getcwd())

    # Create PowerShell script to robocopy extracted files into app_dir
    ps_script = os.path.join(temp_dir, f"{REPO_NAME}_apply_update.ps1")
    src = extract_dir.replace('"', '""')
    dest = app_dir.replace('"', '""')
    exe_name = os.path.basename(sys.executable) if getattr(sys, "frozen", False) else "SearchANDGraph.exe"

    ps_contents = f'''$src = "{src}"
$dest = "{dest}"
Start-Sleep -s 1
# Mirror files from extracted dir into install dir
robocopy $src $dest /MIR /COPYALL /R:2 /W:2
# If an exe exists in dest, start it
$exe = Join-Path $dest "{exe_name}"
if (Test-Path $exe) {{ Start-Process -FilePath $exe }}
# Cleanup (best-effort)
Remove-Item -LiteralPath "{extract_dir}" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "{archive_path}" -Force -ErrorAction SilentlyContinue
'''

    with open(ps_script, "w", encoding="utf-8") as f:
        f.write(ps_contents)

    # Launch PowerShell script and exit so it can copy files
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script], close_fds=True)
    sys.exit(0)
