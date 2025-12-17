"""GitHub Releases-based update checker that installs updates via an .exe installer.

Behavior:
- Prefer a .exe installer asset (Inno Setup) and launch it elevated.
- The application does not apply file updates itself; the installer handles
    closing the app (via AppMutex/CloseApplications) and updating files.

This module is Windows-focused (PowerShell + Start-Process)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Optional, Callable

import requests
from packaging import version as pkg_version

from version import CURRENT_VERSION, REPO_NAME, REPO_OWNER


def check_for_updates(timeout_s: float = 5.0) -> Optional[dict[str, Any]]:
    """Return release JSON if a newer version exists; otherwise None."""

    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        headers = {"User-Agent": f"{REPO_NAME}"}
        response = requests.get(url, headers=headers, timeout=timeout_s)
        if response.status_code != 200:
            print(f"Update check failed: HTTP {response.status_code}")
            return None

        data = response.json()
        latest_tag = str(data.get("tag_name") or "").strip()
        if not latest_tag:
            print("Update check failed: no tag_name in response")
            return None

        latest_clean = latest_tag.lstrip("v").strip()
        if pkg_version.parse(latest_clean) > pkg_version.parse(CURRENT_VERSION):
            print(f"Update available: {CURRENT_VERSION} -> {latest_clean}")
            return data
    except Exception:
        return None

    return None


def _pick_asset_url(release_data: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return (url, kind) where kind is 'exe'.

    Prefer exact-named Inno installer 'SearchANDGraphSetup.exe' (case-insensitive),
    then any .exe. ZIP/archive updates are intentionally not supported.
    """
    assets = release_data.get("assets") or []
    if not isinstance(assets, list):
        return None

    exe_url = None

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").lower()
        url = asset.get("browser_download_url")
        if not url:
            continue
        # Prefer an exact installer filename first
        if name == f"{REPO_NAME.lower()}setup.exe":
            exe_url = url
            break
        if name.endswith(".exe") and not exe_url:
            exe_url = url

    if exe_url:
        return exe_url, "exe"
    return None


def perform_update(
    release_data: dict[str, Any], *, silent: bool = False, progress_callback: Optional[Callable[[int, str], None]] = None
) -> dict[str, Any]:
    """Download the installer asset and launch it elevated.

    When provided, `progress_callback` is called as `progress_callback(percent, message)`.
    Returns a dict describing the action taken.
    """

    picked = _pick_asset_url(release_data)
    if not picked:
        raise RuntimeError("No suitable installer (.exe) asset found in the latest release.")

    download_url, kind = picked
    temp_dir = tempfile.gettempdir()

    headers = {"User-Agent": f"{REPO_NAME}"}

    # Windows-only flag used to detach child processes from our console.
    # Define it unconditionally to avoid NameError in fallback paths.
    DETACHED_PROCESS = 0x00000008 if os.name == "nt" else 0

    # Prepare Windows-specific startup flags to avoid showing a console window
    win_startupinfo = None
    win_creationflags = 0
    if os.name == "nt":
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # 0 == SW_HIDE
            si.wShowWindow = 0
            win_startupinfo = si
            win_creationflags = subprocess.CREATE_NO_WINDOW
            # Detached process flag (don't keep child attached to our console)
            DETACHED_PROCESS = 0x00000008
        except Exception:
            win_startupinfo = None
            win_creationflags = 0

    if kind != "exe":
        raise RuntimeError("Only .exe installer updates are supported.")

    # Download installer to a unique temp file
    try:
        r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download installer: {e}")

    total = int(r.headers.get("content-length") or 0)
    written = 0
    if progress_callback:
        progress_callback(0, "Downloading installer...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".exe", prefix=f"{REPO_NAME}_Setup_", dir=temp_dir) as tf:
        installer_path = tf.name
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                continue
            tf.write(chunk)
            written += len(chunk)
            if total and progress_callback:
                pct = int(written * 100 / total)
                try:
                    progress_callback(min(pct, 99), "Downloading installer")
                except Exception:
                    pass

    # Make sure file is writable
    try:
        os.chmod(installer_path, 0o755)
    except Exception:
        pass

    # Use PowerShell to launch installer elevated so UAC is shown reliably.
    # IMPORTANT: we do not exit the app here; the installer handles closing via AppMutex.
    if silent:
        args = ["/VERYSILENT", "/NORESTART"]
    else:
        # Interactive install allows the end-of-install "Launch app" checkbox.
        args = ["/NORESTART"]

    arg_list = ",".join([f'"{a}"' for a in args])
    ps_command = (
        f'Start-Process -FilePath "{installer_path}" '
        f'-ArgumentList {arg_list} -Verb RunAs'
    )

    log_path = os.path.join(temp_dir, f"{REPO_NAME}_installer_launch.log")
    try:
        # Don't use DETACHED_PROCESS for the PowerShell launcher - it needs to show UAC
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
            capture_output=True,
            text=True,
            startupinfo=win_startupinfo,
            creationflags=win_creationflags,  # Remove DETACHED_PROCESS here
            close_fds=True,
        )
        
        # Write output to log
        with open(log_path, "w") as logf:
            logf.write(f"Return code: {proc.returncode}\n")
            logf.write(f"STDOUT:\n{proc.stdout}\n")
            logf.write(f"STDERR:\n{proc.stderr}\n")
        
        print(f"PowerShell output: {proc.stdout}")
        print(f"PowerShell errors: {proc.stderr}")
        
        if proc.returncode != 0:
            raise RuntimeError(f"Installer bootstrap failed (exit {proc.returncode}). See log: {log_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to launch installer: {e}. Log: {log_path}")

    if progress_callback:
        try:
            progress_callback(100, "Installer launched")
        except Exception:
            pass

    return {"action": "installer_launched", "installer": installer_path, "silent": bool(silent)}
